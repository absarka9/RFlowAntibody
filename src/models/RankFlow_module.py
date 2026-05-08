from __future__ import annotations
import logging
import numpy as np
from scipy import stats
from typing import Any, Dict, List, Optional, Tuple
import random
import torch
import math
from math import atanh
import esm
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, PearsonCorrCoef
#import torchsort
from torch.nn import functional as F
from dataclasses import dataclass
from src.models.unet_1d import Unet1D

def soft_rank(x, tau=1.0):
    diff = x.unsqueeze(-1) - x.unsqueeze(-2) 
    P = torch.sigmoid(-diff / tau)

    return 1 + P.sum(dim=-1)

def spearmanr(pred, target, **kw):
    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    pred = soft_rank(pred, tau=0.5) 
    target = soft_rank(target, tau=0.5)

    pred = pred - pred.mean(dim=1, keepdim=True)
    pred = pred / (pred.std(dim=1, keepdim=True) + 1e-8)
    target = target - target.mean(dim=1, keepdim=True)
    target = target / (target.std(dim=1, keepdim=True) + 1e-8)

    return (pred * target).sum(dim=1).mean()

LOG = logging.getLogger(__name__)

from src.models.components.modules import TriangularSelfAttentionBlock

from torch import nn
from torch.nn import LayerNorm


class EmbFlowHead(nn.Module):
    """v_theta on embedding space: input/output (B, K, d_model)."""
    def __init__(self, rep_dim: int):
        super().__init__()
        self.rep_proj = nn.Linear(rep_dim, 32)
        self.rep_ln   = nn.LayerNorm(32)
        self.x_proj = nn.Linear(rep_dim, 32)
        self.x_ln   = nn.LayerNorm(rep_dim)

        self.pos_emb  = nn.Embedding(64, 16)
        self.aa_emb = nn.Embedding(20, 16)
        
        chan_in = rep_dim + 32 + 16
        self.net = Unet1D(dim=128, channels=chan_in, out_dim=rep_dim,
                          dim_mults=(1,2), dropout=0.1, fitness_pos=False)
        

    def forward(self, x0_embed, t, rep_ctx,
                mutant_pos_list, mutant_aa_list,
                add_noise: bool = True,
                site_gate=None,
                ):
        # x0_embed: (B,K,d), rep_ctx: (B,L,d) or (B,K,d) (we'll broadcast mean if needed)
        if x0_embed.ndim == 2: x0_embed = x0_embed[None, ...]
        if rep_ctx.ndim == 2: rep_ctx = rep_ctx[None, ...]
        B, K, d = x0_embed.shape
        max_pos = self.pos_emb.num_embeddings - 1

        idx0 = torch.clamp(mutant_pos_list.long(), 0, max_pos) 
        e = self.pos_emb(idx0) + self.aa_emb(torch.clamp(mutant_aa_list.long(), 0, 19))  

        cond = x0_embed.new_zeros(B, K, e.size(-1))
        cond.scatter_add_(1, idx0.unsqueeze(-1).expand_as(e), e) 

        # build context per site (broadcast a global mean)
        rep_c = self.rep_ln(self.rep_proj(rep_ctx))

        x_in = self.x_ln(x0_embed)
        if add_noise:
            std   = t.to(x_in.device).view(B,1,1)   # σ(t)=t
            alpha = 1.0 - t                         # μ(t)=1−t
            z     = torch.randn_like(x_in)
            x_t  = x_in * alpha + z * std          # VP path

            #x_cat = torch.cat([x_t, rep_c], dim=-1) 
            x_cat = torch.cat([x_t, rep_c, cond], dim=-1)
        else:
            z = None
            #x_cat = torch.cat([x_in, rep_c], dim=-1) 
            x_cat = torch.cat([x_in, rep_c, cond], dim=-1)
   
        v     = self.net(x_cat.permute(0,2,1), t.view(B)).permute(0,2,1)  # (B,K,d)
        if site_gate is not None:                   # (B,K,1)
            if site_gate.ndim == 2: site_gate = site_gate[None, ...]
            v = v * site_gate
        return v.squeeze(0), (z.squeeze(0) if z is not None else None), x_in.squeeze(0)
     
class RankFlow(LightningModule):
    def __init__(
        self,
        model,
        efm_beta,
        steps_ref,
        eta_ref,
        t0_ref,
        lambda_y,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        # MINT backbone options
        mint_checkpoint_path: Optional[str] = None,
        # Structure encoder options
        structure_encoder: str = "esm-if1",
        proteinmpnn_checkpoint_path: Optional[str] = None,
        proteinmpnn_hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.efm_beta     = efm_beta
        self.steps_ref    = steps_ref
        self.eta_ref      = eta_ref
        self.t0_ref       = t0_ref
        self.lambda_y     = lambda_y
        if model == 'esm2':
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        elif model == 'esm1v':
            self.model, self.alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
            self.model.embed_dim = self.model.args.embed_dim
            self.model.attention_heads = self.model.args.attention_heads
        elif model == 'mint':
            from src.models.components.mint_backbone import MINTBackbone
            self.model = MINTBackbone(
                checkpoint_path=mint_checkpoint_path,
                use_multimer=True,
            )
            self.alphabet = self.model.alphabet

        self.AA20 = "ACDEFGHIKLMNPQRSTVWY"

        canon_idx33 = [self.alphabet.get_idx(a) for a in self.AA20]  
        self.canon_idx33 = torch.tensor(canon_idx33, dtype=torch.long)

        self.idx33_to_aa20 = {int(self.canon_idx33[i]): i for i in range(20)}

        # Whether to use residue-aligned (special-token-free) tensors throughout.
        # True for MINT multichain; False for legacy ESM2/ESM1v single-chain.
        self.use_residue_aligned: bool = (model == 'mint')

        # ----------------------------------------------------------------
        # Structure encoder
        # ----------------------------------------------------------------
        if structure_encoder == 'proteinmpnn':
            from src.models.components.proteinmpnn_encoder import ProteinMPNNEncoder
            self.structure_model = ProteinMPNNEncoder(
                checkpoint_path=proteinmpnn_checkpoint_path,
                hidden_dim=proteinmpnn_hidden_dim,
            )
            d_struct = self.structure_model.hidden_dim
            self.structure_encoder_type = 'proteinmpnn'
        else:
            self.structure_model, self.structure_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
            d_struct = self.structure_model.encoder.args.encoder_embed_dim
            self.structure_encoder_type = 'esm-if1'

        self.model, self.structure_model = self.model.eval(), self.structure_model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        for name, param in self.structure_model.named_parameters():
            param.requires_grad = False

        self.state = {}
        
        self.val_spearman = MeanMetric()
        self.val_spearman_best = MaxMetric()
        self.criterion = PearsonCorrCoef()

        self.automatic_optimization = False

        self.repr_dim = self.model.embed_dim
        self.repr_combine = nn.Parameter(torch.zeros(self.model.num_layers + 1))
        self.repr_mlp = nn.Sequential(LayerNorm(self.repr_dim), nn.Linear(self.repr_dim, self.repr_dim), nn.GELU(), nn.Linear(self.repr_dim, self.repr_dim))
        self.structure_repr_mlp = nn.Sequential(LayerNorm(d_struct), nn.Linear(d_struct, self.repr_dim), nn.GELU(), nn.Linear(self.repr_dim, self.repr_dim))
        self.attn_dim = 32
        self.attn_num = self.model.num_layers * self.model.attention_heads
        self.attn_mlp = nn.Sequential(LayerNorm(self.attn_num), nn.Linear(self.attn_num, self.attn_num), nn.GELU(), nn.Linear(self.attn_num, self.attn_dim))

        self.count_parameters(self.repr_mlp)
        self.count_parameters(self.structure_repr_mlp)
        self.count_parameters(self.attn_mlp)

        self.num_blocks = 1
        self.blocks = nn.ModuleList(
            [
                TriangularSelfAttentionBlock(
                    sequence_state_dim=self.repr_dim,
                    pairwise_state_dim=self.attn_dim,
                    sequence_head_width=32,
                    pairwise_head_width=32,
                    dropout=0.2
                ) for _ in range(self.num_blocks)
            ]
        )
        self.count_parameters(self.blocks)

        self.emb_flow_head = EmbFlowHead(rep_dim=self.repr_dim)    
        self.count_parameters(self.emb_flow_head)

        self.assay_stats = {} 
        self.delta_mix_logit = nn.Parameter(torch.tensor(0.0))

    def _encode_structure(self, coord) -> torch.Tensor:
        """Encode structure using the active structure encoder.

        For ProteinMPNN: coord is (X, mask, residue_idx, chain_encoding_all).
        For ESM-IF1:     coord is (coord_tensor, padding_mask, confidence).

        Returns:
            structure_repr: (1, L, d_struct) detached tensor.
        """
        with torch.no_grad():
            if self.structure_encoder_type == 'proteinmpnn':
                X, mask, residue_idx, chain_encoding_all = coord
                h_V = self.structure_model.encode(X, mask, residue_idx, chain_encoding_all)
                return h_V.detach()  # (B, L, hidden_dim)
            else:
                # ESM-IF1: coord = (coord_tensor, padding_mask, confidence)
                return (
                    self.structure_model.encoder.forward(*coord)['encoder_out'][0]
                    .detach()
                    .transpose(0, 1)
                )  # (B, L, encoder_dim)

    def _aa20_cols(self, device):
        self._ensure_vocab_buffers()
        return self.canon_idx33.to(device)

    # ------------------------------------------------------------------
    # Residue-alignment helpers (used with MINT multichain)
    # ------------------------------------------------------------------

    def _build_res_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask (B, T) that is True at residue positions.

        Special tokens (CLS, EOS, PAD) are False; all standard amino-acid
        tokens are True.  Works for single-chain and multichain sequences.
        """
        cls_idx = getattr(self.alphabet, "cls_idx", None)
        eos_idx = getattr(self.alphabet, "eos_idx", None)
        pad_idx = getattr(self.alphabet, "padding_idx", None)
        special = torch.zeros_like(tokens, dtype=torch.bool)
        if cls_idx is not None:
            special |= tokens == cls_idx
        if eos_idx is not None:
            special |= tokens == eos_idx
        if pad_idx is not None:
            special |= tokens == pad_idx
        return ~special  # True = residue position

    def _gather_res_1d(self, tensor: torch.Tensor, res_idx: torch.Tensor) -> torch.Tensor:
        """Select positions in a (B, T, ...) tensor using 1-D residue indices.

        Args:
            tensor:  (B, T, ...)
            res_idx: (L,) 1-D integer tensor of token positions to keep

        Returns:
            (B, L, ...)
        """
        expand_shape = (tensor.shape[0],) + (len(res_idx),) + tensor.shape[2:]
        idx = res_idx.to(tensor.device)
        # Expand idx to match all trailing dimensions
        for _ in range(tensor.dim() - 2):
            idx = idx.unsqueeze(-1)
        idx = idx.unsqueeze(0).expand(expand_shape)
        return tensor.gather(1, idx)

    def _gather_res_attn(
        self, attn: torch.Tensor, res_idx: torch.Tensor
    ) -> torch.Tensor:
        """Select residue positions on both token axes of attention (B, T, T, C).

        Args:
            attn:    (B, T, T, C)
            res_idx: (L,) integer positions to keep

        Returns:
            (B, L, L, C)
        """
        idx = res_idx.to(attn.device)
        attn = attn[:, idx, :, :]   # (B, L, T, C)
        attn = attn[:, :, idx, :]   # (B, L, L, C)
        return attn

    def _decode_logits20_from_embed(self, H):  # H: (K,d) or (1,K,d)
        if H.ndim == 2:
            H = H.unsqueeze(0)
        # ESM has a final norm before the LM head; use it if present
        ln = getattr(self.model, "emb_layer_norm_after", None) or getattr(self.model, "layer_norm", None)
        Hn = ln(H) if ln is not None else H
        logits33 = self.model.lm_head(Hn)              # (1,K,33)
        return logits33[0, :, self._aa20_cols(H.device)] 
    
    def count_parameters(self, module: nn.Module):
        print(f"{sum(p.numel() for p in module.parameters() if p.requires_grad)/1e6:.2f}M")
    
    def state_setup(self, masked_x_mt, y_label, chain_ids=None):
        y = []
        locations = set()
        for item in y_label:
            score, mutants = item[0], item[1]
            y.append({
                'score': score,
                'mutants': [(mutant[0], mutant[1], mutant[2]) for mutant in mutants]
            })
            for mutant in mutants:
                locations.add(mutant[0])

        masked_batch_tokens = masked_x_mt.clone()
        with torch.no_grad():
            kwargs = {}
            if self.use_residue_aligned and chain_ids is not None:
                kwargs['chain_ids'] = chain_ids
            result = self.model(
                masked_batch_tokens,
                repr_layers=range(self.model.num_layers + 1),
                need_head_weights=True,
                **kwargs,
            )

        # Stack representations: (B, T, num_layers+1, d)
        rep_full = torch.stack(
            [v.detach() for _, v in sorted(result['representations'].items())], dim=2
        )
        # Logits: (B, T, vocab) → take first batch item
        logits_full = result['logits'].detach()[0]       # (T, vocab)
        # Attentions: MINT returns (B, L, H, T, T); permute → (B, T, T, L*H)
        attn_full = (
            result['attentions']
            .detach()
            .permute(0, 4, 3, 1, 2)
            .flatten(3, 4)
        )  # (B, T, T, num_layers*heads)

        if self.use_residue_aligned:
            res_mask = self._build_res_mask(masked_batch_tokens)  # (B, T)
            res_idx = res_mask[0].nonzero(as_tuple=True)[0]       # (L,)
            rep_full   = self._gather_res_1d(rep_full, res_idx)    # (B, L, n_layers+1, d)
            logits_full = logits_full[res_idx]                     # (L, vocab)
            attn_full  = self._gather_res_attn(attn_full, res_idx) # (B, L, L, n_layers*heads)

        x = {
            'input': masked_x_mt,
            'logits': logits_full,
            'representation': rep_full,
            'attention': attn_full,
        }
        if self.use_residue_aligned:
            x['res_idx'] = res_idx  # store for downstream use
        return (x, y)

    def forward(self, x, structure_repr):
        representation, attention = x['representation'], x['attention'] 

        if self.use_residue_aligned:
            # Representations and attention are already residue-aligned (no special tokens).
            # residx is just 0..L-1; mask is all-ones.
            L = representation.shape[1]
            residx = torch.arange(L, device=self.device).unsqueeze(0).expand(
                representation.shape[0], -1
            )
            mask = torch.ones(representation.shape[0], L, device=self.device)
        else:
            residx = torch.arange(x['input'].shape[1], device=self.device).expand_as(x['input'])
            mask = torch.ones_like(x['input'])

        representation = self.repr_mlp((self.repr_combine.softmax(0).unsqueeze(0) @ representation).squeeze(2)) + self.structure_repr_mlp(structure_repr).repeat(representation.shape[0], 1, 1)
        attention = self.attn_mlp(attention)

        def trunk_iter(s, z, residx, mask):
            for block in self.blocks:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=None)
            return s, z
        
        representation, _ = trunk_iter(representation, attention, residx, mask)

        return representation
        
    @torch.no_grad()
    def _get_rep_logits(self, tokens, *, repr_layer=33, chain_ids=None):
        kwargs = {}
        if self.use_residue_aligned and chain_ids is not None:
            kwargs['chain_ids'] = chain_ids
        out = self.model(
            tokens,
            repr_layers=range(self.model.num_layers + 1),
            need_head_weights=True,
            **kwargs,
        )

        rep = torch.stack(
            [v.detach() for _, v in sorted(out['representations'].items())], dim=2
        )  # (B, T, num_layers+1, d)

        # Attentions: MINT/ESM both return (B, num_layers, num_heads, T, T)
        # Permute → (B, T, T, num_layers*heads)
        attn = out['attentions'].detach().permute(0, 4, 3, 1, 2).flatten(3, 4)

        if self.use_residue_aligned:
            res_mask = self._build_res_mask(tokens)            # (B, T)
            res_idx  = res_mask[0].nonzero(as_tuple=True)[0]  # (L,)
            rep      = self._gather_res_1d(rep, res_idx)       # (B, L, n_layers+1, d)
            logits33 = out["logits"].detach()[0][res_idx]       # (L, vocab)
            attn     = self._gather_res_attn(attn, res_idx)    # (B, L, L, n*h)
            last_rep = out["representations"][repr_layer].detach()[0][res_idx]  # (L, d)
        else:
            logits33 = out["logits"][0, 1:-1, :]               # (L, vocab)
            attn     = attn                                     # (B, T, T, n*h)
            last_rep = out["representations"][repr_layer][0, 1:-1, :]  # (L, d)

        return rep, logits33, attn, last_rep
    
    def _make_aa_feat(self, L, mutants, device):
        aa_feat = torch.zeros(1, L, 16, device=device)
        if len(mutants) == 0: 
            return aa_feat
        rows, aa = [], []
        for (pos1, wt33, mut33) in mutants:
            rows.append(int(pos1) - 1)
            aa.append(int(self.idx33_to_aa20[int(mut33)]))
        rows = torch.tensor(rows, device=device, dtype=torch.long)
        aa   = torch.tensor(aa,   device=device, dtype=torch.long)
        aa_feat[0, rows, :] = self.emb_flow_head.aa_emb(aa)
        return aa_feat


    def loss_compute_and_backward(self, x, y, structure_repr, msa_bank, name=None):
        opt = self.optimizers()
        device = next(self.model.parameters()).device

        # Tune these two depending on GPU memory
        chunk_size = 8          # smaller -> less memory per forward
        grad_acc_steps = 8       # number of items to accumulate before optimizer.step()

        # Global stats (for logging, not strictly needed for grad)
        global_num_acc = torch.tensor(0.0, device=device)
        global_den_acc = torch.tensor(0.0, device=device)

        mu, sigma = self.assay_stats[name]

        # Wild-type tokens
        wt_tokens_master = self.state[f"{name}-wt_tokens"].to(device)

        # Compute L (number of residues) correctly for single- and multi-chain.
        if self.use_residue_aligned:
            # res_idx was stored in state_setup; L = number of residue positions
            res_mask_wt = self._build_res_mask(wt_tokens_master)
            _res_idx_wt = res_mask_wt[0].nonzero(as_tuple=True)[0]
            L = int(_res_idx_wt.shape[0])
            chain_ids_master = self.state.get(f"{name}-chain_ids")
        else:
            Lp2 = wt_tokens_master.shape[1]  # L+2 for single-chain
            L = Lp2 - 2
            chain_ids_master = None

        # For rank loss accumulation over windows
        window_num_acc = torch.tensor(0.0, device=device)
        window_den_acc = torch.tensor(0.0, device=device)
        window_scores_hat = []
        window_scores_true = []
        window_count = 0

        # Ensure we start with clean grads
        opt.zero_grad(set_to_none=True)

        num_chunks = (len(y) + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, len(y))
            chunk = y[start:end]

            for index in range(len(chunk)):
                item = chunk[index]
                tok = wt_tokens_master.clone()  
                mask_idx = self.alphabet.mask_idx

                # item['mutants']: list of tuples (pos1, wt33, mut33)
                # pos1 is the 1-indexed token position in the full token sequence.
                # For residue-aligned mode: convert to 0-indexed residue row by
                # counting residue positions before pos1 in the mask.
                # For legacy mode: residue row = pos1 - 1 (one CLS token offset).
                mut_rows, wt20_list, mut20_list = [], [], []
                for (pos1, wt33, mut33) in item['mutants']:
                    pos1 = int(pos1)
                    tok[0, pos1] = mask_idx
                    # For residue-aligned mode, compute the residue row from
                    # the number of valid residue positions before pos1.
                    if self.use_residue_aligned:
                        res_mask_tok = self._build_res_mask(tok)
                        # Count residues before this token position
                        row = int(res_mask_tok[0, :pos1].sum().item())
                    else:
                        row = pos1 - 1
                    mut_rows.append(row)
                    wt20_list.append(int(self.idx33_to_aa20[int(wt33)]))
                    mut20_list.append(int(self.idx33_to_aa20[int(mut33)]))

                rows = torch.tensor(mut_rows, device=device, dtype=torch.long)
                if rows.numel() == 0:
                    continue

                # Get base reps / logits from the sequence model
                chain_ids_tok = chain_ids_master
                rep_L0, logits33_L, attn, last_rep = self._get_rep_logits(
                    tok, chain_ids=chain_ids_tok
                )

                aa20 = self._aa20_cols(device)
                base_logits20 = logits33_L[:, aa20].index_select(0, rows)

                # Contextual representation with structure
                rep_L = self.forward(
                    {'input': tok, 'representation': rep_L0, 'attention': attn},
                    structure_repr=self.state[f'{name}-structure']
                )
                # For residue-aligned mode, forward() already returns residue-only
                # (B, L, d); for legacy mode, strip CLS/EOS here.
                if self.use_residue_aligned:
                    rep_L = rep_L[0]        # (L, d)
                else:
                    rep_L = rep_L[0, 1:-1, :]  # strip CLS/SEP
                rep_ctx = rep_L.unsqueeze(0)  

                # Energy / weight
                y_score = torch.as_tensor(item['score'], device=device, dtype=torch.float32)
                norm = (y_score - mu) / (sigma + 1e-8)
                E = self.lambda_y * y_score + (1 - self.lambda_y) * ((y_score - mu) / (sigma + 1e-8))
                w = torch.exp(self.efm_beta * (-E)).detach()  

                t = torch.rand(1, 1, device=device)

                # Flow head inputs
                x0_embed = last_rep  
                m = torch.zeros(L, 1, device=device)
                m[rows, 0] = 1.0

                mut20 = torch.tensor(mut20_list, device=device, dtype=torch.long)

                v_pred, z, x_in = self.emb_flow_head(
                    x0_embed, t, rep_ctx,
                    site_gate=m, add_noise=True,
                    mutant_pos_list=rows.view(1, -1),
                    mutant_aa_list=mut20.view(1, -1),
                )

                # EFM (flow) loss 
                loss_i = (v_pred - (z - x_in)).pow(2).mean(dim=-1).mean() 

                # Accumulate EF loss (global stats + window stats)
                global_num_acc = global_num_acc + w * loss_i.detach()
                global_den_acc = global_den_acc + w

                window_num_acc = window_num_acc + w * loss_i
                window_den_acc = window_den_acc + w

                # Refinement steps (from x0)
                refined_H = x0_embed.clone()
                if self.steps_ref > 0:
                    t_fixed = torch.tensor([[self.t0_ref]], device=device)
                    for _ in range(self.steps_ref):
                        v1, _, _ = self.emb_flow_head(
                            refined_H, t_fixed, rep_ctx,
                            mutant_pos_list=rows.view(1, -1),
                            mutant_aa_list=mut20.view(1, -1),
                            site_gate=m, add_noise=False
                        )
                        x_e = refined_H - self.eta_ref * v1
                        v2, _, _ = self.emb_flow_head(
                            x_e, t_fixed, rep_ctx,
                            mutant_pos_list=rows.view(1, -1),
                            mutant_aa_list=mut20.view(1, -1),
                            site_gate=m, add_noise=False
                        )
                        refined_H = refined_H - 0.5 * self.eta_ref * (v1 + v2)

                # Decode logits and compute advantage difference
                ref_logits20 = self._decode_logits20_from_embed(refined_H)  # (L, 20)
                ref_lp = F.log_softmax(ref_logits20.index_select(0, rows), dim=-1)
                base_lp = F.log_softmax(base_logits20, dim=-1)

                base_vals, delta_vals = [], []
                for r, (w20, m20) in enumerate(zip(wt20_list, mut20_list)):
                    if w20 >= 0 and m20 >= 0:
                        base_adv = base_lp[r, m20] - base_lp[r, w20]
                        refined_adv = ref_lp[r, m20] - ref_lp[r, w20]
                        base_vals.append(base_adv)
                        delta_vals.append(refined_adv - base_adv)

                if not base_vals:
                    continue

                base_score = torch.stack(base_vals).mean()
                delta_score = torch.stack(delta_vals).mean()
                gamma = torch.sigmoid(self.delta_mix_logit)
                pred_score = base_score + gamma * delta_score  # scalar tensor

                # prediction/target for rank loss
                window_scores_hat.append(pred_score)         # keep as tensor (no float())
                window_scores_true.append(norm)                    # also tensor

                window_count += 1

                # ---- Gradient accumulation window ----
                if window_count % grad_acc_steps == 0:
                    # EFM loss for this window
                    ef_loss = window_num_acc / (window_den_acc + 1e-8)

                    # Rank loss for this window
                    if window_scores_hat:
                        scores_hat_batch = torch.stack(window_scores_hat, dim=0)      # (Nwin,)
                        scores_true_batch = torch.stack(window_scores_true, dim=0)    # (Nwin,)
                        loss_rank = 1.0 - spearmanr(scores_hat_batch, scores_true_batch)
                    else:
                        loss_rank = torch.tensor(0.0, device=device)

                    total_loss = 1.0 * ef_loss + 0.5 * loss_rank

                    # Backward with normalized loss to keep scale stable
                    (total_loss / 1.0).backward()

                    opt.step()
                    opt.zero_grad(set_to_none=True)

                    # Reset window accumulators
                    window_num_acc = torch.tensor(0.0, device=device)
                    window_den_acc = torch.tensor(0.0, device=device)
                    window_scores_hat = []
                    window_scores_true = []

                    # Optional: free some references (not strictly needed)
                    del ef_loss, loss_rank, total_loss

            # end for index in chunk
        # end for chunk_idx in range(num_chunks)

        # Handle leftover items in the last window (if any)
        if window_den_acc.item() > 0:
            ef_loss = window_num_acc / (window_den_acc + 1e-8)
            if window_scores_hat:
                scores_hat_batch = torch.stack(window_scores_hat, dim=0)
                scores_true_batch = torch.stack(window_scores_true, dim=0)
                loss_rank = 1.0 - spearmanr(scores_hat_batch, scores_true_batch)
            else:
                loss_rank = torch.tensor(0.0, device=device)

            total_loss = 1.0 * ef_loss + 0.5 * loss_rank
            total_loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        # You can return global stats if you want to log them
        final_loss = global_num_acc / (global_den_acc + 1e-8)
        return final_loss, None

    def output_process(self, x_logits, y):
        x_logits = torch.stack([sum([x_logits[mutant[0], mutant[2]] - x_logits[mutant[0], mutant[1]] for mutant in y[index]['mutants']]) / len(y[index]['mutants']) for index in range(len(y))], dim=-1)
        y_scores = torch.stack([y[index]['score'] for index in range(len(y))])
        return x_logits, y_scores
    
    def on_train_start(self) -> None:
        self.val_spearman.reset()
        self.val_spearman_best.reset()
    
    def seq_mean_acts(self, tokens, return_logits=False, chain_ids=None):
        with torch.no_grad():
            repr_layers=[32]
            kwargs = {}
            if self.use_residue_aligned and chain_ids is not None:
                kwargs['chain_ids'] = chain_ids
            out = self.model(tokens, repr_layers=repr_layers, **kwargs)
            h = torch.stack([v.detach() for _, v in sorted(out['representations'].items())], dim=2)         
            logits = out['logits'].detach()[0]
        if return_logits == True:
            return h, logits   

        return h         
    
    def seq_acts(self, tokens, chain_ids=None):
        with torch.no_grad():
            kwargs = {}
            if self.use_residue_aligned and chain_ids is not None:
                kwargs['chain_ids'] = chain_ids
            result = self.model(
                tokens,
                repr_layers=range(self.model.num_layers + 1),
                need_head_weights=True,
                **kwargs,
            )
        # Attentions: (B, num_layers, num_heads, T, T) → (B, T, T, num_layers*heads)
        attn = result['attentions'].detach().permute(0, 4, 3, 1, 2).flatten(3, 4)
        if self.use_residue_aligned:
            res_mask = self._build_res_mask(tokens)
            res_idx  = res_mask[0].nonzero(as_tuple=True)[0]
            rep = self._gather_res_1d(
                torch.stack([v.detach() for _, v in sorted(result['representations'].items())], dim=2),
                res_idx,
            )
            logits = result['logits'].detach()[0][res_idx]
            attn = self._gather_res_attn(attn, res_idx)
            x = {'input': tokens, 'logits': logits, 'representation': rep, 'attention': attn, 'res_idx': res_idx}
        else:
            x = {
                'input': tokens,
                'logits': result['logits'].detach()[0],
                'representation': torch.stack([v.detach() for _, v in sorted(result['representations'].items())], dim=2),
                'attention': attn,
            }
        return x   
    
    def _ensure_vocab_buffers(self):
        # Build once: 33-vocab → AA20 columns, and 33→20 index map
        if hasattr(self, "canon_idx33") and hasattr(self, "idx33_to_aa20"):
            return
        AA20 = "ACDEFGHIKLMNPQRSTVWY"
        canon_idx33 = torch.tensor([self.alphabet.tok_to_idx[a] for a in AA20], dtype=torch.long)
        id2aa20 = torch.full((len(self.alphabet.tok_to_idx),), -1, dtype=torch.long)
        for col, a in enumerate(AA20):
            id2aa20[self.alphabet.tok_to_idx[a]] = col
        self.register_buffer("canon_idx33", canon_idx33, persistent=False)
        self.register_buffer("idx33_to_aa20", id2aa20, persistent=False)


    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # Batch may include chain_ids (multichain) or not (single-chain legacy).
        if len(batch) == 9:
            assay_names, batch_tokens, chain_ids_list, coords, train_labels, _, msa_bank, higher_mutseqs_100, lower_mutseqs_100 = batch
        else:
            assay_names, batch_tokens, coords, train_labels, _, msa_bank, higher_mutseqs_100, lower_mutseqs_100 = batch
            chain_ids_list = [None] * len(assay_names)

        for name, x_wt, chain_ids, coord, train_label in zip(assay_names, batch_tokens, chain_ids_list, coords, train_labels):
            key = f'{name}-steer_v'
            if key not in self.state:
                with torch.no_grad():
                    Hpos = torch.stack([self.seq_mean_acts(t, return_logits=False) for t in higher_mutseqs_100]).mean(dim=0)
                    Hneg = torch.stack([self.seq_mean_acts(t, return_logits=False) for t in lower_mutseqs_100]).mean(dim=0)
                steer_v = (Hpos - Hneg).detach()
                self.state[f"{name}-steer_v"] = steer_v.squeeze() 
            
            if f'{name}-train' not in self.state:
                self.state[f'{name}-train'] = self.state_setup(x_wt, train_label, chain_ids=chain_ids)
            if f'{name}-structure' not in self.state:
                with torch.no_grad():
                    self.state[f'{name}-structure'] = self._encode_structure(coord)
            x, y = self.state[f'{name}-train']

            key_wt = f"{name}-wt_tokens"
            if key_wt not in self.state:
                self.state[key_wt] = x_wt.clone().to(x_wt.device)
            if chain_ids is not None:
                self.state[f"{name}-chain_ids"] = chain_ids.clone().to(x_wt.device)

            if (name not in self.assay_stats) or (len(self.assay_stats[name]) < 3):
                ys = torch.tensor([float(it["score"]) for it in y], device=x_wt.device)
                mu    = ys.mean().item()
                sigma = ys.std(unbiased=False).clamp_min(1e-8).item()
                self.assay_stats[name] = (mu, sigma)
                
            loss, _ = self.loss_compute_and_backward(x, y, structure_repr=self.state[f'{name}-structure'], msa_bank=msa_bank, name=name)
            LOG.info(f'Training assay {name}: loss {loss}')
    
    def on_validation_epoch_start(self):
        self.val_spearman.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        # Batch may include chain_ids (multichain) or not (single-chain legacy).
        if len(batch) == 9:
            assay_names, batch_tokens, chain_ids_list, coords, _, valid_labels, msa_bank, _, _ = batch
        else:
            assay_names, batch_tokens, coords, _, valid_labels, msa_bank, _, _ = batch
            chain_ids_list = [None] * len(assay_names)

        device = batch_tokens[0].device
        self._ensure_vocab_buffers()

        for name, x_wt, chain_ids, coord, valid_label in zip(assay_names, batch_tokens, chain_ids_list, coords, valid_labels):
            if f'{name}-valid' not in self.state:
                self.state[f'{name}-valid'] = self.state_setup(x_wt, valid_label, chain_ids=chain_ids)
            if f'{name}-structure' not in self.state:
                with torch.no_grad():
                    self.state[f'{name}-structure'] = self._encode_structure(coord)
            x, y = self.state[f'{name}-valid']
            key_wt = f"{name}-wt_tokens"
            if key_wt not in self.state:
                self.state[key_wt] = x_wt.clone().to(device)
            if chain_ids is not None:
                self.state.setdefault(f"{name}-chain_ids", chain_ids.clone().to(device))
            wt_tokens_master = self.state[key_wt]
            chain_ids_master = self.state.get(f"{name}-chain_ids")
            mu, sigma = self.assay_stats.get(name, (0.0, 1.0))

            scores_hat_all, scores_true_all = [], []

            for sample in y:
                mutants = sample['mutants']

                tok = wt_tokens_master.clone()             
                mask_idx = self.alphabet.mask_idx

                rows_list, wt20_list, mut20_list = [], [], []
                for (pos1, wt33, mut33) in mutants:
                    pos1 = int(pos1)
                    tok[0, pos1] = mask_idx
                    if self.use_residue_aligned:
                        res_mask_tok = self._build_res_mask(tok)
                        row = int(res_mask_tok[0, :pos1].sum().item())
                    else:
                        row = pos1 - 1
                    rows_list.append(row)
                    wt20  = int(self.idx33_to_aa20[int(wt33)])
                    mut20 = int(self.idx33_to_aa20[int(mut33)])
                    wt20_list.append(wt20); mut20_list.append(mut20)

                rows = torch.tensor(rows_list, device=device, dtype=torch.long)
                if rows.numel() == 0:
                    continue  

                rep_L, logits33_L, attn, last_rep = self._get_rep_logits(
                    tok, chain_ids=chain_ids_master
                )

                rep_L = self.forward(
                    {'input': tok, 'representation': rep_L, 'attention': attn},
                    structure_repr=self.state[f'{name}-structure'],
                ).detach()
                if self.use_residue_aligned:
                    rep_L = rep_L[0]        # (L, d)
                else:
                    rep_L = rep_L[0, 1:-1, :]

                aa20 = self._aa20_cols(device)
                base_logits20 = logits33_L[:, aa20].index_select(0, rows)  # (K,20)
                x0_embed      = last_rep
                K = x0_embed.size(0)

                if self.use_residue_aligned:
                    mut_rows = sorted({r for r in rows_list})
                else:
                    mut_rows = sorted({int(pos) - 1 for (pos, _wt, _mut) in mutants})
                idx = torch.as_tensor(mut_rows, device=device, dtype=torch.long)
                m = torch.zeros(K, 1, device=device, dtype=x0_embed.dtype)
                m[idx] = 1.0

                rep_ctx = rep_L.unsqueeze(0)   # (1,L,d)

                mut20 = torch.tensor(mut20_list, device=device, dtype=torch.long)
                refined_H = x0_embed.clone()
                if self.steps_ref > 0:
                    t_fixed = torch.tensor([[self.t0_ref]], device=device)
                    for _ in range(self.steps_ref):
                        v1, _, _ = self.emb_flow_head(
                            refined_H, t_fixed, rep_ctx,
                            mutant_pos_list=idx.view(1,-1),
                            mutant_aa_list=mut20.view(1,-1),
                            site_gate=m, add_noise=False
                        )
                        x_e = refined_H - self.eta_ref * v1
                        v2, _, _ = self.emb_flow_head(x_e, t_fixed, rep_ctx, 
                                                      mutant_pos_list=idx.view(1,-1),
                                                      mutant_aa_list=mut20.view(1,-1),  
                                                      site_gate=m, add_noise=False)
                        refined_H = refined_H - 0.5 * self.eta_ref * (v1 + v2)
                with torch.no_grad():
                    ref_logits20 = self._decode_logits20_from_embed(refined_H) 

                ref_logits20 = ref_logits20.index_select(0, rows)
                ref_lp  = F.log_softmax(ref_logits20, dim=-1)
                base_lp = F.log_softmax(base_logits20, dim=-1)

                base_vals, delta_vals = [], []
                for r, (wt20, mut20) in enumerate(zip(wt20_list, mut20_list)):
                    if wt20 >= 0 and mut20 >= 0:
                        base_adv_r  = base_lp[r, mut20] - base_lp[r, wt20]
                        refined_adv = ref_lp[r, mut20]  - ref_lp[r, wt20]
                        delta_adv_r = refined_adv - base_adv_r
                        base_vals.append(base_adv_r)
                        delta_vals.append(delta_adv_r)

                if not base_vals:
                    continue

                base_score  = torch.stack(base_vals).mean()
                delta_score = torch.stack(delta_vals).mean()

                gamma = torch.sigmoid(self.delta_mix_logit)
                pred_score = base_score + gamma * delta_score

                # orientation-aware prediction
                mu, sigma = self.assay_stats.get(name, (0.0, 1.0))
                scores_hat_all.append(pred_score.item())
                scores_true_all.append((float(sample['score']) - mu) / (sigma + 1e-8))

            spearman = stats.spearmanr(np.asarray(scores_hat_all,  dtype=np.float64), np.asarray(scores_true_all, dtype=np.float64)).statistic
   
            LOG.info(f'Testing assay {name}: spearman {spearman}')
            self.val_spearman(spearman)
    
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        self.log("val/spearman", self.val_spearman.compute(), sync_dist=True, prog_bar=True)

        spearman = self.val_spearman.compute() 
        self.val_spearman_best(spearman) 

        self.log("val/spearman_best", self.val_spearman_best.compute(), sync_dist=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            # compile the actual model that does the work
            self.emb_flow_head.net = torch.compile(self.emb_flow_head.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/spearman_best",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
    @torch.no_grad()
    def _compute_assay_sign_like_validation(self, wt_tokens, y, chain_ids=None):
        self._ensure_vocab_buffers()
        device   = wt_tokens.device
        mask_idx = self.alphabet.mask_idx

        hat, tru = [], []
        for item in y[:len(y)]:  # small slice is fine
            tok = wt_tokens.clone()
            rows, wt20, mut20 = [], [], []
            for (pos1, wt33, mut33) in item['mutants']:     # pos1 is 1-based
                p = int(pos1)
                tok[0, p] = mask_idx
                if self.use_residue_aligned:
                    res_mask_tok = self._build_res_mask(tok)
                    row = int(res_mask_tok[0, :p].sum().item())
                else:
                    row = p - 1                             # row index in (L,·)
                rows.append(row)
                wt20.append(int(self.idx33_to_aa20[int(wt33)]))
                mut20.append(int(self.idx33_to_aa20[int(mut33)]))
            if not rows: 
                continue

            _, logits33_L, _, _ = self._get_rep_logits(tok, chain_ids=chain_ids)
            aa20 = self._aa20_cols(device)
            base_lp = F.log_softmax(logits33_L[:, aa20].index_select(
                0, torch.tensor(rows, device=device)
            ), dim=-1)                                      # (K,20)

            vals = []
            for r, (w, m) in enumerate(zip(wt20, mut20)):
                if w >= 0 and m >= 0:
                    vals.append(base_lp[r, m] - base_lp[r, w])
            if not vals:
                continue

            hat.append(torch.stack(vals).mean().item())
            tru.append(float(item['score']))

        if len(hat) < 3:
            return 1

        # Spearman sign (torch-only)
        hx = torch.tensor(hat, dtype=torch.float32, device=device)
        hy = torch.tensor(tru, dtype=torch.float32, device=device)
        rx = torch.argsort(torch.argsort(hx)).float()
        ry = torch.argsort(torch.argsort(hy)).float()
        rx = (rx - rx.mean()) / (rx.std(unbiased=False) + 1e-8)
        ry = (ry - ry.mean()) / (ry.std(unbiased=False) + 1e-8)
        rho = float((rx * ry).mean().item())
        return 1 if rho >= 0 else -1

