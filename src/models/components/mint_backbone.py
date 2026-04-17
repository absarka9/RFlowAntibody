"""MINT ESM2 backbone wrapper with multichain support.

Wraps the mint.model.esm2.ESM2 model (VarunUllanat/mint) and provides an
interface compatible with the existing RankFlow architecture while adding
multichain (chain_ids) support.

Checkpoint loading strips the 'model.' prefix from state_dict keys, as done
in mint/helpers/extract.py.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

LOG = logging.getLogger(__name__)


class MINTBackbone(nn.Module):
    """Wrapper around MINT's ESM2 model with multichain support.

    Exposes the same interface as fair-esm's ESM2 so that the rest of the
    RankFlow architecture can remain unchanged:
      - embed_dim, num_layers, attention_heads attributes
      - lm_head and emb_layer_norm_after properties
      - forward(tokens, chain_ids=None, repr_layers=None, need_head_weights=False)
        returning {'logits', 'representations', 'attentions'}

    Args:
        checkpoint_path: Path to a MINT checkpoint file that contains
            checkpoint['state_dict'] with 'model.' prefixed keys.
        use_multimer: Whether to enable MINT's multimer mode (required for
            multichain inputs).  Defaults to True.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        use_multimer: bool = True,
    ) -> None:
        super().__init__()

        try:
            from mint.model.esm2 import ESM2
            from mint.data import Alphabet
        except ImportError as exc:
            raise ImportError(
                "The 'mint' package is required for MINTBackbone.  "
                "Install it from https://github.com/VarunUllanat/mint"
            ) from exc

        self.alphabet = Alphabet.from_architecture("ESM-1b")
        # Register as a proper submodule so that parameter management works
        # correctly with nn.Module.named_parameters() / parameters().
        self._inner: nn.Module = ESM2(use_multimer=use_multimer)

        if checkpoint_path is not None:
            LOG.info("Loading MINT checkpoint from %s", checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            raw_sd = ckpt["state_dict"]
            # Strip leading 'model.' prefix as in mint/helpers/extract.py
            new_sd: Dict[str, torch.Tensor] = {}
            for k, v in raw_sd.items():
                new_k = k.removeprefix("model.")
                new_sd[new_k] = v
            missing, unexpected = self._inner.load_state_dict(new_sd, strict=False)
            if missing:
                LOG.warning("Missing keys when loading MINT checkpoint: %s", missing)
            if unexpected:
                LOG.warning("Unexpected keys when loading MINT checkpoint: %s", unexpected)

        # Expose scalar attributes expected by RankFlow
        self.embed_dim: int = self._inner.embed_dim
        self.num_layers: int = self._inner.num_layers
        self.attention_heads: int = self._inner.attention_heads
        self.mask_idx: int = self.alphabet.mask_idx

    # ------------------------------------------------------------------
    # Properties that forward to _inner (avoid duplicate module registration)
    # ------------------------------------------------------------------

    @property
    def lm_head(self) -> nn.Module:
        return self._inner.lm_head

    @property
    def emb_layer_norm_after(self) -> Optional[nn.Module]:
        return getattr(self._inner, "emb_layer_norm_after", None)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor,
        chain_ids: Optional[torch.Tensor] = None,
        repr_layers: Optional[List[int]] = None,
        need_head_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run MINT ESM2 forward pass.

        Args:
            tokens: (B, T) token indices.
            chain_ids: (B, T) integer chain identifiers (0-based). When None,
                a zero tensor is used (single-chain behaviour).
            repr_layers: Layer indices whose hidden states to return. When
                None, no representations are returned.
            need_head_weights: Whether to return per-head attention weights.

        Returns:
            Dictionary with keys 'logits', 'representations', and optionally
            'attentions' (shape B, num_layers, num_heads, T, T).
        """
        if chain_ids is None:
            chain_ids = torch.zeros_like(tokens)

        if repr_layers is None:
            repr_layers = []

        return self._inner(
            tokens,
            chain_ids=chain_ids,
            repr_layers=repr_layers,
            need_head_weights=need_head_weights,
        )
