"""ProteinMPNN-based structure encoder wrapper.

Replaces the ESM-IF1 encoder with the ProteinMPNN encoder stack
(dauparas/ProteinMPNN).  The wrapper:

  - Loads a v_48_XXX.pt ProteinMPNN checkpoint via checkpoint['model_state_dict'].
  - Exposes an encode() method that returns per-residue node features
    h_V of shape (B, L, hidden_dim) from the encoder stack.
  - Uses hidden_dim = 128 by default (d_struct for structure_repr_mlp).

ProteinMPNN featurisation inputs (X, mask, residue_idx, chain_encoding_all)
can be precomputed from a PDB file using the utilities in this module
(featurize_pdb / featurize_chains).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PDB featurisation helpers (no ProteinMPNN import required)
# ---------------------------------------------------------------------------

_BACKBONE_ATOMS = ("N", "CA", "C", "O")
_NAN_COORDS = np.full((4, 3), np.nan, dtype=np.float32)


def _residue_coords(residue) -> np.ndarray:
    """Return (4, 3) backbone coordinates for a Bio.PDB residue; NaN for missing atoms."""
    coords = _NAN_COORDS.copy()
    for i, atom_name in enumerate(_BACKBONE_ATOMS):
        if residue.has_id(atom_name):
            coords[i] = residue[atom_name].get_vector().get_array()
    return coords


def featurize_chains(
    chain_map: Dict[str, "Bio.PDB.Chain.Chain"],  # noqa: F821
    chain_order: Optional[list] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a mapping of chain_id → Bio.PDB chain objects into ProteinMPNN
    input tensors (single-structure batch, B=1).

    Args:
        chain_map: Dict mapping chain letter to Bio.PDB Chain object.
        chain_order: Ordered list of chain letters to use.  Defaults to
            sorted(chain_map.keys()).

    Returns:
        X:                  (1, L_total, 4, 3)  backbone coordinates
        mask:               (1, L_total)         1.0 for valid residues
        residue_idx:        (1, L_total)         residue sequence numbers
        chain_encoding_all: (1, L_total)         integer chain id (0-based)
    """
    if chain_order is None:
        chain_order = sorted(chain_map.keys())

    all_coords: list = []
    all_mask: list = []
    all_resnum: list = []
    all_chain_enc: list = []

    for chain_enc_idx, ch_id in enumerate(chain_order):
        chain = chain_map[ch_id]
        residues = [r for r in chain.get_residues() if r.id[0] == " "]
        for res in residues:
            coords = _residue_coords(res)
            valid = not np.any(np.isnan(coords))
            coords = np.nan_to_num(coords, nan=0.0)
            all_coords.append(coords)
            all_mask.append(1.0 if valid else 0.0)
            all_resnum.append(res.id[1])
            all_chain_enc.append(chain_enc_idx)

    L = len(all_coords)
    X = torch.tensor(np.stack(all_coords, axis=0), dtype=torch.float32).unsqueeze(0)  # (1,L,4,3)
    mask = torch.tensor(all_mask, dtype=torch.float32).unsqueeze(0)                   # (1,L)
    residue_idx = torch.tensor(all_resnum, dtype=torch.long).unsqueeze(0)             # (1,L)
    chain_encoding_all = torch.tensor(all_chain_enc, dtype=torch.long).unsqueeze(0)   # (1,L)

    return X, mask, residue_idx, chain_encoding_all


def featurize_pdb(
    pdb_path: str,
    chain_ids: Optional[list] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a PDB file and return ProteinMPNN input tensors (B=1).

    Args:
        pdb_path: Path to PDB or mmCIF file.
        chain_ids: Subset of chain letters to include.  Defaults to all chains.

    Returns:
        Same four tensors as featurize_chains.
    """
    try:
        from Bio import PDB
    except ImportError as exc:
        raise ImportError("Biopython is required for featurize_pdb.") from exc

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = next(structure.get_models())
    chain_map = {ch.id: ch for ch in model.get_chains()}

    if chain_ids is not None:
        chain_map = {k: v for k, v in chain_map.items() if k in chain_ids}
        order = [c for c in chain_ids if c in chain_map]
    else:
        order = sorted(chain_map.keys())

    return featurize_chains(chain_map, chain_order=order)


# ---------------------------------------------------------------------------
# ProteinMPNN encoder wrapper
# ---------------------------------------------------------------------------


class ProteinMPNNEncoder(nn.Module):
    """Per-residue structure encoder based on the ProteinMPNN encoder stack.

    Args:
        checkpoint_path: Path to a ProteinMPNN checkpoint (e.g. v_48_020.pt).
            Must contain a 'model_state_dict' key.  Set to None to use
            randomly initialised weights (for testing).
        hidden_dim: Hidden dimension of the ProteinMPNN model (default 128).
        num_encoder_layers: Number of encoder layers (default 3).
        k_neighbors: Number of neighbours in the kNN graph (default 48).
        augment_eps: Coordinate noise for augmentation during training
            (default 0.0 – disabled for inference).
    """

    HIDDEN_DIM: int = 128  # default d_struct exposed to structure_repr_mlp

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        k_neighbors: int = 48,
        augment_eps: float = 0.0,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim

        try:
            from protein_mpnn_utils import ProteinMPNN as _PMPNN
        except ImportError as exc:
            raise ImportError(
                "protein_mpnn_utils is required for ProteinMPNNEncoder.  "
                "Clone dauparas/ProteinMPNN and add it to your Python path."
            ) from exc

        self._model = _PMPNN(
            node_features=hidden_dim,
            edge_features=hidden_dim,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,
            vocab=21,
            k_neighbors=k_neighbors,
            augment_eps=augment_eps,
            dropout=0.0,
        )

        if checkpoint_path is not None:
            LOG.info("Loading ProteinMPNN checkpoint from %s", checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            self._model.load_state_dict(ckpt["model_state_dict"])

        # Freeze encoder weights – only structure_repr_mlp is trainable
        for param in self._model.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Encoder forward
    # ------------------------------------------------------------------

    def encode(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
    ) -> torch.Tensor:
        """Run the ProteinMPNN encoder and return per-residue node features.

        This replicates the encoder portion of ProteinMPNN.forward() without
        running the decoder, following the standard ProteinMPNN architecture
        in protein_mpnn_utils.py.

        Args:
            X:                  (B, L, 4, 3) backbone coordinates (N, CA, C, O)
            mask:               (B, L)        1.0 for valid residues
            residue_idx:        (B, L)        residue sequence numbers
            chain_encoding_all: (B, L)        integer chain id (0-based)

        Returns:
            h_V: (B, L, hidden_dim) per-residue encoder features
        """
        device = X.device

        # --- build edge features and kNN graph ---
        E, E_idx = self._model.features(X, mask, residue_idx, chain_encoding_all)

        # --- initialise node features (zeros, as in ProteinMPNN forward) ---
        B, L, _ = E.shape[:3]
        h_V = torch.zeros(B, L, self.hidden_dim, device=device)
        h_E = self._model.W_e(E)

        # --- attention mask ---
        mask_attend = _gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        # --- encoder layers ---
        for layer in self._model.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask_attend)

        return h_V  # (B, L, hidden_dim)

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for encode() – returns h_V."""
        return self.encode(X, mask, residue_idx, chain_encoding_all)


# ---------------------------------------------------------------------------
# Internal helper (mirrors ProteinMPNN's gather_nodes)
# ---------------------------------------------------------------------------


def _gather_nodes(
    nodes: torch.Tensor, neighbor_idx: torch.Tensor
) -> torch.Tensor:
    """Gather neighbour node features using a pre-built kNN index.

    Args:
        nodes:        (B, N, d)
        neighbor_idx: (B, N, K)

    Returns:
        (B, N, K, d)
    """
    B, N, d = nodes.shape
    K = neighbor_idx.shape[-1]
    idx_flat = neighbor_idx.view(B, -1)  # (B, N*K)
    idx_flat = idx_flat.unsqueeze(-1).expand(-1, -1, d)
    gathered = nodes.gather(1, idx_flat)  # (B, N*K, d)
    return gathered.view(B, N, K, d)
