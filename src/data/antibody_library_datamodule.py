"""AntibodyLibraryData – datamodule for training RankFlow from a master CSV.

Each row in the CSV represents one antibody variant in a library.  Rows are
grouped by ``library_id``; each library is treated as one assay for RankFlow's
list-wise ranking.

Expected CSV columns
--------------------
    universal_id       – unique variant identifier (ignored after grouping)
    library_id         – groups rows into assays
    heavy_sequence     – VH (or VHH) amino-acid string; may be empty/NaN for
                         antigen-only libraries
    light_sequence     – VL amino-acid string; may be empty/NaN for nanobodies
    antigen_sequence   – antigen amino-acid string; may be empty/NaN
    norm_affinity      – normalised affinity score (float, higher = better)

Any additional columns are silently ignored.

Batch tuple format (9-element, multichain) — same as AntibodyAffinityData
--------------------------------------------------------------------------
    assay_names            list[str]
    batch_tokens           list[Tensor(1,T)]   wildtype tokens per library
    chain_ids_list         list[Tensor(1,T)]   chain-id tokens per library
    coords                 list[4-tuple]       ProteinMPNN feats per library
    train_labels           list[list[tuple]]   per-library train label lists
    valid_labels           list[list[tuple]]   per-library valid label lists
    msa_bank               None  (not used)
    higher_mutseqs_100     list[Tensor]        top-100 encoded sequences
    lower_mutseqs_100      list[Tensor]        bottom-100 encoded sequences

Label tuple format: (score, mutant_list, tokens, chain_ids)
    score        – torch.float32 scalar (norm_affinity value)
    mutant_list  – [] (empty; no mutation strings are parsed)
    tokens       – (1, T) LongTensor of multichain tokens for this variant
    chain_ids    – (1, T) LongTensor of per-token chain indices
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.antibody_affinity_datamodule import (
    _extract_seqs_from_pdb,
    _get_mint_alphabet,
    encode_multichain,
)
from src.models.components.proteinmpnn_encoder import featurize_pdb

LOG = logging.getLogger(__name__)

# Column names expected in the master CSV
_COL_LIB  = "library_id"
_COL_HEAVY = "heavy_sequence"
_COL_LIGHT = "light_sequence"
_COL_AG    = "antigen_sequence"
_COL_SCORE = "norm_affinity"

# Chain-letter → CSV column mapping (must match chain_order convention)
_CHAIN_COL: Dict[str, str] = {
    "H": _COL_HEAVY,
    "L": _COL_LIGHT,
    "A": _COL_AG,
}


# ---------------------------------------------------------------------------
# Per-library dataset
# ---------------------------------------------------------------------------


class _LibraryAssay:
    """Holds all pre-computed data for a single library/assay.

    Attributes are set by ``AntibodyLibraryData.setup()`` and then accessed
    by ``AntibodyLibraryDataset.__getitem__``.
    """

    __slots__ = (
        "assay_name",
        "batch_tokens",
        "chain_ids",
        "pmpnn_feats",
        "train_labels",
        "valid_labels",
        "higher_mutseqs_100",
        "lower_mutseqs_100",
    )

    def __init__(
        self,
        assay_name: str,
        batch_tokens: torch.Tensor,
        chain_ids: torch.Tensor,
        pmpnn_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        train_labels: List,
        valid_labels: List,
        higher_mutseqs_100: List[torch.Tensor],
        lower_mutseqs_100: List[torch.Tensor],
    ) -> None:
        self.assay_name        = assay_name
        self.batch_tokens      = batch_tokens
        self.chain_ids         = chain_ids
        self.pmpnn_feats       = pmpnn_feats
        self.train_labels      = train_labels
        self.valid_labels      = valid_labels
        self.higher_mutseqs_100 = higher_mutseqs_100
        self.lower_mutseqs_100  = lower_mutseqs_100


class AntibodyLibraryDataset(Dataset):
    """Dataset that wraps a collection of per-library assays.

    Each ``__getitem__`` call returns the 9-element tuple for one library,
    wrapped in lists to match the format expected by the RankFlow training
    loop (which zips over lists of assays).

    Args:
        assays: Pre-built list of ``_LibraryAssay`` objects.
    """

    def __init__(self, assays: List[_LibraryAssay]) -> None:
        super().__init__()
        self._assays = assays

    def __len__(self) -> int:
        return len(self._assays)

    def __getitem__(self, index: int):
        a = self._assays[index]
        return (
            [a.assay_name],           # assay_names
            [a.batch_tokens],         # batch_tokens  list[(1,T)]
            [a.chain_ids],            # chain_ids_list list[(1,T)]
            [a.pmpnn_feats],          # coords  list[4-tuple]
            [a.train_labels],         # train_labels
            [a.valid_labels],         # valid_labels
            None,                     # msa_bank  (not used)
            a.higher_mutseqs_100,     # higher_mutseqs_100
            a.lower_mutseqs_100,      # lower_mutseqs_100
        )


# ---------------------------------------------------------------------------
# Helper: build per-row tokens and labels
# ---------------------------------------------------------------------------


def _row_chain_seqs(
    row: pd.Series,
    chain_order: List[str],
) -> Dict[str, str]:
    """Extract chain sequences from a CSV row, using empty string for missing values."""
    seqs: Dict[str, str] = {}
    for ch in chain_order:
        col = _CHAIN_COL.get(ch)
        if col is not None and col in row.index:
            val = row[col]
            seqs[ch] = "" if (val is None or (isinstance(val, float) and np.isnan(val))) else str(val)
        else:
            seqs[ch] = ""
    return seqs


def _build_library_labels(
    df: pd.DataFrame,
    chain_order: List[str],
    alphabet,
) -> List[Tuple]:
    """Build a label list for one library split (train or valid).

    Each entry is ``(score, mutant_list, tokens, chain_ids)`` where:
        score       – torch.float32 scalar (norm_affinity)
        mutant_list – [] (empty; sequences are provided directly)
        tokens      – (1, T) LongTensor
        chain_ids   – (1, T) LongTensor
    """
    labels = []
    for _, row in df.iterrows():
        score = torch.tensor(float(row[_COL_SCORE]), dtype=torch.float32)
        chain_seqs = _row_chain_seqs(row, chain_order)
        tokens, chain_ids = encode_multichain(chain_seqs, chain_order, alphabet)
        labels.append((score, [], tokens, chain_ids))
    return labels


def _encode_seqs_as_tokens(
    df: pd.DataFrame,
    chain_order: List[str],
    alphabet,
) -> List[torch.Tensor]:
    """Encode all variant sequences in *df* and return a list of (1,T) token tensors."""
    encoded = []
    for _, row in df.iterrows():
        chain_seqs = _row_chain_seqs(row, chain_order)
        tokens, _ = encode_multichain(chain_seqs, chain_order, alphabet)
        encoded.append(tokens)
    return encoded


# ---------------------------------------------------------------------------
# Helper: load YAML / JSON library→PDB map
# ---------------------------------------------------------------------------


def _load_pdb_map(path: str) -> Dict[str, str]:
    """Load a library_id → PDB path mapping from a YAML or JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"library_pdb_map file not found: {path}")
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(p) as f:
                return yaml.safe_load(f)
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load a .yaml library_pdb_map.  "
                "Install it with: pip install pyyaml"
            ) from exc
    elif suffix == ".json":
        import json
        with open(p) as f:
            return json.load(f)
    else:
        raise ValueError(
            f"Unsupported library_pdb_map format '{suffix}'.  Use .yaml/.yml or .json."
        )


# ---------------------------------------------------------------------------
# LightningDataModule
# ---------------------------------------------------------------------------


class AntibodyLibraryData(LightningDataModule):
    """LightningDataModule for training RankFlow from a master CSV of antibody
    libraries.

    Each ``library_id`` in the CSV becomes one assay for RankFlow's list-wise
    ranking.  A ``library_pdb_map`` YAML/JSON file maps each ``library_id`` to
    its wildtype complex PDB path; this structure is reused for all variants in
    the library.

    Args:
        data_dir:          Root directory.  Relative paths in ``master_csv``
                           and ``library_pdb_map`` are resolved against this.
        master_csv:        Path to the master CSV (absolute or relative to
                           ``data_dir``).
        library_pdb_map:   Path to a YAML or JSON file mapping
                           ``library_id`` → PDB path (absolute or relative to
                           ``data_dir``).
        chain_order:       Ordered list of chain letters to encode, e.g.
                           ``["H", "L", "A"]`` for heavy, light, antigen.
                           Supported letters: H (heavy_sequence), L
                           (light_sequence), A (antigen_sequence).
        split_type:        Fold column prefix for cross-validation splits.
                           The column ``fold_{split_type}_5`` is looked up in
                           the CSV.  If absent, an 80/20 split is used.
        split_index:       Which fold to use as the validation set (0-4).
        batch_size:        DataLoader batch size (always 1 per item; ignored
                           by the inner collation logic).
        num_workers:       DataLoader worker count.
        pin_memory:        Whether to pin DataLoader memory.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        master_csv: str = "antibody_libraries.csv",
        library_pdb_map: str = "library_pdb_map.yaml",
        chain_order: Optional[List[str]] = None,
        split_type: Optional[str] = None,
        split_index: int = 0,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val:   Optional[Dataset] = None

    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        pass

    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_train is not None:
            return

        hp = self.hparams
        data_dir = Path(hp.data_dir)

        def _resolve(p: str) -> Path:
            pp = Path(p)
            return pp if pp.is_absolute() else data_dir / pp

        csv_path = _resolve(hp.master_csv)
        pdb_map_path = _resolve(hp.library_pdb_map)

        # Load master CSV
        LOG.info("Loading master CSV from %s", csv_path)
        master_df = pd.read_csv(csv_path)
        _require_columns(master_df, [_COL_LIB, _COL_SCORE])

        # Load library → PDB mapping
        pdb_map = _load_pdb_map(str(pdb_map_path))

        # Determine chain order
        chain_order: List[str] = list(hp.chain_order) if hp.chain_order else ["H", "L", "A"]

        # Resolve PDB paths relative to data_dir if not absolute
        def _pdb_path(raw: str) -> str:
            p = Path(raw)
            if p.is_absolute():
                return str(p)
            candidate = data_dir / p
            return str(candidate)

        alphabet = _get_mint_alphabet()

        fold_col = f"fold_{hp.split_type}_5" if hp.split_type else None

        train_assays: List[_LibraryAssay] = []
        valid_assays: List[_LibraryAssay] = []

        for lib_id, lib_df in master_df.groupby(_COL_LIB):
            lib_df = lib_df.reset_index(drop=True)
            lib_id_str = str(lib_id)

            if lib_id_str not in pdb_map:
                LOG.warning(
                    "library_id '%s' not found in library_pdb_map; skipping.", lib_id_str
                )
                continue

            pdb_path = _pdb_path(pdb_map[lib_id_str])
            if not os.path.isfile(pdb_path):
                LOG.warning(
                    "PDB file '%s' for library '%s' does not exist; skipping.",
                    pdb_path, lib_id_str,
                )
                continue

            # ProteinMPNN featurisation from wildtype PDB
            try:
                pmpnn_feats = featurize_pdb(pdb_path, chain_ids=chain_order)
            except Exception as exc:
                LOG.warning(
                    "featurize_pdb failed for library '%s' (%s): %s; skipping.",
                    lib_id_str, pdb_path, exc,
                )
                continue

            # Wildtype sequences from PDB (used as reference / steering vectors)
            try:
                wt_chain_seqs = _extract_seqs_from_pdb(pdb_path, chain_order)
            except Exception as exc:
                LOG.warning(
                    "_extract_seqs_from_pdb failed for library '%s': %s; "
                    "falling back to empty wildtype sequences.",
                    lib_id_str, exc,
                )
                wt_chain_seqs = {ch: "" for ch in chain_order}

            # Wildtype multichain tokens (reference sequence)
            wt_tokens, wt_chain_ids = encode_multichain(wt_chain_seqs, chain_order, alphabet)

            # Train / validation split
            train_df, valid_df = _split_library(
                lib_df, fold_col=fold_col, split_index=hp.split_index
            )

            if len(train_df) == 0:
                LOG.warning("Library '%s' has no training rows after split; skipping.", lib_id_str)
                continue

            # Build labels
            train_labels = _build_library_labels(train_df, chain_order, alphabet)
            valid_labels = _build_library_labels(valid_df, chain_order, alphabet) if len(valid_df) else []

            # Top/bottom 100 encoded sequences for steering vectors
            sorted_df = lib_df.sort_values(_COL_SCORE, ascending=False).reset_index(drop=True)
            higher_100 = _encode_seqs_as_tokens(sorted_df.head(100), chain_order, alphabet)
            lower_100  = _encode_seqs_as_tokens(sorted_df.tail(100), chain_order, alphabet)

            assay = _LibraryAssay(
                assay_name=lib_id_str,
                batch_tokens=wt_tokens,
                chain_ids=wt_chain_ids,
                pmpnn_feats=pmpnn_feats,
                train_labels=train_labels,
                valid_labels=valid_labels,
                higher_mutseqs_100=higher_100,
                lower_mutseqs_100=lower_100,
            )
            train_assays.append(assay)
            valid_assays.append(assay)

            LOG.info(
                "AntibodyLibraryData: lib=%s  train=%d  val=%d",
                lib_id_str, len(train_df), len(valid_df),
            )

        if not train_assays:
            raise RuntimeError(
                "AntibodyLibraryData: no libraries could be loaded.  "
                "Check your master_csv, library_pdb_map, and chain_order."
            )

        self.data_train = AntibodyLibraryDataset(train_assays)
        self.data_val   = AntibodyLibraryDataset(valid_assays)

    # ------------------------------------------------------------------

    def _collator(self, raw_batch):
        return raw_batch[0]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self._collator,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self._collator,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"master CSV is missing required columns: {missing}.  "
            f"Found columns: {list(df.columns)}"
        )


def _split_library(
    df: pd.DataFrame,
    fold_col: Optional[str],
    split_index: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, valid_df) for one library.

    If *fold_col* is not None and exists as a column in *df*, rows with
    ``fold_col == split_index`` are used for validation; the rest for training.
    Otherwise an 80/20 split is performed.
    """
    if fold_col is not None and fold_col in df.columns:
        train_df = df[df[fold_col] != split_index].reset_index(drop=True)
        valid_df = df[df[fold_col] == split_index].reset_index(drop=True)
    else:
        if fold_col is not None:
            LOG.warning(
                "Fold column '%s' not found in library; falling back to 80/20 split.",
                fold_col,
            )
        n = len(df)
        split = max(1, int(0.8 * n))
        train_df = df.iloc[:split].reset_index(drop=True)
        valid_df = df.iloc[split:].reset_index(drop=True)
    return train_df, valid_df
