"""Antibody affinity datamodule with multichain support.

Supports antibody deep-mutational scanning (DMS) datasets where each library
has a wildtype complex structure (PDB) and mutations may span multiple chains
(e.g., heavy-chain VH and light-chain VL).

Key differences from ProteinGymSubstitutionData:
  - Tokenises each chain separately using mint.data.Alphabet and concatenates
    them with CLS/EOS separators.
  - Builds chain_ids (B, T) tensors alongside tokens (B, T).
  - Precomputes ProteinMPNN featurisation (X, mask, residue_idx,
    chain_encoding_all) from a PDB file for each library.
  - Mutation positions (pos1) are token-level positions in the concatenated
    multichain token sequence so that the RankFlow training loop can mask
    them without additional bookkeeping.

Batch tuple format (9-element, multichain):
    assay_names, batch_tokens, chain_ids_list, coords,
    train_labels, valid_labels, msa_bank,
    higher_mutseqs_100, lower_mutseqs_100

Mutation tuple format: (tok_pos, wt_tok_idx, mut_tok_idx)
    tok_pos   – 1-indexed token position in the concatenated sequence
    wt_tok_idx  – alphabet index of the wildtype amino acid
    mut_tok_idx – alphabet index of the mutant amino acid
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

LOG = logging.getLogger(__name__)

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {a: i for i, a in enumerate(AA20)}


# ---------------------------------------------------------------------------
# MINT alphabet / tokeniser helpers
# ---------------------------------------------------------------------------


def _get_mint_alphabet():
    """Load mint.data.Alphabet (ESM-1b architecture)."""
    try:
        from mint.data import Alphabet
        return Alphabet.from_architecture("ESM-1b")
    except ImportError as exc:
        raise ImportError(
            "The 'mint' package is required for AntibodyAffinityData.  "
            "Install it from https://github.com/VarunUllanat/mint"
        ) from exc


def encode_multichain(
    chain_sequences: Dict[str, str],
    chain_order: List[str],
    alphabet,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode multiple chains into a single concatenated token sequence.

    Token layout per chain: [CLS] aa1 aa2 ... aaN [EOS]
    Chains are concatenated directly without additional separator.

    Args:
        chain_sequences: mapping from chain letter → amino-acid string.
        chain_order: ordered list of chain letters.
        alphabet: mint.data.Alphabet instance.

    Returns:
        tokens:    (1, T) LongTensor
        chain_ids: (1, T) LongTensor (0-based chain index for each token)
    """
    tokens_list: List[int] = []
    chain_id_list: List[int] = []

    for chain_enc_idx, ch in enumerate(chain_order):
        seq = chain_sequences[ch]
        # CLS
        tokens_list.append(alphabet.cls_idx)
        chain_id_list.append(chain_enc_idx)
        # Residues
        for aa in seq:
            tokens_list.append(alphabet.get_idx(aa))
            chain_id_list.append(chain_enc_idx)
        # EOS
        tokens_list.append(alphabet.eos_idx)
        chain_id_list.append(chain_enc_idx)

    tokens    = torch.tensor(tokens_list, dtype=torch.long).unsqueeze(0)   # (1, T)
    chain_ids = torch.tensor(chain_id_list, dtype=torch.long).unsqueeze(0)  # (1, T)
    return tokens, chain_ids


def token_position_for_residue(
    chain_idx: int,
    residue_pos_in_chain: int,  # 1-based
    chain_order: List[str],
    chain_sequences: Dict[str, str],
) -> int:
    """Return the 1-indexed token position of a residue in the concatenated sequence.

    Token layout: [CLS aa1..aaN EOS] repeated for each chain.
    Token index 0 = CLS of chain 0; token index 1 = aa1 of chain 0; etc.

    Args:
        chain_idx: 0-based index of the target chain in chain_order.
        residue_pos_in_chain: 1-based residue number within the chain.
        chain_order: ordered list of chain letters.
        chain_sequences: chain_letter → sequence string.

    Returns:
        1-indexed token position (suitable for tok[0, pos1] indexing).
    """
    offset = 0
    for i, ch in enumerate(chain_order):
        # +2 per chain: CLS + EOS
        chain_len = len(chain_sequences[ch]) + 2
        if i == chain_idx:
            # CLS is at offset, residue k is at offset + k
            return offset + residue_pos_in_chain
        offset += chain_len
    raise ValueError(f"chain_idx {chain_idx} out of range for {chain_order}")


# ---------------------------------------------------------------------------
# Mutation parsing helpers
# ---------------------------------------------------------------------------


def parse_antibody_mutant(
    mutant_str: str,
    chain_order: List[str],
    chain_sequences: Dict[str, str],
    alphabet,
) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
    """Parse an antibody mutation string and return token-level tuples.

    Supports formats:
      - "HA5G"          – chain H, position 5, wt=A → mut=G
      - "H:A5G"         – same with explicit colon separator
      - "A5G"           – no chain prefix → assume first chain
      - Multiple mutations separated by ':'  e.g. "HA5G:LK10R"

    Returns list of (tok_pos, wt_tok_idx, mut_tok_idx) tuples where
    tok_pos is 1-indexed in the concatenated token sequence.
    """
    result = []
    parts = mutant_str.split(":")
    for part in parts:
        part = part.strip()
        # Try "CH_LETTER + POSITION + MUT" e.g. "HA5G" or "H:A5G"
        m = re.match(r"^([A-Za-z]):?([A-Z])(\d+)([A-Z])$", part)
        if m:
            ch_letter = m.group(1).upper()
            wt_aa     = m.group(2)
            pos_in_ch = int(m.group(3))
            mut_aa    = m.group(4)
        else:
            # No chain prefix: standard ProteinGym format "A5G"
            m2 = re.match(r"^([A-Z])(\d+)([A-Z])$", part)
            if m2 is None:
                LOG.warning("Cannot parse mutation '%s', skipping.", part)
                continue
            ch_letter = chain_order[0]
            wt_aa     = m2.group(1)
            pos_in_ch = int(m2.group(2))
            mut_aa    = m2.group(3)

        if ch_letter not in chain_order:
            LOG.warning("Chain '%s' not in chain_order %s, skipping.", ch_letter, chain_order)
            continue

        chain_idx = chain_order.index(ch_letter)
        tok_pos = token_position_for_residue(chain_idx, pos_in_ch, chain_order, chain_sequences)

        wt_tok  = torch.tensor(alphabet.get_idx(wt_aa),  dtype=torch.long)
        mut_tok = torch.tensor(alphabet.get_idx(mut_aa), dtype=torch.long)
        result.append((tok_pos, wt_tok, mut_tok))

    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AntibodyAffinityDataset(Dataset):
    """Dataset for antibody affinity DMS data with multichain support.

    Args:
        assay_name: Identifier string for this assay/library.
        chain_sequences: Dict mapping chain letter → wildtype amino-acid string.
        chain_order: Ordered list of chain letters (determines token layout).
        pmpnn_feats: Tuple (X, mask, residue_idx, chain_encoding_all) of
            ProteinMPNN features pre-computed from the wildtype PDB.
        train_data: DataFrame with columns 'mutant' and 'DMS_score'.
        valid_data: DataFrame with columns 'mutant' and 'DMS_score'.
        msa_dir: Optional path to directory containing A2M MSA files.
    """

    def __init__(
        self,
        assay_name: str,
        chain_sequences: Dict[str, str],
        chain_order: List[str],
        pmpnn_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        msa_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.assay_name   = assay_name
        self.chain_order  = chain_order
        self.chain_seqs   = chain_sequences

        self.alphabet = _get_mint_alphabet()

        # Encode wildtype multichain tokens
        wt_tokens, wt_chain_ids = encode_multichain(chain_sequences, chain_order, self.alphabet)
        self.batch_tokens = wt_tokens      # (1, T)
        self.chain_ids    = wt_chain_ids   # (1, T)

        # ProteinMPNN featurisation (pre-computed)
        self.pmpnn_feats = pmpnn_feats  # (X, mask, residue_idx, chain_encoding_all)

        # MSA bank (optional; uses same A2M logic as ProteinGymSubstitutionDataset)
        full_seq = "".join(chain_sequences[c] for c in chain_order)
        self.msa_bank = _try_load_msa_bank(assay_name, msa_dir, len(full_seq))

        # Precompute global score stats for normalisation
        all_scores = list(train_data["DMS_score"]) + list(valid_data["DMS_score"])
        arr = np.asarray(all_scores, dtype=np.float64)
        self._score_mean = float(arr.mean())
        self._score_std  = float(arr.std()) if arr.std() > 0 else 1.0

        # Build label lists
        self.train_labels = self._build_labels(train_data)
        self.valid_labels = self._build_labels(valid_data)

        # Top/bottom 100 for steering vector
        sorted_train = train_data.sort_values("DMS_score", ascending=False).reset_index(drop=True)
        self.higher_mutseqs_100 = self._encode_mutseqs(sorted_train["mutated_sequence"].tolist()[:100])
        self.lower_mutseqs_100  = self._encode_mutseqs(sorted_train["mutated_sequence"].tolist()[-100:])

    # ------------------------------------------------------------------

    def _build_labels(self, df: pd.DataFrame) -> List:
        labels = []
        for _, row in df.iterrows():
            mutant_str  = str(row.get("mutant", ""))
            mutant_list = parse_antibody_mutant(
                mutant_str, self.chain_order, self.chain_seqs, self.alphabet
            )
            score = torch.tensor(float(row["DMS_score"]), dtype=torch.float32)

            # Encode the mutated multichain sequence
            mut_seq_str = row.get("mutated_sequence", None)
            if mut_seq_str is not None and isinstance(mut_seq_str, str):
                # Rebuild chain_sequences with mutations applied
                mut_chain_seqs = _apply_mutant_str_to_chains(
                    mutant_str, self.chain_seqs, self.chain_order
                )
            else:
                mut_chain_seqs = self.chain_seqs

            mut_tokens, mut_chain_ids = encode_multichain(mut_chain_seqs, self.chain_order, self.alphabet)
            labels.append((score, mutant_list, mut_tokens, mut_chain_ids))
        return labels

    def _encode_mutseqs(self, sequences: List[str]) -> List[torch.Tensor]:
        """Encode a list of full concatenated sequences (single string) into token tensors."""
        encoded = []
        for seq in sequences:
            # seq is the concatenated sequence of all chains
            # Rebuild per-chain sequences by splitting at known lengths
            chain_lens = [len(self.chain_seqs[c]) for c in self.chain_order]
            offset = 0
            chain_seqs_mut: Dict[str, str] = {}
            for ch, length in zip(self.chain_order, chain_lens):
                chain_seqs_mut[ch] = seq[offset: offset + length]
                offset += length
            tokens, _ = encode_multichain(chain_seqs_mut, self.chain_order, self.alphabet)
            encoded.append(tokens)
        return encoded

    def __getitem__(self, index):
        # Wrap assay-level entries in lists so that training_step can zip over
        # multiple assays (consistent with ProteinGymSubstitutionDataset format).
        return (
            [self.assay_name],          # assay_names
            [self.batch_tokens],        # batch_tokens (list of (1,T) tensors)
            [self.chain_ids],           # chain_ids_list (list of (1,T) tensors)
            [self.pmpnn_feats],         # coords / pmpnn_feats (list of 4-tuples)
            [self.train_labels],        # train_labels (list of label lists)
            [self.valid_labels],        # valid_labels
            self.msa_bank,
            self.higher_mutseqs_100,
            self.lower_mutseqs_100,
        )

    def __len__(self):
        return 1


# ---------------------------------------------------------------------------
# Mutation application helper
# ---------------------------------------------------------------------------


def _apply_mutant_str_to_chains(
    mutant_str: str,
    chain_seqs: Dict[str, str],
    chain_order: List[str],
) -> Dict[str, str]:
    """Apply mutations from a mutant string to wildtype chain sequences."""
    mut_seqs = {ch: list(chain_seqs[ch]) for ch in chain_order}
    for part in mutant_str.split(":"):
        part = part.strip()
        m = re.match(r"^([A-Za-z]):?([A-Z])(\d+)([A-Z])$", part)
        if m:
            ch    = m.group(1).upper()
            pos   = int(m.group(3)) - 1  # 0-indexed
            mut_aa = m.group(4)
        else:
            m2 = re.match(r"^([A-Z])(\d+)([A-Z])$", part)
            if m2 is None:
                continue
            ch    = chain_order[0]
            pos   = int(m2.group(2)) - 1
            mut_aa = m2.group(3)
        if ch in mut_seqs and 0 <= pos < len(mut_seqs[ch]):
            mut_seqs[ch][pos] = mut_aa
    return {ch: "".join(mut_seqs[ch]) for ch in chain_order}


# ---------------------------------------------------------------------------
# MSA loading helper (re-uses logic from ProteinGym datamodule)
# ---------------------------------------------------------------------------


def _try_load_msa_bank(
    assay_name: str, msa_dir: Optional[str], seq_len: int
) -> Optional[torch.Tensor]:
    if msa_dir is None:
        return None
    try:
        from src.data.proteingym_substitution_datamodule import build_msa_bank_a2m
        return build_msa_bank_a2m(
            name=[assay_name], msa_dir=msa_dir, length=seq_len
        )
    except Exception as exc:
        LOG.warning("Could not load MSA bank for %s: %s", assay_name, exc)
        return None


# ---------------------------------------------------------------------------
# LightningDataModule
# ---------------------------------------------------------------------------


class AntibodyAffinityData(LightningDataModule):
    """LightningDataModule for antibody affinity DMS datasets.

    Expects a CSV file with columns:
      - 'mutant'            – mutation string(s) e.g. "HA5G:LK10R"
      - 'DMS_score'         – experimental fitness score
      - 'mutated_sequence'  – (optional) full concatenated mutant sequence
      - 'fold_random_5'     – (optional) fold assignment for cross-validation

    Args:
        data_dir: Root directory containing 'substitutions/', 'structure/',
            and optionally 'msa/' subdirectories.
        assay_csv: Path (relative to data_dir or absolute) to the DMS CSV.
        pdb_path: Path to the wildtype complex PDB file.
        chain_order: Ordered list of chain letters to include from the PDB
            (e.g. ['H', 'L'] for antibody heavy + light chain).
        split_type: Cross-validation split column prefix (default 'random').
        split_index: Which fold to use as validation (default 0).
        msa_dir: Optional directory containing A2M MSA files.
        batch_size: Batch size (always 1 for this datamodule; ignored).
        num_workers: DataLoader workers.
        pin_memory: Whether to pin DataLoader memory.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        assay_csv: str = "substitutions/assay.csv",
        pdb_path: str = "structure/wildtype.pdb",
        chain_order: Optional[List[str]] = None,
        split_type: str = "random",
        split_index: int = 0,
        msa_dir: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val:   Optional[Dataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_train is not None:
            return

        data_dir = Path(self.hparams.data_dir)
        pdb_full = Path(self.hparams.pdb_path) if Path(self.hparams.pdb_path).is_absolute() \
            else data_dir / self.hparams.pdb_path
        csv_full = Path(self.hparams.assay_csv) if Path(self.hparams.assay_csv).is_absolute() \
            else data_dir / self.hparams.assay_csv

        # Load DMS data
        assay_df = pd.read_csv(csv_full)
        assay_name = csv_full.stem

        # Determine chain order
        chain_order = self.hparams.chain_order or ["H", "L"]

        # Load PDB and extract ProteinMPNN features
        from src.models.components.proteinmpnn_encoder import featurize_pdb
        X, mask, residue_idx, chain_encoding_all = featurize_pdb(
            str(pdb_full), chain_ids=chain_order
        )
        pmpnn_feats = (X, mask, residue_idx, chain_encoding_all)

        # Extract wildtype sequences from PDB
        chain_seqs = _extract_seqs_from_pdb(str(pdb_full), chain_order)

        # Split into train / validation
        fold_col = f"fold_{self.hparams.split_type}_5"
        if fold_col in assay_df.columns:
            train_df = assay_df[assay_df[fold_col] != self.hparams.split_index].reset_index(drop=True)
            valid_df = assay_df[assay_df[fold_col] == self.hparams.split_index].reset_index(drop=True)
        else:
            LOG.warning("Column %s not found; using 80/20 random split.", fold_col)
            n = len(assay_df)
            split = int(0.8 * n)
            train_df = assay_df.iloc[:split].reset_index(drop=True)
            valid_df = assay_df.iloc[split:].reset_index(drop=True)

        msa_dir_resolved = None
        if self.hparams.msa_dir:
            msa_dir_resolved = str(data_dir / self.hparams.msa_dir) \
                if not Path(self.hparams.msa_dir).is_absolute() \
                else self.hparams.msa_dir

        self.data_train = AntibodyAffinityDataset(
            assay_name=assay_name,
            chain_sequences=chain_seqs,
            chain_order=chain_order,
            pmpnn_feats=pmpnn_feats,
            train_data=train_df,
            valid_data=valid_df,
            msa_dir=msa_dir_resolved,
        )
        self.data_val = self.data_train  # same dataset; train/val split is internal

        LOG.info(
            "AntibodyAffinityData: assay=%s chains=%s train=%d val=%d",
            assay_name, chain_order, len(train_df), len(valid_df),
        )

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
# PDB sequence extraction helper
# ---------------------------------------------------------------------------


def _extract_seqs_from_pdb(
    pdb_path: str, chain_order: List[str]
) -> Dict[str, str]:
    """Extract amino-acid sequences for each chain from a PDB file.

    Uses Biopython's PDBIO and three-letter → one-letter code mapping.
    Falls back to reading SEQRES records if Biopython is not available.
    """
    try:
        from Bio import PDB
        from Bio.PDB.Polypeptide import protein_letters_3to1

        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("prot", pdb_path)
        model = next(structure.get_models())

        seqs: Dict[str, str] = {}
        for ch_id in chain_order:
            try:
                chain = model[ch_id]
            except KeyError:
                raise ValueError(f"Chain '{ch_id}' not found in {pdb_path}")
            residues = [r for r in chain.get_residues() if r.id[0] == " "]
            aa_seq = ""
            for res in residues:
                resname = res.get_resname().strip()
                one = protein_letters_3to1.get(resname, "X")
                aa_seq += one
            seqs[ch_id] = aa_seq
        return seqs

    except ImportError:
        # Fallback: parse ATOM records directly
        from collections import OrderedDict

        _3to1 = {
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
            "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
            "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
            "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        }
        chain_residues: Dict[str, OrderedDict] = {ch: OrderedDict() for ch in chain_order}
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                ch_id   = line[21]
                res_num = int(line[22:26].strip())
                resname = line[17:20].strip()
                if ch_id in chain_residues and res_num not in chain_residues[ch_id]:
                    chain_residues[ch_id][res_num] = _3to1.get(resname, "X")
        return {ch: "".join(chain_residues[ch].values()) for ch in chain_order}
