#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class NumberedSeq:
    positions: Tuple[str, ...]   # ordered position keys like "27", "27A", ...
    residues: Tuple[str, ...]    # residue per position, or "-"


def _run_anarcii(
    input_fasta_path: str, output_path: str, scheme: str, seq_type: str, cpu: bool
) -> None:
    """
    Runs ANARCII CLI and writes output to output_path (.csv or .msgpack).
    """
    cmd = ["anarcii", "--scheme", scheme, "--seq_type", seq_type, "-o", output_path, input_fasta_path]
    if cpu:
        cmd.insert(1, "--cpu")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "Could not find 'anarcii' on PATH. You installed it, but ensure the venv is active."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ANARCII failed.\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}\n") from e


def _sort_pos_key(k: str) -> Tuple[int, str]:
    """
    Sort keys like 27, 27A, 100, 100B in numeric+insertion order.
    """
    m = re.match(r"^(\d+)([A-Z]?)$", str(k))
    if not m:
        return (10**9, str(k))
    return (int(m.group(1)), m.group(2) or "")


def _build_alignment_space(pos_maps: List[Dict[str, str]]) -> List[str]:
    keys = set()
    for d in pos_maps:
        keys.update(d.keys())
    return sorted(keys, key=_sort_pos_key)


def _to_numbered_seq(space: List[str], pos_to_res: Dict[str, str]) -> NumberedSeq:
    return NumberedSeq(tuple(space), tuple(pos_to_res.get(p, "-") for p in space))


def _consensus(numbered: List[NumberedSeq]) -> NumberedSeq:
    if not numbered:
        return NumberedSeq(tuple(), tuple())
    positions = numbered[0].positions
    cons = []
    for i in range(len(positions)):
        col = [ns.residues[i] for ns in numbered]
        counts = Counter([c for c in col if c != "-" and c is not None])
        cons.append(counts.most_common(1)[0][0] if counts else "-")
    return NumberedSeq(positions, tuple(cons))


def _distance(a: NumberedSeq, b: NumberedSeq) -> int:
    assert a.positions == b.positions
    d = 0
    for ra, rb in zip(a.residues, b.residues):
        if ra == "-" and rb == "-":
            continue
        if ra != rb:
            d += 1
    return d


def _read_anarcii_csv(numbered_csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    Parse ANARCII wide CSV output into: seq_id -> {pos_key -> residue}.
    """
    df = pd.read_csv(numbered_csv_path, dtype=str)

    if "Name" not in df.columns:
        raise RuntimeError(f"Expected 'Name' column in ANARCII CSV. Found: {sorted(df.columns)}")

    pos_cols: List[str] = []
    for c in df.columns:
        cs = str(c).strip()
        if re.match(r"^\d+[A-Z]?$", cs):
            pos_cols.append(cs)

    if not pos_cols:
        raise RuntimeError(
            "Could not find any numbered position columns in ANARCII CSV. "
            f"Found columns: {sorted(df.columns)}"
        )

    pos_cols = sorted(pos_cols, key=_sort_pos_key)

    out: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        sid = str(row["Name"])
        pos_map: Dict[str, str] = {}
        for p in pos_cols:
            val = row.get(p)
            if val is None:
                continue
            aa = str(val).strip()
            if not aa or aa.lower() == "nan" or aa == "-":
                continue
            pos_map[p] = aa[0]
        out[sid] = pos_map

    return out


def _number_all_sequences_once(
    df: pd.DataFrame,
    seq_col: str,
    prefix: str,
    scheme: str,
    seq_type: str,
    cpu: bool,
    *,
    timing_label: str,
) -> Dict[int, Dict[str, str]]:
    """
    Run ANARCII once for all sequences in df[seq_col].

    Returns: row_index -> {pos_key -> residue}
    Only rows with non-empty sequences and successfully-numbered outputs are included.
    """
    t_all0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as td:
        fasta_path = os.path.join(td, f"{prefix}.fasta")
        out_csv = os.path.join(td, f"{prefix}_numbered.csv")

        # FASTA write timing
        t0 = time.perf_counter()
        entries: List[Tuple[str, str]] = []
        for idx, val in df[seq_col].items():
            s = "" if pd.isna(val) else str(val).strip()
            if not s:
                continue
            entries.append((f"{prefix}_{idx}", s))

        if not entries:
            print(f"[timing] {timing_label}: no sequences found; skipping", flush=True)
            return {}

        with open(fasta_path, "w", encoding="utf-8") as f:
            for sid, seq in entries:
                f.write(f">{sid}\n{seq}\n")
        t1 = time.perf_counter()
        print(f"[timing] {timing_label}: wrote FASTA ({len(entries)} seqs) in {t1 - t0:.3f}s", flush=True)

        # ANARCII timing
        t2 = time.perf_counter()
        _run_anarcii(fasta_path, out_csv, scheme=scheme, seq_type=seq_type, cpu=cpu)
        t3 = time.perf_counter()
        print(f"[timing] {timing_label}: anarcii run in {t3 - t2:.3f}s", flush=True)

        # CSV parse timing
        t4 = time.perf_counter()
        numbered = _read_anarcii_csv(out_csv)  # seq_id -> pos_map
        t5 = time.perf_counter()
        print(f"[timing] {timing_label}: parsed output CSV in {t5 - t4:.3f}s", flush=True)

        # Map back to dataframe row index
        t6 = time.perf_counter()
        out: Dict[int, Dict[str, str]] = {}
        for sid, pos_map in numbered.items():
            m = re.match(rf"^{re.escape(prefix)}_(\d+)$", sid)
            if not m:
                continue
            out[int(m.group(1))] = pos_map
        t7 = time.perf_counter()
        print(f"[timing] {timing_label}: mapped numbering to rows in {t7 - t6:.3f}s", flush=True)

    t_all1 = time.perf_counter()
    print(f"[timing] {timing_label}: total block time {t_all1 - t_all0:.3f}s", flush=True)
    return out


def _update_aligned_sequence_columns_per_library(
    df: pd.DataFrame,
    library_col: str,
    seq_col: str,
    numbered_by_idx: Dict[int, Dict[str, str]],
) -> None:
    """
    Overwrite df[seq_col] with a gapped ("-") sequence in an alignment space
    computed PER library_id (group-specific union of numbered positions).

    Rows that were not successfully numbered are left unchanged.
    """
    for _, g in df.groupby(library_col, sort=False):
        idxs = [i for i in g.index if i in numbered_by_idx]
        if not idxs:
            continue

        group_space = _build_alignment_space([numbered_by_idx[i] for i in idxs])

        for i in idxs:
            pos_map = numbered_by_idx[i]
            df.at[i, seq_col] = "".join(pos_map.get(p, "-") for p in group_space)


def compute_wild_type_from_numbered(
    g: pd.DataFrame,
    heavy_numbered_by_idx: Dict[int, Dict[str, str]],
    light_numbered_by_idx: Dict[int, Dict[str, str]],
) -> pd.Series:
    """
    Compute wild_type for one library group using precomputed numbering maps.
    """
    heavy_pos_maps = [heavy_numbered_by_idx[i] for i in g.index if i in heavy_numbered_by_idx]
    light_pos_maps = [light_numbered_by_idx[i] for i in g.index if i in light_numbered_by_idx]

    heavy_space = _build_alignment_space(heavy_pos_maps) if heavy_pos_maps else []
    light_space = _build_alignment_space(light_pos_maps) if light_pos_maps else []

    heavy_aligned: Dict[int, NumberedSeq] = {}
    light_aligned: Dict[int, NumberedSeq] = {}

    for i in g.index:
        if i in heavy_numbered_by_idx and heavy_space:
            heavy_aligned[i] = _to_numbered_seq(heavy_space, heavy_numbered_by_idx[i])
        if i in light_numbered_by_idx and light_space:
            light_aligned[i] = _to_numbered_seq(light_space, light_numbered_by_idx[i])

    heavy_cons = _consensus(list(heavy_aligned.values())) if heavy_aligned else NumberedSeq(tuple(), tuple())
    light_cons = _consensus(list(light_aligned.values())) if light_aligned else NumberedSeq(tuple(), tuple())

    scores: Dict[int, int] = {}
    for i in g.index:
        score = 0
        have_any = False
        if i in heavy_aligned and heavy_cons.positions:
            score += _distance(heavy_aligned[i], heavy_cons)
            have_any = True
        if i in light_aligned and light_cons.positions:
            score += _distance(light_aligned[i], light_cons)
            have_any = True
        if have_any:
            scores[i] = score

    wild = pd.Series(False, index=g.index)
    if not scores:
        return wild

    best_idx = min(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]
    wild.loc[best_idx] = True
    return wild


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--library-col", default="library_id")
    ap.add_argument("--heavy-col", default="heavy_sequence")
    ap.add_argument("--light-col", default="light_sequence")
    ap.add_argument("--scheme", default="imgt", choices=["martin", "kabat", "chothia", "imgt", "aho"])
    ap.add_argument("--seq-type", default="antibody", choices=["antibody", "tcr", "vnar", "vhh", "shark", "unknown"])
    ap.add_argument("--cpu", action="store_true", help="Force ANARCII to run on CPU")
    args = ap.parse_args()

    t_total0 = time.perf_counter()

    # Keep IDs as strings (preserve leading zeros)
    df = pd.read_csv(
        args.input,
        dtype={args.library_col: "string", "universal_id": "string"},
        keep_default_na=False,
    )

    for col in (args.library_col, args.heavy_col, args.light_col):
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")

    # Ensure stable integer index for mapping h_<idx>/l_<idx> <-> row
    df = df.reset_index(drop=True)

    # Run ANARCII once for all heavy and once for all light (with timings)
    heavy_numbered_by_idx = _number_all_sequences_once(
        df,
        seq_col=args.heavy_col,
        prefix="h",
        scheme=args.scheme,
        seq_type=args.seq_type,
        cpu=args.cpu,
        timing_label="heavy",
    )
    light_numbered_by_idx = _number_all_sequences_once(
        df,
        seq_col=args.light_col,
        prefix="l",
        scheme=args.scheme,
        seq_type=args.seq_type,
        cpu=args.cpu,
        timing_label="light",
    )

    # Update output sequences to gapped/aligned versions PER library_id group
    t0 = time.perf_counter()
    _update_aligned_sequence_columns_per_library(
        df, library_col=args.library_col, seq_col=args.heavy_col, numbered_by_idx=heavy_numbered_by_idx
    )
    _update_aligned_sequence_columns_per_library(
        df, library_col=args.library_col, seq_col=args.light_col, numbered_by_idx=light_numbered_by_idx
    )
    t1 = time.perf_counter()
    print(f"[timing] update aligned sequences (per library): {t1 - t0:.3f}s", flush=True)

    # Compute wild_type per library_id
    t2 = time.perf_counter()
    wild_df = (
        df.groupby(args.library_col, group_keys=False)
          .apply(lambda g: compute_wild_type_from_numbered(
              g,
              heavy_numbered_by_idx=heavy_numbered_by_idx,
              light_numbered_by_idx=light_numbered_by_idx,
          ).to_frame("wild_type"))
    )
    df["wild_type"] = wild_df["wild_type"].astype(bool)
    t3 = time.perf_counter()
    print(f"[timing] compute wild_type: {t3 - t2:.3f}s", flush=True)

    t4 = time.perf_counter()
    df.to_csv(args.output, index=False)
    t5 = time.perf_counter()
    print(f"[timing] write output CSV: {t5 - t4:.3f}s", flush=True)

    t_total1 = time.perf_counter()
    print(f"[timing] TOTAL: {t_total1 - t_total0:.3f}s", flush=True)


if __name__ == "__main__":
    main()