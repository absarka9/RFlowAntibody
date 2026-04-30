"""generate_library_pdb_map.py – Scan an AlphaFold3 output directory and write
a ``library_pdb_map.yaml`` mapping each library_id to its wildtype CIF file.

Directory layout assumed
------------------------
    <base_dir>/
        gpu_batch_1/
            <library_id>/
                <library_id>_model.cif
        gpu_batch_2/
            <library_id>/
                <library_id>_model.cif
        ...

Each batch sub-directory is named ``gpu_batch_<N>`` where ``<N>`` is an integer.
If the same ``library_id`` appears in multiple batch directories the entry from
the **lowest** batch number is used.

Optionally, a master CSV can be supplied via ``--master-csv``; when provided
only ``library_id`` values found in that CSV are written to the map (others are
silently ignored).  This is useful for restricting the map to the libraries
actually used in training.

Usage
-----
    python scripts/generate_library_pdb_map.py \\
        --base-dir /pub/absara/datasets/ASD/af3/output \\
        --output   /path/to/library_pdb_map.yaml

    # Restrict to libraries present in master CSV
    python scripts/generate_library_pdb_map.py \\
        --base-dir   /pub/absara/datasets/ASD/af3/output \\
        --master-csv /path/to/master.csv \\
        --output     /path/to/library_pdb_map.yaml

    # Dry-run: print map to stdout without writing a file
    python scripts/generate_library_pdb_map.py \\
        --base-dir /pub/absara/datasets/ASD/af3/output \\
        --dry-run
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Set

LOG = logging.getLogger(__name__)

# Regex that matches "gpu_batch_<N>" and captures the integer N.
_BATCH_RE = re.compile(r"^gpu_batch_(\d+)$")


def scan_cif_structures(
    base_dir: Path,
    allowed_ids: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """Scan *base_dir* for ``gpu_batch_*`` sub-directories and return a
    ``library_id → cif_path`` mapping.

    Parameters
    ----------
    base_dir:
        Root directory that contains ``gpu_batch_*`` sub-directories.
    allowed_ids:
        If not ``None``, only library IDs present in this set are included in
        the output map.

    Returns
    -------
    dict
        ``{library_id: absolute_cif_path}`` using the lowest-batch-number
        entry when duplicates exist.
    """
    if not base_dir.is_dir():
        raise NotADirectoryError(f"base_dir does not exist or is not a directory: {base_dir}")

    # Collect (batch_num, library_id, cif_path) triples from all batch dirs.
    candidates: list[tuple[int, str, Path]] = []

    for batch_dir in sorted(base_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
        m = _BATCH_RE.match(batch_dir.name)
        if not m:
            continue
        batch_num = int(m.group(1))

        for lib_dir in batch_dir.iterdir():
            if not lib_dir.is_dir():
                continue
            lib_id = lib_dir.name
            cif_path = lib_dir / f"{lib_id}_model.cif"
            if cif_path.is_file():
                candidates.append((batch_num, lib_id, cif_path))
            else:
                LOG.debug(
                    "Expected CIF not found: %s (skipping)", cif_path
                )

    # Build map: for each library_id keep the entry with the lowest batch number.
    best: Dict[str, tuple[int, Path]] = {}
    for batch_num, lib_id, cif_path in candidates:
        if lib_id not in best or batch_num < best[lib_id][0]:
            best[lib_id] = (batch_num, cif_path)

    # Apply optional filter
    result: Dict[str, str] = {}
    for lib_id, (batch_num, cif_path) in sorted(best.items()):
        if allowed_ids is not None and lib_id not in allowed_ids:
            continue
        result[lib_id] = str(cif_path)
        LOG.debug("  %s  →  %s  (batch %d)", lib_id, cif_path, batch_num)

    return result


def _load_allowed_ids(csv_path: Path) -> Set[str]:
    """Return the set of library_id values from *csv_path*."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to read a master CSV.  "
            "Install it with: pip install pandas"
        ) from exc

    df = pd.read_csv(csv_path)
    col = "library_id"
    if col not in df.columns:
        raise ValueError(
            f"master CSV does not contain a '{col}' column.  "
            f"Found columns: {list(df.columns)}"
        )
    return set(df[col].astype(str).unique())


def _write_yaml(mapping: Dict[str, str], output_path: Path) -> None:
    """Write *mapping* to *output_path* as YAML."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to write the output map.  "
            "Install it with: pip install pyyaml"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        yaml.dump(mapping, fh, default_flow_style=False, sort_keys=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan an AlphaFold3 output directory and generate a "
            "library_pdb_map YAML file for AntibodyLibraryData."
        )
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        metavar="DIR",
        help=(
            "Root directory containing gpu_batch_* sub-directories, e.g. "
            "/pub/absara/datasets/ASD/af3/output"
        ),
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help=(
            "Path where the YAML map will be written.  "
            "Defaults to <base_dir>/library_pdb_map.yaml.  "
            "Ignored when --dry-run is set."
        ),
    )
    parser.add_argument(
        "--master-csv",
        metavar="CSV",
        default=None,
        help=(
            "Optional master CSV whose library_id column restricts which "
            "libraries are included in the output map."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the map to stdout instead of writing a file.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    base_dir = Path(args.base_dir).resolve()

    allowed_ids: Optional[Set[str]] = None
    if args.master_csv:
        csv_path = Path(args.master_csv).resolve()
        LOG.info("Loading library IDs from master CSV: %s", csv_path)
        allowed_ids = _load_allowed_ids(csv_path)
        LOG.info("  %d unique library IDs found.", len(allowed_ids))

    LOG.info("Scanning for CIF structures under: %s", base_dir)
    mapping = scan_cif_structures(base_dir, allowed_ids=allowed_ids)

    if not mapping:
        LOG.warning(
            "No CIF structures found under '%s'.  "
            "Check that the directory contains gpu_batch_* sub-directories "
            "with the expected layout: "
            "{base_dir}/gpu_batch_N/{library_id}/{library_id}_model.cif",
            base_dir,
        )
        return 1

    LOG.info("Found %d library → CIF mapping(s).", len(mapping))

    if args.dry_run:
        try:
            import yaml
            print(yaml.dump(mapping, default_flow_style=False, sort_keys=True), end="")
        except ImportError:
            for lib_id, cif_path in sorted(mapping.items()):
                print(f"{lib_id}: {cif_path}")
        return 0

    output_path = Path(args.output).resolve() if args.output else base_dir / "library_pdb_map.yaml"
    LOG.info("Writing map to: %s", output_path)
    _write_yaml(mapping, output_path)
    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
