# Property-aware Transport for Protein Optimization

We propose a conditional flow-matching framework for protein fitness prediction that learns a property-aware landscape on top of pretrained PLMs. By introducing an energy function tied to assay-specific fitness and a rank-consistent objective, it shapes the flow so that mutants are ordered coherently by their functional properties. In addition, the property-aware steering gate focuses learning on relevant positions, improving performance on diverse protein engineering tasks. Across ProteinGym and additional protein engineering benchmarks, RankFlow consistently matches or surpasses state-of-the-art supervised methods under the same training protocols, using far fewer trainable parameters than full PLM fine-tuning and offering a robust, transferable approach to protein fitness prediction.

## Requirements
- **Python**: 3.9  
- **PyTorch**: 2.4.1 with CUDA 11.8 (`pytorch`, `torchvision`, `torchaudio`, `pytorch-cuda=11.8`)  
- **Key libraries (pip)**:
  - `pytorch-lightning` / `lightning`
  - `torch-geometric`, `torch-scatter`, `torch-sparse`, `torch-cluster`
  - `fair-esm`
  - `openfold`
  - `hydra-core`, `omegaconf`, `hydra-optuna-sweeper`, `hydra-colorlog`
  - `optuna`
  - `numpy`, `scipy`, `pandas`

All exact versions are pinned in `environment.yml`.

## Installation

Create and activate the Conda environment:
```
>> conda env create --file environment.yml
>> conda activate rankflow
```

## Quick Start
Before training, place DMS and structure data in `./data`, and modify the data configuration file `./configs/data/proteingym.yaml`. Using a subfolder structure, we can, for example, organize the ProteinGym data as follows: create a main folder ProteinGym, and inside it place the reference file DMS_substitutions.csv along with two subfolders: DMS_ProteinGym_substitutions and ProteinGym_AF2_structures.

We can train and test RankFlow as follows.

```
>> bash ./scripts/run.sh
```

The program will start training with the default settings. To modify any hyperparameters, edit `configs/model/RankFlow.yaml` or `configs/rankflow.yaml`. In the example provided, we train on the assay `SPG1_STRSG_Olson_2014`. If you want to train on a different assay, update the assay_index field in `configs/data/proteingym.yaml`.

## Training with AntibodyLibraryData (master CSV + per-library PDBs)

`AntibodyLibraryData` allows you to train RankFlow directly from a master CSV
that groups antibody variants by library, without needing per-variant mutation
strings.  Each library becomes one assay for RankFlow's list-wise ranking.

### 1. Prepare the master CSV

The CSV must contain at least the following columns (extra columns are ignored):

| Column             | Description                                      |
|--------------------|--------------------------------------------------|
| `library_id`       | Groups rows into assays                          |
| `heavy_sequence`   | VH (or VHH) amino-acid string; empty for Ag-only |
| `light_sequence`   | VL amino-acid string; empty for nanobodies       |
| `antigen_sequence` | Antigen amino-acid string; may be empty          |
| `norm_affinity`    | Normalised affinity score (float, higher = better) |

Optionally include a fold column (e.g. `fold_random_5`) for cross-validation.

### 2. Prepare wildtype PDB structures

For each `library_id`, provide a predicted (or experimental) wildtype complex
structure.  RankFlow reuses one structure per library for all its variants.

#### Option A – Manual map

Create a YAML (or JSON) mapping file, for example `library_pdb_map.yaml`:

```yaml
lib_001: /data/structures/lib_001_wt.pdb
lib_002: /data/structures/lib_002_wt.pdb
```

#### Option B – Auto-generate from AlphaFold3 batch output

If your structures were produced by AlphaFold3 and are organised as:

```
<base_dir>/
    gpu_batch_1/<library_id>/<library_id>_model.cif
    gpu_batch_2/<library_id>/<library_id>_model.cif
    ...
```

use the bundled helper script to scan the directory tree and write the map
automatically:

```bash
python scripts/generate_library_pdb_map.py \
    --base-dir /pub/absara/datasets/ASD/af3/output \
    --output   /path/to/library_pdb_map.yaml
```

If the same `library_id` appears in multiple `gpu_batch_*` directories the
entry from the **lowest** batch number is used.

Optionally restrict the output to only the libraries present in your master CSV:

```bash
python scripts/generate_library_pdb_map.py \
    --base-dir   /pub/absara/datasets/ASD/af3/output \
    --master-csv /path/to/master.csv \
    --output     /path/to/library_pdb_map.yaml
```

Run `python scripts/generate_library_pdb_map.py --help` for all options.

### 3. Configure and run training

Edit `configs/data/antibody_library.yaml`:

```yaml
data_dir: /path/to/your/data/
master_csv: antibody_libraries.csv       # relative to data_dir, or absolute
library_pdb_map: library_pdb_map.yaml   # relative to data_dir, or absolute
chain_order: [H, L, A]                  # omit L for nanobody libraries
split_type: random                      # uses fold_random_5 column if present
split_index: 0
```

Then launch training with:

```bash
python src/train.py data=antibody_library
```

Or override parameters on the command line:

```bash
python src/train.py data=antibody_library \
    data.master_csv=/abs/path/to/master.csv \
    data.library_pdb_map=/abs/path/to/pdb_map.yaml \
    data.chain_order="[H,A]"
```

### Optional toggles: `use_wild_type_row`, `filter_modal_length`, `filter_wild_type_length`, and `min_variants_per_library`

These optional flags let you control wildtype-source selection, modal-length
column filtering, wildtype-length sequence filtering, and minimum library size.

#### `use_wild_type_row` (default: `True`)

When `True`, the datamodule looks for a row whose `wild_type` column equals
`True` within each `library_id` group and uses that row's
`heavy_sequence` / `light_sequence` / `antigen_sequence` values as the
wildtype chain sequences for steering vectors and reference tokens.  If no
such row exists the code falls back to extracting sequences from the PDB
(original behaviour).

Set to `False` to always derive wildtype sequences from the PDB.

```yaml
use_wild_type_row: True   # default – use CSV wild_type row when present
```

#### `filter_modal_length` (default: `False`)

When `True`, each library's rows are pre-filtered to only those where
`modal_length == True` before the train/validation split and label
construction.  This is useful when you want to restrict training to
variants whose CDR lengths match the most common (modal) length in the
library, which can reduce noise from non-modal insertions/deletions.

If the filter removes **all** rows from a library, that library is skipped
with a warning rather than raising an error.

```yaml
filter_modal_length: True   # only use modal-length rows
```

#### `filter_wild_type_length` (default: `False`)

When `True`, each library's rows are pre-filtered to only those where the
`heavy_sequence` / `light_sequence` / `antigen_sequence` lengths match that
library's resolved wildtype chain lengths before train/validation splitting.
Wildtype lengths are taken from the same wildtype source used by the
datamodule (`wild_type` row when available and enabled, otherwise PDB).

If the filter removes **all** rows from a library, that library is skipped
with a warning rather than raising an error.

```yaml
filter_wild_type_length: True   # only keep wildtype-length variants
```

#### `min_variants_per_library` (default: `0`)

Sets a per-library minimum retained-variant cutoff. After optional row-level
filters (`filter_modal_length`, `filter_wild_type_length`) are applied, any
library with fewer variants than this value is skipped.

Set to `0` to disable.

```yaml
min_variants_per_library: 100   # skip libraries with <100 retained variants
```

All flags can be combined:

```bash
python src/train.py data=antibody_library \
    data.use_wild_type_row=True \
    data.filter_modal_length=True \
    data.filter_wild_type_length=True \
    data.min_variants_per_library=100
```

## Contact
Thank you for your interest in our work!

Please feel free to ask about any questions about the algorithms, codes, as well as problems encountered in running them by contacting l.yu@latrobe.edu.au.
