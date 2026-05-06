#!/bin/bash
#SBATCH --job-name=rank_test
#SBATCH --output=/pub/absara/projects/antibodies/RFlowAntibody/logs/%A.out
#SBATCH --error=/pub/absara/projects/antibodies/RFlowAntibody/logs/%A.err
#SBATCH -A CCL_LAB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --partition=standard

set -euo pipefail

source /pub/absara/projects/antibodies/RFlowAntibody/.venv/bin/activate
export PYTHONPATH=""
export PYTHONPATH="/pub/absara/models/ProteinMPNN:$PYTHONPATH"

# ---- AF3 archive staging ----
ARCHIVE="/pub/absara/datasets/ASD/af3/af3_predictions.tar.zst"

# Define temporary paths for both the structures and the YAML map
SCRATCH_BASE="${TMPDIR:-/tmp}/structures"
YAML_MAP="${TMPDIR:-/tmp}/af3_prediction_map.yaml"

echo "Creating temp directory at $SCRATCH_BASE"
mkdir -p "$SCRATCH_BASE"

# Clean up both the structures AND the yaml map when the job finishes
trap 'rm -rf "$SCRATCH_BASE" "$YAML_MAP"' EXIT

# 1. Extract tar.zst into $TMPDIR/structures
echo "Extracting archive to $SCRATCH_BASE..."
zstd -d -c "$ARCHIVE" | tar -xf - -C "$SCRATCH_BASE"

# 2. Generate the YAML map inside the temp directory
echo "Generating dynamic PDB map at $YAML_MAP..."
python /pub/absara/projects/antibodies/RFlowAntibody/scripts/generate_library_pdb_map.py \
    --base-dir "$SCRATCH_BASE" \
    --output "$YAML_MAP"
# --------------------------------

# 3. Run training
# Note: You must pass the temporary YAML map path to your datamodule.
# Ensure `data.library_pdb_map` matches the actual key in your Hydra config!
echo "Starting training..."
python /pub/absara/projects/antibodies/RFlowAntibody/src/train.py \
  debug=fdr \
  trainer.accelerator=cpu \
  trainer.devices=1 \
  trainer.precision=32 \
  data=antibody_library \
  data.library_pdb_map="$YAML_MAP"