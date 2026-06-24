#!/bin/bash
#SBATCH --job-name=RFlowAb_train
#SBATCH --output=/pub/absara/projects/antibodies/RFlowAntibody/logs/%A.out
#SBATCH --error=/pub/absara/projects/antibodies/RFlowAntibody/logs/%A.err
#SBATCH -A ABSARA_UROP_GPU
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

set -euo pipefail

source /pub/absara/projects/antibodies/RFlowAntibody/.venv/bin/activate
export PYTHONPATH=""
export PYTHONPATH="/pub/absara/models/ProteinMPNN:${PYTHONPATH:-}"

# (Optional) ensure GPU 0 is used
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="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"
echo "Key length: ${WANDB_API_KEY}"

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

# 3. Run training on GPU
echo "Starting training..."
python /pub/absara/projects/antibodies/RFlowAntibody/src/train.py \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.precision=bf16-mixed \
  data=antibody_library \
  data.library_pdb_map="$YAML_MAP"