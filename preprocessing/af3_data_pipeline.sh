#!/bin/bash
#SBATCH --job-name=af3_data_array
#SBATCH --output=/pub/absara/datasets/ASD/af3/json_data/logs/%A_%a.out  # %A is the array master job ID, %a is the task ID
#SBATCH --error=/pub/absara/datasets/ASD/af3/json_data/logs/%A_%a.err
#SBATCH --array=1-32                  # IMPORTANT: Change '8' to your actual total number of batches
#SBATCH -A CCL_LAB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # MSA generation is CPU-intensive; scale this based on your cluster limits
#SBATCH --mem=64G                    # Adjust memory based on MSA depth/database size
#SBATCH --time=24:00:00              # Adjust expected time limit
#SBATCH --partition=standard         # IMPORTANT: Change to your cluster's specific CPU partition

# Define the batch directory name using the SLURM array ID
BATCH_ID="batch_${SLURM_ARRAY_TASK_ID}"

# Set up the dynamic input and output paths
INPUT_DIR="/pub/absara/datasets/ASD/af3/json_input/${BATCH_ID}"
OUTPUT_DIR="/pub/absara/datasets/ASD/af3/json_data/${BATCH_ID}"

# Ensure the output directory exists before running
mkdir -p "${OUTPUT_DIR}"

echo "Starting data pipeline for ${BATCH_ID} on node $(hostname)"

# Note: I added a leading slash to 'pub/' in your output_dir to make it an absolute path
# Note: I comma-separated the bind mounts (-B) and --env vars to match standard Singularity syntax
/opt/apps/singularity/3.11.3/bin/singularity exec \
    --nv \
    -B "/pub/absara/models/af3/databases,/pub/absara/models/af3/weights,/pub/absara/datasets/ASD/af3" \
    --env CUDA_VISIBLE_DEVICES=0,NVIDIA_VISIBLE_DEVICES=0 \
    /pub/absara/models/af3/alphafold3_40gb.sif \
    python /app/alphafold/run_alphafold.py \
    --input_dir="${INPUT_DIR}" \
    --db_dir=/pub/absara/models/af3/databases \
    --output_dir="${OUTPUT_DIR}" \
    --model_dir=/pub/absara/models/af3/weights \
    --flash_attention_implementation=triton \
    --run_data_pipeline=True \
    --run_inference=False

echo "Finished data pipeline for ${BATCH_ID}"