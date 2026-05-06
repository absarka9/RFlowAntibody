#!/bin/bash
#SBATCH --job-name=af3_inference
#SBATCH --output=/pub/absara/datasets/ASD/af3/output/logs/%A_%a.out
#SBATCH --error=/pub/absara/datasets/ASD/af3/output/logs/%A_%a.err
#SBATCH --array=1-8                  # We are compressing 32 CPU batches into 8 GPU batches
#SBATCH --account=ccl_lab_gpu
#SBATCH --gres=gpu:A30:1       # Request exactly 1 GPU per job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # AF3 inference still uses some CPU for data loading
#SBATCH --mem=64G                    
#SBATCH --time=48:00:00              # GPU inference can take time depending on structure size
#SBATCH --partition=gpu              # IMPORTANT: Change to your specific GPU partition

# ==============================================================================
# BATCH MAPPING & EXCLUSION ZONE
# ==============================================================================
# Group your original 32 CPU batches into the 8 GPU tasks here.
# If a CPU batch is NOT complete (e.g., batch 11), simply delete it from the list.
# For example, if you change (9 10 11 12) to (9 10 12), batch 11 will be safely ignored.

case ${SLURM_ARRAY_TASK_ID} in
    1) CPU_BATCHES=(1 2 3 4) ;;
    2) CPU_BATCHES=(5 6 7 8) ;;
    3) CPU_BATCHES=(9 10 11 12) ;;
    4) CPU_BATCHES=(13 14 15 16) ;;
    5) CPU_BATCHES=(17 18 19 20) ;;
    6) CPU_BATCHES=(21 22 23 24) ;;
    7) CPU_BATCHES=(25 26 27 28) ;;
    8) CPU_BATCHES=(29 30 31 32) ;;
    *) echo "Invalid task ID"; exit 1 ;;
esac

# ==============================================================================

# Define global paths
JSON_DATA_BASE="/pub/absara/datasets/ASD/af3/json_data2/json_data"
GPU_INPUT_DIR="/pub/absara/datasets/ASD/af3/json_inference/gpu_batch_${SLURM_ARRAY_TASK_ID}"
OUTPUT_DIR="/pub/absara/datasets/ASD/af3/output/gpu_batch_${SLURM_ARRAY_TASK_ID}"

# Ensure directories exist
mkdir -p "${GPU_INPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Clear any old symlinks if this task is being rerun
rm -f "${GPU_INPUT_DIR}"/*.json

echo "Setting up GPU Task ${SLURM_ARRAY_TASK_ID}..."
echo "Pulling completed _data.json files from CPU batches: ${CPU_BATCHES[*]}"

VALID_FILES=0

# Loop through the assigned CPU batches and symlink their completed _data.json files
for batch_num in "${CPU_BATCHES[@]}"; do
    TARGET_DIR="${JSON_DATA_BASE}/batch_${batch_num}"
    
    if [ -d "${TARGET_DIR}" ]; then
        # Find all _data.json files in that batch and symlink them to our GPU input folder
        for data_file in "${TARGET_DIR}"/*/*_data.json; do
            # Ensure the file exists (handles the case where the glob returns no matches)
            if [ -f "${data_file}" ]; then
                ln -s "${data_file}" "${GPU_INPUT_DIR}/$(basename "${data_file}")"
                VALID_FILES=$((VALID_FILES+1))
            fi
        done
    else
        echo "Warning: Directory ${TARGET_DIR} does not exist. Skipping."
    fi
done

echo "Found ${VALID_FILES} valid data payloads to run inference on."

if [ "$VALID_FILES" -eq 0 ]; then
    echo "No valid files found for GPU Task ${SLURM_ARRAY_TASK_ID}. Exiting."
    exit 0
fi

echo "Starting GPU inference on node $(hostname)"

# Run AF3 Inference
/opt/apps/singularity/3.11.3/bin/singularity exec \
    --nv \
    -B "/pub/absara/models/af3/databases,/pub/absara/models/af3/weights,/pub/absara/datasets/ASD/af3" \
    /pub/absara/models/af3/alphafold3_40gb.sif \
    python /app/alphafold/run_alphafold.py \
    --input_dir="${GPU_INPUT_DIR}" \
    --db_dir=/pub/absara/models/af3/databases \
    --output_dir="${OUTPUT_DIR}" \
    --model_dir=/pub/absara/models/af3/weights \
    --flash_attention_implementation=triton \
    --run_data_pipeline=False \
    --run_inference=True

echo "Finished GPU inference for Task ${SLURM_ARRAY_TASK_ID}"