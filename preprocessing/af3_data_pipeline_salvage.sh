#!/bin/bash
#SBATCH --job-name=af3_data_array_salvage
#SBATCH --output=/pub/absara/datasets/ASD/af3/json_data/logs/%A_%a.out  # %A is the array master job ID, %a is the task ID
#SBATCH --error=/pub/absara/datasets/ASD/af3/json_data/logs/%A_%a.err
#SBATCH --array=3,5                  # IMPORTANT: Change '8' to your actual total number of batches
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
PENDING_DIR="/pub/absara/datasets/ASD/af3/json_input/${BATCH_ID}_pending"

# Define the global base directory to search across ALL batches
JSON_DATA_BASE="/pub/absara/datasets/ASD/af3/json_data"

# Ensure the output and pending directories exist before running
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PENDING_DIR}"

# Clear out any old symlinks or lists from previous attempts in the pending directory
rm -f "${PENDING_DIR}"/*.json
rm -f "${PENDING_DIR}/completed_files.txt"

echo "Scanning for globally completed structures in ${JSON_DATA_BASE}..."
COMPLETED_LIST="${PENDING_DIR}/completed_files.txt"

# Find ALL _data.json files across any batch or subfolder and save to a temporary list
# Doing this once is much faster than running 'find' inside the loop
find "${JSON_DATA_BASE}" -type f -name "*_data.json" > "${COMPLETED_LIST}"

# Loop through all original inputs and symlink only the unprocessed ones
PENDING_COUNT=0
for input_file in "${INPUT_DIR}"/*.json; do
    # Gracefully handle the case where the directory is empty
    [ -e "$input_file" ] || continue
    
    filename=$(basename "$input_file")
    base_id="${filename%.json}"
    
    # Check if this file exists ANYWHERE in the global completed list
    # The leading slash ensures we match the exact filename (e.g., /000200_data.json)
    if ! grep -q "/${base_id}_data.json" "${COMPLETED_LIST}"; then
        ln -s "$input_file" "${PENDING_DIR}/${filename}"
        PENDING_COUNT=$((PENDING_COUNT+1))
    fi
done

echo "Found ${PENDING_COUNT} pending structures to process on node $(hostname)"

# If all files are processed, exit cleanly without spinning up the container
if [ "$PENDING_COUNT" -eq 0 ]; then
    echo "Batch ${BATCH_ID} is already 100% complete! Exiting early to save resources."
    rm -rf "${PENDING_DIR}"
    exit 0
fi

echo "Starting data pipeline for the remaining ${PENDING_COUNT} files..."

# Run AF3, pointing it explicitly to the newly created PENDING_DIR
/opt/apps/singularity/3.11.3/bin/singularity exec \
    --nv \
    -B "/pub/absara/models/af3/databases,/pub/absara/models/af3/weights,/pub/absara/datasets/ASD/af3" \
    --env CUDA_VISIBLE_DEVICES=0,NVIDIA_VISIBLE_DEVICES=0 \
    /pub/absara/models/af3/alphafold3_40gb.sif \
    python /app/alphafold/run_alphafold.py \
    --input_dir="${PENDING_DIR}" \
    --db_dir=/pub/absara/models/af3/databases \
    --output_dir="${OUTPUT_DIR}" \
    --model_dir=/pub/absara/models/af3/weights \
    --flash_attention_implementation=triton \
    --run_data_pipeline=True \
    --run_inference=False

echo "Finished data pipeline chunk for ${BATCH_ID}"