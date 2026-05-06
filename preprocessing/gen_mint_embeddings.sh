#!/bin/bash
#SBATCH --job-name=mint_array
#SBATCH --partition=gpu
#SBATCH --account=ccl_lab_gpu
#SBATCH --array=1-10           # Spawns 10 independent jobs (IDs 1 through 10)
#SBATCH --gres=gpu:A30:1       # Request exactly 1 GPU per job
#SBATCH --mem=64G              # 64G is plenty for one master dictionary + one model
#SBATCH --time=24:00:00
#SBATCH --output=/pub/absara/datasets/ASD/mint_embeddings/nb_affinity_2/logs/array_%A_%a.out
#SBATCH --error=/pub/absara/datasets/ASD/mint_embeddings/nb_affinity_2/logs/array_%A_%a.err

# Environment Setup
module load cuda
source /opt/apps/miniconda3/24.9.2/etc/profile.d/conda.sh
conda activate /pub/absara/myconda/24.9.2/envs/mint

# Zero-pad the Slurm Array Task ID to 3 digits (e.g., 1 becomes 001, 10 becomes 010)
PADDED_ID=$(printf "%03d" $SLURM_ARRAY_TASK_ID)

INPUT_FOLDER="/pub/absara/datasets/ASD/csv/non_binary_affinity_unique/split_csv_2/${PADDED_ID}"
OUTPUT_FOLDER="/pub/absara/datasets/ASD/mint_embeddings/nb_affinity_2/${PADDED_ID}"

echo "Starting array task $SLURM_ARRAY_TASK_ID"
echo "Processing input folder: $INPUT_FOLDER"
echo "Saving outputs to: $OUTPUT_FOLDER"

python gen_mint_embeddings.py \
-m /pub/absara/datasets/ASD/csv/non_binary_affinity_unique/asd_seq_id.csv \
-i $INPUT_FOLDER \
-o $OUTPUT_FOLDER \
-c /pub/absara/models/mint/mint.ckpt \
--batch_size 32