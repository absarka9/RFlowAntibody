#!/bin/bash
#SBATCH --job-name=anarcii_align
#SBATCH --partition=gpu
#SBATCH --account=ccl_lab_gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/pub/absara/datasets/ASD/csv/non_binary_affinity_unique/logs/%J.out
#SBATCH --error=/pub/absara/datasets/ASD/csv/non_binary_affinity_unique/logs/%J.err

source ../.venv/bin/activate

python align.py --input /pub/absara/datasets/ASD/csv/non_binary_affinity_unique/asd_seq_id_test_000200.csv --output /pub/absara/datasets/ASD/csv/non_binary_affinity_unique/align_test.csv --seq-type unknown

echo "ANARCII alignment and wt calling done"