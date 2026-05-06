#!/bin/bash

#SBATCH --job-name=merge_embeddings
#SBATCH -A CCL_LAB
#SBATCH -p standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --error=/pub/absara/datasets/ASD/mint_embeddings/non_binary_affinity_unique/logs/%J.err
#SBATCH --output=/pub/absara/datasets/ASD/mint_embeddings/non_binary_affinity_unique/logs/%J.out
#SBATCH --time=1:00:00

source ../.venv/bin/activate

python merge_embeddings.py /pub/absara/datasets/ASD/mint_embeddings/nb_affinity_2/