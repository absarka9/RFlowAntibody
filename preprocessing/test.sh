#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gpu
#SBATCH --account=ccl_lab_gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=./%J.out
#SBATCH --error=./%J.err

module load cuda
source ../.venv/bin/activate

protenix pred -h
