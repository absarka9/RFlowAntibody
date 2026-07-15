#!/bin/bash
#SBATCH --job-name=RFlowAb_compare_priors
#SBATCH --output=/pub/absara/projects/antibodies/RFlowAntibody/logs/analysis/%A.out
#SBATCH --error=/pub/absara/projects/antibodies/RFlowAntibody/logs/analysis/%A.err
#SBATCH -A ABSARA_UROP_GPU
#SBATCH --gres=gpu:A30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

set -euo pipefail
module load cuda

source /pub/absara/projects/antibodies/RFlowAntibody/.venv/bin/activate

# 2. Fix the Matplotlib permission denied error
export MPLCONFIGDIR="/tmp/matplotlib_$SLURM_JOB_ID"
mkdir -p $MPLCONFIGDIR

# 3. Setup Python paths
export PYTHONPATH=""
export PYTHONPATH="/pub/absara/models/ProteinMPNN:${PYTHONPATH:-}"

# (Optional) ensure GPU 0 is used
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. Run the script
python src/analysis/compare_priors.py \
    --csv "/dfs6b/pub/absara/datasets/ASD/csv/non_binary_affinity_unique/asd_full_align_norm.csv" \
    --yaml_map "/dfs6b/pub/absara/datasets/ASD/af3/output/library_pdb_map.yaml" \
    --library_id "000196" \
    --model "mint" \
    --mint_ckpt "/dfs6b/pub/absara/models/mint/mint.ckpt" \
    --mint_config "/dfs6b/pub/absara/models/mint/data/esm2_t33_650M_UR50D.json" \
    --output "/dfs6b/pub/absara/projects/antibodies/RFlowAntibody/figures/analysis/prior_comparison.png"
    