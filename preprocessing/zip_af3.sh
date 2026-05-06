#!/bin/bash
#SBATCH --job-name=zip_af3
#SBATCH --output=/pub/absara/projects/antibodies/RFlowAntibody/logs/%j.out
#SBATCH --error=/pub/absara/projects/antibodies/RFlowAntibody/logs/%j.err
#SBATCH -A CCL_LAB
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=standard

set -euo pipefail

# ---- USER SETTINGS ----
BASE_DIR="/dfs6b/pub/absara/datasets/ASD/af3/output"
OUT_TAR="/dfs6b/pub/absara/datasets/ASD/af3/af3_predictions.tar.zst"
# -----------------------

if [[ ! -d "$BASE_DIR" ]]; then
  echo "ERROR: base dir not found: $BASE_DIR" >&2
  exit 1
fi

tmp_list="$(mktemp)"
trap 'rm -f "$tmp_list"' EXIT

# Change to the base directory first
cd "$BASE_DIR"

# Run find using the current directory (.) so paths in tmp_list are relative
find . \
  -type f \
  -path "*/gpu_batch_*/[0-9][0-9][0-9][0-9][0-9][0-9]/*" \
  -not -path "*/gpu_batch_*/[0-9][0-9][0-9][0-9][0-9][0-9]/*/*" \
  -print > "$tmp_list"

# Run tar (you no longer need -C since you are already in the directory)
tar -cf - -T "$tmp_list" | zstd -T0 -3 -o "$OUT_TAR"

echo "Wrote: $OUT_TAR"