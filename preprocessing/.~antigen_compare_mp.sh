#!/bin/bash
#SBATCH --job-name=antigen_compare_mp
#SBATCH --partition=standard
#SBATCH --account=ccl_lab
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/pub/absara/projects/antibodies/RFlowAntibody/logs/preprocessing/%J.out
#SBATCH --error=/pub/absara/projects/antibodies/RFlowAntibody/logs/preprocessing/%J.err

# 1. Fix the matplotlib permission denied error by pointing to a local tmp folder
export MPLCONFIGDIR=/tmp/matplotlib_cache_${SLURM_JOB_ID}
mkdir -p $MPLCONFIGDIR

# 2. Use the absolute path to your virtual environment
source /dfs6b/pub/absara/projects/antibodies/RFlowAntibody/.venv/bin/activate

# 3. Run the python script
python preprocessing/antigen_compare_mp.py \
    --input "/dfs6b/pub/absara/datasets/ASD/csv/non_binary_affinity_unique/asd_full_align_norm.csv" \
    --output "/dfs6b/pub/absara/projects/antibodies/RFlowAntibody/figures/preprocessing/run1" \
    --cores $SLURM_CPUS_PER_TASK \
    --ref_seq "MKAILVVLLYTFATANADTLCIGYHANNSTDTVDTVLEKNVTVTHSVNLLEDKHNGKLCKLRGVAPLHLGKCNIAGWILGNPECESLSTASSWSYIVETPSSDNGTCYPGDFIDYEELREQLSSVSSFERFEIFPKTSSWPNHDSNKGVTAACPHAGAKSFYKNLIWLVKKGNSYPKLSKSYINDKGKEVLVLWGIHHPSTSADQQSLYQNADTYVFVGSSRYSKKFKPEIAIRPKVRDQEGRMNYYWTLVEPGDKITFEATGNLVVPRYAFAMERNAGSGIIISDTPVHDCNTTCQTPKGAINTSLPFQNIHPITIGKCPKYVKSTKLRLATGLRNIPSIQSRGLFGAIAGFIEGGWTGMVDGWYGYHHQNEQGSGYAADLKSTQNAIDEITNKVNSVIEKMNTQFTAVGKEFNHLEKRIENLNKKVDDGFLDIWTYNAELLVLLENERTLDYHDSNVKNLYEKVRSQLKNNAKEIGNGCFEFYHKCDNTCMESVKNGTYDYPKYSEEAKLNREEIDGVKLESTRIYQILAIYSTVASSLVLVVSLGAISFWMCSNGSLQCRICI"

# Clean up the temp matplotlib directory
rm -rf $MPLCONFIGDIR

echo "Done!"