#!/usr/bin/env bash
# Tiny end-to-end phi run: the same stages on a small vocabulary, into a scratch directory.
#
#   sbatch scripts/model/phi/smoke.sbatch [task] [k] [recent]
#
# It exists to prove the chain runs (and to time it) before paying for a production vocabulary.
# The scratch output directory is set explicitly, so a smoke run can never overwrite production
# artifacts, and the whole thing is meant to fit in minutes on the debug MIG.
# Job name: phi:smoke:<task>
#SBATCH --partition=debug
#SBATCH --qos=debug_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60000M
#SBATCH --time=04:00:00
#SBATCH --job-name=phi:smoke
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
source scripts/model/common_logging.sh phi/smoke
task="${1:-product}"
k="${2:-4}"
recent="${3:-16}"
if [ "${SLURM_JOB_NAME:-}" = "phi:smoke" ]; then
    scontrol update JobId="${SLURM_JOB_ID}" JobName="phi:smoke:${task}" 2>/dev/null || true
fi
export PHI_DEYI_K="$k" PHI_DEYI_RECENT="$recent"
export PHI_OUTPUT_DIR="${PHI_OUTPUT_DIR:-output/model/phi-smoke}"
export LOG_STAGE="model/phi/smoke"
python_bin="${DEYIPHI_PYTHON:-python}"
flags=(--task "$task" --vocabulary 32 --reservoir 2048 --head 64 --tail 256 --calibration 16
       --iterations 5 --router-epochs 5)
for stage in plan reservoir estimate codebook router; do
    "$python_bin" -u -m src.model.phi.cli "$stage" "${flags[@]}"
done
for split in train test; do
    "$python_bin" -u -m src.model.phi.cli tokenize "${flags[@]}" --split "$split"
done
"$python_bin" -u -m src.model.phi.cli route "${flags[@]}" --split train
"$python_bin" -u -m src.model.phi.cli audit "${flags[@]}" --split train
echo "[$(date '+%H:%M:%S')] smoke finished; artifacts under $PHI_OUTPUT_DIR"

