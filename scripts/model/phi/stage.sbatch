#!/usr/bin/env bash
# Run one phi stage (rerun a single step without redoing the chain).
#
#   sbatch scripts/model/phi/stage.sbatch <stage> <task> [k] [recent] [extra CLI flags...]
#   sbatch scripts/model/phi/stage.sbatch estimate product 4 16 --head 512 --tail 2048
#
# Stages: plan reservoir estimate codebook router tokenize route audit
# Job name: phi:<stage>:<task>
#SBATCH --partition=debug
#SBATCH --qos=debug_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120000M
#SBATCH --time=1-00:00:00
#SBATCH --job-name=phi:stage
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
stage="${1:?usage: sbatch scripts/model/phi/stage.sbatch <stage> <task> [k] [recent] [flags]}"
task="${2:?usage: sbatch scripts/model/phi/stage.sbatch <stage> <task> [k] [recent] [flags]}"
k="${3:-4}"
recent="${4:-16}"
shift 4 2>/dev/null || shift $#
source scripts/model/common_logging.sh "phi/$stage"
if [ "${SLURM_JOB_NAME:-}" = "phi:stage" ]; then
    scontrol update JobId="${SLURM_JOB_ID}" JobName="phi:${stage}:${task}" 2>/dev/null || true
fi
export PHI_DEYI_K="$k" PHI_DEYI_RECENT="$recent"
export LOG_STAGE="model/phi/$stage"
exec "${DEYIPHI_PYTHON:-python}" -u -m src.model.phi.cli "$stage" \
    --config "${PHI_CONFIG:-config/model/phi.yaml}" --task "$task" "$@"

