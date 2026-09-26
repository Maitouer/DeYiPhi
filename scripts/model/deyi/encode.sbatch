#!/usr/bin/env bash
# Export train/test DeYi states for one arm.
#
#   sbatch scripts/model/deyi/encode.sbatch <task> <k> <recent>
# Job name: deyi:encode:<task>:k<k>:r<recent>
#SBATCH --partition=debug
#SBATCH --qos=debug_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120000M
#SBATCH --time=1-00:00:00
#SBATCH --job-name=deyi:encode
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
source scripts/model/common_logging.sh deyi/encode
task="${1:?usage: sbatch scripts/model/deyi/encode.sbatch <task> <k> <recent>}"
k="${2:?usage: sbatch scripts/model/deyi/encode.sbatch <task> <k> <recent>}"
recent="${3:?usage: sbatch scripts/model/deyi/encode.sbatch <task> <k> <recent>}"
case "$task" in short_video|ad|product) ;; *) echo "unknown task $task" >&2; exit 2 ;; esac
# See train.sbatch: the arm suffix cannot live in ``#SBATCH`` because it is only known here,
# and an explicit ``sbatch --job-name=`` is left alone.
if [ "${SLURM_JOB_NAME:-}" = "deyi:encode" ]; then
    scontrol update JobId="${SLURM_JOB_ID}" \
        JobName="deyi:encode:${task}:k${k}:r${recent}" 2>/dev/null || true
fi
export LOG_STAGE="model/deyi/encode"
log_dir="${MODEL_LOG_DIR:-log/model/deyi/encode}"
mkdir -p "$log_dir"
exec > "$log_dir/$task-k${k}-r${recent}.console.log" 2> "$log_dir/$task-k${k}-r${recent}.errors.log"
exec "${DEYIPHI_PYTHON:-python}" -u -m src.model.deyi.cli encode \
  --config "${DEYI_CONFIG:-config/model/deyi.yaml}" \
  --task "$task" --num-interests "$k" --recent "$recent"
