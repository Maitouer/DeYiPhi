#!/usr/bin/env bash
# Train one DeYi arm.
#
#   sbatch scripts/model/deyi/train.sbatch <task> <k> <recent> [extra CLI flags...]
#   sbatch scripts/model/deyi/train.sbatch ad 4 32 --item-temperature 0.10 --lambda-div 0.02
#
# Output: output/model/deyi/<task>/k<k>_r<recent>/  (recent is part of the arm name because
# it changes what is trained).
# Job name: deyi:train:<task>:k<k>:r<recent>
#SBATCH --partition=gpu
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=11
#SBATCH --mem=125000M
#SBATCH --time=1-00:00:00
#SBATCH --job-name=deyi:train
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
source scripts/model/common_logging.sh deyi/train
task="${1:?usage: sbatch scripts/model/deyi/train.sbatch <task> <k> <recent> [flags]}"
k="${2:?usage: sbatch scripts/model/deyi/train.sbatch <task> <k> <recent> [flags]}"
recent="${3:?usage: sbatch scripts/model/deyi/train.sbatch <task> <k> <recent> [flags]}"
shift 3
case "$task" in short_video|ad|product) ;; *) echo "unknown task $task" >&2; exit 2 ;; esac
# ``#SBATCH`` is fixed at submit time, so the arm is appended here: the running job then shows
# up in squeue/sacct as the exact arm it is training. Only when the name is still this script's
# default -- an explicit ``sbatch --job-name=`` is never clobbered.
if [ "${SLURM_JOB_NAME:-}" = "deyi:train" ]; then
    scontrol update JobId="${SLURM_JOB_ID}" \
        JobName="deyi:train:${task}:k${k}:r${recent}" 2>/dev/null || true
fi
export LOG_STAGE="model/deyi/train"
log_dir="${MODEL_LOG_DIR:-log/model/deyi/train}"
mkdir -p "$log_dir"
exec > "$log_dir/$task-k${k}-r${recent}.console.log" 2> "$log_dir/$task-k${k}-r${recent}.errors.log"
# ``--resume`` is always on: the trainer writes a checkpoint after every epoch, so a re-submitted
# stage continues where it stopped instead of dying with "fresh DeYi training needs an unused arm
# directory". With no checkpoint on disk the flag is a no-op, so a genuinely fresh arm is unaffected.
exec "${DEYIPHI_PYTHON:-python}" -u -m src.model.deyi.cli train \
  --config "${DEYI_CONFIG:-config/model/deyi.yaml}" \
  --task "$task" --num-interests "$k" --recent "$recent" --resume "$@"
