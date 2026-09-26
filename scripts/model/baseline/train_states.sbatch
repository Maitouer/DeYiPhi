#!/usr/bin/env bash
# Train and export the memory states of one learned controlled baseline (04_baseline.md).
#
#   sbatch scripts/model/baseline/train_states.sbatch <chronicle|hicogen> [task] [recent] [K]
#
# Both stages run with the same ``--method``, so the trainer and the exporter agree on the arm
# directory (output/model/compress/<method>/<dataset>/<task>/k<K>_r<R>) and on the model.
# Job name: baseline:states
#SBATCH --partition=gpu
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120000M
#SBATCH --time=4:00:00
#SBATCH --job-name=baseline:states
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
method="${1:?usage: sbatch scripts/model/baseline/train_states.sbatch <chronicle|hicogen> [task] [recent] [K]}"
task="${2:-product}"
recent="${3:-16}"
interests="${4:-4}"
case "$method" in chronicle|hicogen) ;; *) echo "unknown method $method" >&2; exit 2 ;; esac
source scripts/model/common_logging.sh baseline/states
export LOG_STAGE="model/baseline/states"
if [ "${SLURM_JOB_NAME:-}" = "baseline:states" ]; then
    scontrol update JobId="${SLURM_JOB_ID}" \
        JobName="baseline:${method}:${task}:k${interests}:r${recent}" 2>/dev/null || true
fi
PYTHON_BIN="${DEYIPHI_PYTHON:-python}"
log_dir="${MODEL_LOG_DIR:-log/model/baseline/states}"
mkdir -p "$log_dir"
exec > "$log_dir/${method}-${task}-k${interests}-r${recent}.console.log" \
     2> "$log_dir/${method}-${task}-k${interests}-r${recent}.errors.log"
"$PYTHON_BIN" -u -m src.model.deyi.cli train \
    --config config/model/deyi.yaml --task "$task" \
    --num-interests "$interests" --recent "$recent" --method "$method"
"$PYTHON_BIN" -u -m src.model.deyi.cli encode \
    --config config/model/deyi.yaml --task "$task" \
    --num-interests "$interests" --recent "$recent" --method "$method"
