#!/usr/bin/env bash
# Export the memory states of an already trained baseline arm (no training).
#
#   sbatch scripts/model/baseline/encode_states.sbatch <chronicle|hicogen> [task] [recent] [K]
#
# Split out from train_states.sbatch because the encoder is a separate stage: a crashed or
# interrupted export can be re-run against the finished checkpoint without retraining.
#SBATCH --partition=gpu
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120000M
#SBATCH --time=1:00:00
#SBATCH --job-name=baseline:encode
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
method="${1:?usage: sbatch scripts/model/baseline/encode_states.sbatch <chronicle|hicogen> [task] [recent] [K]}"
task="${2:-product}"
recent="${3:-16}"
interests="${4:-4}"
source scripts/model/common_logging.sh baseline/encode
export LOG_STAGE="model/baseline/encode"
PYTHON_BIN="${DEYIPHI_PYTHON:-python}"
log_dir="${MODEL_LOG_DIR:-log/model/baseline/encode}"
mkdir -p "$log_dir"
exec > "$log_dir/${method}-${task}-k${interests}-r${recent}.console.log" \
     2> "$log_dir/${method}-${task}-k${interests}-r${recent}.errors.log"
exec "$PYTHON_BIN" -u -m src.model.deyi.cli encode \
    --config config/model/deyi.yaml --task "$task" \
    --num-interests "$interests" --recent "$recent" --method "$method"
