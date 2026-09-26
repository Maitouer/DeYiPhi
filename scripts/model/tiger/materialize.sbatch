#!/usr/bin/env bash
# Materialise the Tiger training tensors once: sbatch scripts/model/tiger/materialize.sbatch <task> <mode>
#SBATCH --partition=gpu
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=11
#SBATCH --mem=125000M
#SBATCH --time=00:40:00
#SBATCH --job-name=tiger:materialize
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
task="${1:?usage: sbatch scripts/model/tiger/materialize.sbatch short_video|ad|product full|recent}"
mode="${2:?usage: sbatch scripts/model/tiger/materialize.sbatch short_video|ad|product full|recent}"
source scripts/model/common_logging.sh tiger/materialize
export LOG_STAGE="model/tiger/materialize"
PYTHON_BIN="${DEYIPHI_PYTHON:-python}"
"$PYTHON_BIN" -u -m src.model.tiger.materialize --config config/model/tiger.yaml \
    --task "$task" --mode "$mode"
