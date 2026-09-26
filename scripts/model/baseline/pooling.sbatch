#!/usr/bin/env bash
# Pooling memory states: one continuous summary of H_old (06_analysis_baseline.md §2).
#
#   sbatch scripts/model/baseline/pooling.sbatch [task] [recent] [K]
#
# Deterministic and training-free (a plain mean over the unit item vectors), so it is a data-side
# stage: one GPU on the gpu partition, no training allocation.
# Job name: baseline:pooling
#SBATCH --partition=gpu
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120000M
#SBATCH --time=2:00:00
#SBATCH --job-name=baseline:pooling
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
task="${1:-product}"
recent="${2:-16}"
interests="${3:-1}"
source scripts/model/common_logging.sh baseline/pooling
export LOG_STAGE="model/baseline/pooling"
"${DEYIPHI_PYTHON:-python}" -u -m src.model.baseline.pooling \
    --task "$task" --recent "$recent" --interests "$interests"
