#!/usr/bin/env bash
# CAUSE-Core memory states (docs/v1_algorithm/04_baseline.md §4).
#
#   sbatch scripts/model/baseline/cause.sbatch [task] [recent] [K]
#
# Deterministic and training-free: the four most recently active level-1 TIGER categories of the
# old history are mean-pooled into four memory states. It is a data-side stage (no gradients), so
# it runs on the debug partition next to the other asset builds and never touches the four-GPU
# training allocation.
# Job name: baseline:cause
#SBATCH --partition=debug
#SBATCH --qos=debug_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120000M
#SBATCH --time=6:00:00
#SBATCH --job-name=baseline:cause
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
task="${1:-product}"
recent="${2:-16}"
interests="${3:-4}"
source scripts/model/common_logging.sh baseline/cause
export LOG_STAGE="model/baseline/cause"
"${DEYIPHI_PYTHON:-python}" -u -m src.model.baseline.cause \
    --task "$task" --recent "$recent" --interests "$interests"
