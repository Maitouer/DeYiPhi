#!/usr/bin/env bash
# Held-out predictive equivalence of one vocabulary (05_analysis_design.md §19-22).
#
#   sbatch scripts/model/analysis/equivalence.sbatch [task] [k] [recent] [states]
#
# Run it once per quantization; the JSON lands next to the vocabulary it scored.
# Job name: analysis:IV
#SBATCH --partition=gpu
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120000M
#SBATCH --time=2:00:00
#SBATCH --job-name=analysis:IV
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
task="${1:-product}"; k="${2:-4}"; recent="${3:-16}"; states="${4:-1024}"
source scripts/model/common_logging.sh analysis/equivalence
export LOG_STAGE="model/analysis/equivalence"
export PHI_DEYI_K="$k" PHI_DEYI_RECENT="$recent"
"${DEYIPHI_PYTHON:-python}" -u -m src.model.phi.equivalence \
    --config config/model/phi.yaml --task "$task" --k "$k" --recent "$recent" --states "$states"
