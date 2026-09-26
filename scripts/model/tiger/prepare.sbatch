#!/usr/bin/env bash
# Build the shared Tiger codebook used by all three tasks and both history views.
# Usage: sbatch scripts/model/tiger/prepare.sbatch [config]
#   default config = config/model/tiger.yaml -> output/model/item/tiger_codebook
#   variant config = config/model/tiger.text.yaml -> output/model/tiger.text/asset
#SBATCH --partition=gpu
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=11
#SBATCH --mem=125000M
#SBATCH --time=3-00:00:00
#SBATCH --job-name=tiger:prepare
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
cd ${DEYIPHI_ROOT:-.}
CONFIG="${1:-config/model/tiger.yaml}"
STAGE="tiger/prepare"
VARIANT="$(basename "$CONFIG" .yaml)"
if [ "$VARIANT" != "tiger" ]; then
    STAGE="tiger/prepare.${VARIANT#tiger.}"
fi
source scripts/model/common_logging.sh "$STAGE"
export LOG_STAGE="model/$STAGE"
PYTHON_BIN=python
"$PYTHON_BIN" -u -m src.model.tiger.prepare --config "$CONFIG"
