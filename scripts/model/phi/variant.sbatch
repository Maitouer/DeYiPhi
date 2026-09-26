#!/usr/bin/env bash
# One alternative quantization of the *same* frozen DeYi states (05_analysis_design.md §11-12).
#
#   sbatch scripts/model/phi/variant.sbatch <euclidean|cosine> [task] [k] [recent]
#
# The reservoir, sufficient statistics and calibration of the predictive run are reused unchanged
# -- "same state reservoir" is a fairness requirement of §12, not an optimisation -- and only the
# codebook is refitted under a different distortion. The router, per-slot tokens, route targets and
# the audit then follow from that partition, so the whole vocabulary lives under
# output/model/phi_variants/<metric>/... and the downstream arm keeps its own run directory.
# Job name: phi:variant
#SBATCH --partition=gpu
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120000M
#SBATCH --time=2:00:00
#SBATCH --job-name=phi:variant
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
metric="${1:?usage: sbatch scripts/model/phi/variant.sbatch <euclidean|cosine> [task] [k] [recent]}"
task="${2:-product}"
k="${3:-4}"
recent="${4:-16}"
case "$metric" in euclidean|cosine) ;; *) echo "unknown metric $metric" >&2; exit 2 ;; esac
source scripts/model/common_logging.sh "phi/variant_${metric}"
export LOG_STAGE="model/phi/variant_${metric}"
export PHI_DEYI_K="$k" PHI_DEYI_RECENT="$recent"
export PHI_OUTPUT_DIR="output/model/phi_variants/${metric}"
base="output/model/phi/single_channel/${task}/k${k}_r${recent}"
out="${PHI_OUTPUT_DIR}/single_channel/${task}/k${k}_r${recent}"
mkdir -p "$out"
for f in reservoir.pt stats.pt calibration.pt; do
    [ -f "$out/$f" ] || cp -a "$base/$f" "$out/$f"
done
PYTHON_BIN="${DEYIPHI_PYTHON:-python}"
for stage in codebook router; do
    echo "[variant:${metric}] stage=$stage"
    "$PYTHON_BIN" -u -m src.model.phi.cli "$stage" --config config/model/phi.yaml \
        --task "$task" --metric "$metric"
done
# Encoder tokens are needed for *both* splits: the supervised split is what the arm trains on and
# the test split is what it evaluates. Routes stay train-only (they are decoder supervision).
for split in train test; do
    echo "[variant:${metric}] stage=tokenize split=$split"
    "$PYTHON_BIN" -u -m src.model.phi.cli tokenize --config config/model/phi.yaml \
        --task "$task" --metric "$metric" --split "$split"
done
for stage in route audit; do
    echo "[variant:${metric}] stage=$stage"
    "$PYTHON_BIN" -u -m src.model.phi.cli "$stage" --config config/model/phi.yaml \
        --task "$task" --metric "$metric"
done
echo "[variant:${metric}] done"
