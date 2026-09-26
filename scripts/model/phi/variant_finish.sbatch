#!/usr/bin/env bash
# Finish an alternative quantization vocabulary: tokenize the held-out split.
#
#   sbatch scripts/model/phi/variant_finish.sbatch [task] [k] [recent]
#
# ``variant.sbatch`` fits the codebook/router and tokenizes the supervised split only. The
# downstream TIGER arm also needs the test split's encoder tokens (routes are supervision and stay
# train-only), so this stage adds them for every fitted metric. Idempotent.
# Job name: phi:variant:finish
#SBATCH --partition=gpu
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120000M
#SBATCH --time=1:00:00
#SBATCH --job-name=phi:variant:finish
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
task="${1:-product}"
k="${2:-1}"
recent="${3:-16}"
source scripts/model/common_logging.sh phi/variant_finish
export LOG_STAGE="model/phi/variant_finish"
export PHI_DEYI_K="$k" PHI_DEYI_RECENT="$recent"
PYTHON_BIN="${DEYIPHI_PYTHON:-python}"
for metric in euclidean cosine; do
    out="output/model/phi_variants/${metric}/single_channel/${task}/k${k}_r${recent}"
    [ -f "$out/codebook.pt" ] || { echo "[finish] no ${metric} vocabulary; skipped" >&2; continue; }
    if [ -f "$out/tokens/test.pt" ]; then
        echo "[finish] ${metric}: test tokens already present"
        continue
    fi
    export PHI_OUTPUT_DIR="output/model/phi_variants/${metric}"
    echo "[finish] ${metric}: tokenize test"
    "$PYTHON_BIN" -u -m src.model.phi.cli tokenize --config config/model/phi.yaml \
        --task "$task" --metric "$metric" --split test
done
echo "[finish] done"
