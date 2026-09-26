#!/usr/bin/env bash
# Submit the three Tiger product arms, each as sft -> eval on four H100 GPUs.
#
#   bash scripts/model/tiger/arms_product.sh [--dry-run]
#
# Arms (all on the ``product`` task):
#   full          native model, whole history                     -> output/model/tiger/product/full
#   recent16      native model, last ``recent_history`` (16)      -> output/model/tiger/product/recent
#   deyi-k4-r16   native model + 4 DeYi interest states, recent   -> output/model/tiger/deyi/product/recent/k4
#
# The DeYi arm reads states from output/model/deyi/<dataset>/<task>/k<K>_r<R>, where
# <dataset> = cfg.dataset and <task> = product. Both the model's recent window and the
# r<R> of the arm come from the single ``recent_history`` setting.
set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"

DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="echo"

submit() {
    local label="$1" sft_script="$2" eval_script="$3"
    shift 3
    if [ -n "$DRY" ]; then
        echo "would submit: $label  sft=$sft_script $*   eval=$eval_script $*"
        return
    fi
    local sft_job eval_job
    sft_job=$(sbatch --parsable --job-name="tiger:${label}:sft" "$sft_script" "$@")
    eval_job=$(sbatch --parsable --dependency=afterok:"$sft_job" \
        --job-name="tiger:${label}:eval" "$eval_script" "$@")
    printf '%-14s sft=%-8s eval=%-8s (eval waits for sft)\n' "$label" "$sft_job" "$eval_job"
}

submit full          scripts/model/tiger/sft.sbatch     scripts/model/tiger/eval.sbatch     product full
submit recent16      scripts/model/tiger/sft.sbatch     scripts/model/tiger/eval.sbatch     product recent
submit deyi-k4-r16   scripts/model/tiger/deyi_sft.sbatch scripts/model/tiger/deyi_eval.sbatch product recent 4

echo
echo "监控: squeue -u \$USER -o '%.8i %.34j %.8T %.10M'"
echo "日志: log/model/tiger/{,deyi_}{sft,eval}/  (由 common_logging.sh 建 run 目录)"
