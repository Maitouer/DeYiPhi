#!/usr/bin/env bash
# Serial Tiger chain: run one plan row (SFT then eval in the same 4-GPU allocation),
# then submit the next row. Usage: sbatch scripts/model/tiger/chain.sbatch [row]
#SBATCH --partition=gpu
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=44
#SBATCH --mem=500000M
#SBATCH --time=3-00:00:00
#SBATCH --job-name=tiger:chain
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
cd ${DEYIPHI_ROOT:-.}
# Slurm client binaries are not always on the compute node PATH (a self-submitting chain died
# with "sbatch: command not found"), so pin the location explicitly.
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
row="${1:-1}"
# The plan file is a second argument so the chain can carry it across its self-submitted jobs.
PLAN="${2:-scripts/model/tiger/chain_plan.tsv}"
source scripts/model/common_logging.sh tiger/chain
export LOG_STAGE="model/tiger/chain"
PYTHON_BIN=python

# 计算节点的 Slurm 客户端不一定在 PATH 里，裸 `sbatch` 会以 "command not found" 静默掐断链。
# 显式解析客户端，缺了就大声报错而不是悄悄停下。
SBATCH_BIN="$(command -v sbatch 2>/dev/null || true)"
for cand in sbatch /usr/local/bin/sbatch /usr/bin/sbatch; do
    if [ -z "$SBATCH_BIN" ] && [ -x "$cand" ]; then SBATCH_BIN="$cand"; fi
done
if [ -z "$SBATCH_BIN" ]; then
    echo "[chain] no Slurm client (sbatch) found on $(hostname); the next row cannot be submitted" >&2
fi
line="$(sed -n "${row}p" "$PLAN")"
if [ -z "$line" ]; then
    echo "[chain] plan exhausted at row $row"; exit 0
fi
read -r task mode variant <<<"$line"
echo "[chain] row=$row task=$task mode=$mode variant=$variant job=${SLURM_JOB_ID}"
if [ "$variant" = "native" ]; then
    run="output/model/tiger/${task}/${mode}/seed2026"; extra=()
elif [ "$variant" = "phi" ]; then
    # The vocabulary belongs to a frozen teacher (k + HPO arm). PHI_TEACHER_* lets one plan cover
    # several teachers; the tag also names the run directory so two teachers never overwrite.
    teacher_k="${PHI_TEACHER_K:-4}"
    teacher_arm="${PHI_TEACHER_ARM:-}"
    teacher_tag="k${teacher_k}"
    [ -n "$teacher_arm" ] && teacher_tag="${teacher_tag}_${teacher_arm}"
    run="output/model/tiger/phi/${task}/${teacher_tag}/seed2026"
    extra=(--phi --phi-deyi-k "$teacher_k")
    [ -n "$teacher_arm" ] && extra+=(--phi-deyi-arm "$teacher_arm")
else
    k="${variant#k}"; extra=(--deyi-k "$k")
    run="output/model/tiger/deyi/${task}/${mode}/k${k}/seed2026"
fi
args=(--config config/model/tiger.yaml --task "$task" --mode "$mode" "${extra[@]}")
if [ -f "$run/model.pt" ] && [ -f "$run/train_summary.json" ]; then
    echo "[chain] sft already complete: $run"
else
    if [ -f "$run/checkpoint.pt" ]; then args+=(--resume); fi
    "$PYTHON_BIN" -u -m torch.distributed.run --standalone --nproc_per_node=4 \
        --module src.model.tiger.train "${args[@]}"
fi
if [ -f "$run/eval/metrics.json" ]; then
    echo "[chain] eval already complete: $run/eval/metrics.json"
else
    "$PYTHON_BIN" -u -m torch.distributed.run --standalone --nproc_per_node=4 \
        --module src.model.tiger.evaluate "${args[@]}" --run-dir "$run"
fi
next=$((row + 1))
if [ -n "$(sed -n "${next}p" "$PLAN")" ]; then
    [ -n "$SBATCH_BIN" ] || { echo "[chain] cannot submit row $next: no sbatch client" >&2; exit 1; }
    "$SBATCH_BIN" scripts/model/tiger/chain.sbatch "$next" "$PLAN" && echo "[chain] submitted row $next"
else
    echo "[chain] chain complete after row $row"
fi
