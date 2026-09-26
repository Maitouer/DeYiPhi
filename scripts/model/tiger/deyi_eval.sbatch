#!/usr/bin/env bash
# Evaluate a Tiger model conditioned on DeYi states on four H100 GPUs.
# Usage: sbatch scripts/model/tiger/deyi_eval.sbatch short_video recent 4 [arm]
# With ``arm`` both the teacher root (output/model/deyi/hpo/<arm>) and the run directory
# (output/model/tiger/deyi/<task>/k<K>/<arm>/seed2026) follow the HPO-arm layout.
#SBATCH --partition=gpu
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=44
#SBATCH --mem=500000M
#SBATCH --time=3-00:00:00
#SBATCH --job-name=tiger:deyi:eval
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
source scripts/model/common_logging.sh tiger/deyi_eval
task="${1:?usage: sbatch scripts/model/tiger/deyi_eval.sbatch short_video|ad|product full|recent [K]}"
mode="${2:?usage: sbatch scripts/model/tiger/deyi_eval.sbatch short_video|ad|product full|recent [K]}"
interests="${3:?usage: sbatch scripts/model/tiger/deyi_eval.sbatch short_video|ad|product full|recent [K]}"
arm="${4:-}"
case "$task" in short_video|ad|product) ;; *) exit 2 ;; esac
case "$mode" in full|recent) ;; *) exit 2 ;; esac
export LOG_STAGE="model/tiger/deyi_eval"
# ``TIGER_DEYI_ROOT`` conditions on a frozen memory bank other than the production DeYi teacher
# (the controlled baselines of 04_baseline.md live under output/model/compress/<method>).
deyi_root="${TIGER_DEYI_ROOT:-}"
if [ -n "$deyi_root" ]; then
    extra=(--deyi-root "$deyi_root")
elif [ -n "$arm" ]; then
    # ``recent`` keeps the legacy path (every shipped run lives there); any other view gets its
    # own directory, so full and recent never write the same run.
    extra=(--deyi-root "output/model/deyi/hpo/${arm}")
else
    extra=()
fi
PYTHON_BIN="${DEYIPHI_PYTHON:-python}"
source scripts/model/tiger/guard_monitor.sh
start_tiger_gpu_monitor
# The run directory must be resolved for the *same* teacher, otherwise ``--resume`` probes the
# wrong arm: a baseline bank keeps one directory per method.
root_arg=""
[ -n "${deyi_root:-}" ] && root_arg=",deyi_root='${deyi_root}'"
run=$("$PYTHON_BIN" -c "from src.model.tiger import load_config; print(load_config('config/model/tiger.yaml','${task}','${mode}',deyi_k=${interests}${root_arg}).run_dir)")
"$PYTHON_BIN" -u -m torch.distributed.run --standalone --nproc_per_node=${TIGER_NPROC:-4} \
    --module src.model.tiger.evaluate --config config/model/tiger.yaml \
    --task "$task" --mode "$mode" --deyi-k "$interests" ${extra[@]+"${extra[@]}"}
