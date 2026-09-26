#!/usr/bin/env bash
# Train Tiger with frozen DeYi interest states on four H100 GPUs.
# Usage: sbatch scripts/model/tiger/deyi_sft.sbatch short_video recent 4 [arm]
# Output: output/model/tiger/single_channel/<task>/deyi_<mode>_r<R>_k<K>/seed2026   (no arm)
#         …/deyi_<mode>_r<R>_k<K>/<arm>/seed2026                                  (arm = reference config)
# With ``arm`` the frozen teacher is output/model/deyi/hpo/<arm>; ``TIGER_DEYI_ROOT`` selects any
# other frozen memory bank, and each root keeps its own run directory.
# The shared native Tiger asset must already exist under output/model/item/tiger_codebook.
#SBATCH --partition=gpu
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=44
#SBATCH --mem=500000M
#SBATCH --time=3-00:00:00
#SBATCH --job-name=tiger:deyi:sft
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
source scripts/model/common_logging.sh tiger/deyi_sft
task="${1:?usage: sbatch scripts/model/tiger/deyi_sft.sbatch short_video|ad|product full|recent [K]}"
mode="${2:?usage: sbatch scripts/model/tiger/deyi_sft.sbatch short_video|ad|product full|recent [K]}"
interests="${3:?usage: sbatch scripts/model/tiger/deyi_sft.sbatch short_video|ad|product full|recent [K]}"
arm="${4:-}"
case "$task" in short_video|ad|product) ;; *) exit 2 ;; esac
case "$mode" in full|recent) ;; *) exit 2 ;; esac
export LOG_STAGE="model/tiger/deyi_sft"
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
# The run directory must be resolved for the *same* teacher, otherwise ``--resume`` probes the
# wrong arm: a baseline bank keeps one directory per method.
root_arg=""
[ -n "${deyi_root:-}" ] && root_arg=",deyi_root='${deyi_root}'"
run=$("$PYTHON_BIN" -c "from src.model.tiger import load_config; print(load_config('config/model/tiger.yaml','${task}','${mode}',deyi_k=${interests}${root_arg}).run_dir)")
args=(--config config/model/tiger.yaml --task "$task" --mode "$mode" --deyi-k "$interests" ${extra[@]+"${extra[@]}"})
if [[ -f "$run/checkpoint.pt" || -f "$run/train_summary.json" ]]; then
    args+=(--resume)
fi
source scripts/model/tiger/guard_monitor.sh
start_tiger_gpu_monitor
"$PYTHON_BIN" -u -m torch.distributed.run --standalone --nproc_per_node=${TIGER_NPROC:-4} \
    --log-dir "log/model/tiger/deyi_sft/torchrun-$SLURM_JOB_ID" --redirects 3 --tee 3 \
    --module src.model.tiger.train "${args[@]}"
