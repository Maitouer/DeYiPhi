#!/usr/bin/env bash
# Run one Tiger full or recent baseline on four complete H100 GPUs.
# Usage: sbatch scripts/model/tiger/sft.sbatch short_video full
# Tasks: short_video, ad, product. Modes: full, recent.
#SBATCH --partition=gpu
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=44
#SBATCH --mem=500000M
#SBATCH --time=3-00:00:00
#SBATCH --job-name=tiger:sft
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
source scripts/model/common_logging.sh tiger/sft
task="${1:?usage: sbatch scripts/model/tiger/sft.sbatch short_video|ad|product full|recent}"
mode="${2:?usage: sbatch scripts/model/tiger/sft.sbatch short_video|ad|product full|recent}"
case "$task" in short_video|ad|product) ;; *) exit 2 ;; esac
case "$mode" in full|recent) ;; *) exit 2 ;; esac
export LOG_STAGE="model/tiger/sft"
PYTHON_BIN="${DEYIPHI_PYTHON:-python}"
NPROC="${TIGER_NPROC:-4}"
run=$("$PYTHON_BIN" -c "from src.model.tiger import load_config; print(load_config('config/model/tiger.yaml','${task}','${mode}').run_dir)")
args=(--config config/model/tiger.yaml --task "$task" --mode "$mode")
if [[ -f "$run/checkpoint.pt" || -f "$run/train_summary.json" ]]; then
    args+=(--resume)
fi
source scripts/model/tiger/guard_monitor.sh
start_tiger_gpu_monitor
"$PYTHON_BIN" -u -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    --module src.model.tiger.train "${args[@]}"
