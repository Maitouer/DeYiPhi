#!/usr/bin/env bash
# Train Tiger on the frozen phi anchors (interest-augmented items) with four H100 GPUs.
#
#   sbatch scripts/model/tiger/phi_sft.sbatch <task> <mode> <teacher k> [teacher arm]
#   e.g. sbatch scripts/model/tiger/phi_sft.sbatch short_video recent 16 sharp
#
# Aligned with the native (full/recent) arm by construction: no DeYi state tokens are injected
# (``--phi-deyi-k/--phi-deyi-arm`` select the *artifacts* only), the history block is the native
# ``<c0..c3><sep>`` plus the item's own region token, and the target chain is ``⟨g_a c0 c1 c2 c3⟩``
# (v6: one anchor position, five chain positions, four scored digits).
# Output: output/model/tiger/phi/<task>/<teacher tag>/seed2026
#SBATCH --partition=gpu
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=44
#SBATCH --mem=500000M
#SBATCH --time=06:00:00
#SBATCH --job-name=tiger:phi:sft
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
source scripts/model/common_logging.sh tiger/phi_sft
task="${1:?usage: sbatch scripts/model/tiger/phi_sft.sbatch <task> <mode> <k> [arm]}"
mode="${2:?usage: sbatch scripts/model/tiger/phi_sft.sbatch <task> <mode> <k> [arm]}"
teacher_k="${3:?usage: sbatch scripts/model/tiger/phi_sft.sbatch <task> <mode> <k> [arm]}"
teacher_arm="${4:-}"
case "$task" in short_video|ad|product) ;; *) exit 2 ;; esac
case "$mode" in full|recent) ;; *) exit 2 ;; esac
export LOG_STAGE="model/tiger/phi_sft"
PYTHON_BIN="${DEYIPHI_PYTHON:-python}"
# The frozen teacher's old/recent cut: the phi chain's own override wins, otherwise the config's
# per-task value (32 for short_video, 16 for product). Never a bare literal -- this value names
# both the vocabulary this arm consumes and the arm's own directory.
if [ -n "${PHI_DEYI_RECENT:-}" ]; then
    recent="$PHI_DEYI_RECENT"
else
    recent=$("$PYTHON_BIN" -c "from omegaconf import OmegaConf; print(OmegaConf.load('config/model/tiger.yaml').recent_history['$task'])")
fi
export PHI_DEYI_K="$teacher_k"
export PHI_DEYI_RECENT="$recent"
[ -n "$teacher_arm" ] && export PHI_DEYI_ARM="$teacher_arm"
# One naming rule for the run directory (``tiger/__init__.py``): derived from the same
# (dataset, task, mode, K, teacher window) that names the vocabulary path above.
run=$("$PYTHON_BIN" -c "from src.model.tiger import load_config; print(load_config('config/model/tiger.yaml','${task}','${mode}',phi=True,phi_deyi_k=${teacher_k}).run_dir)")
args=(--config config/model/tiger.yaml --task "$task" --mode "$mode" --phi
      --phi-deyi-k "$teacher_k" --run-dir "$run")
[ -n "$teacher_arm" ] && args+=(--phi-deyi-arm "$teacher_arm")
if [[ -f "$run/checkpoint.pt" || -f "$run/train_summary.json" ]]; then
    args+=(--resume)
fi
source scripts/model/tiger/guard_monitor.sh
start_tiger_gpu_monitor
"$PYTHON_BIN" -u -m torch.distributed.run --standalone --nproc_per_node=${TIGER_NPROC:-4} \
    --log-dir "log/model/tiger/phi_log/$SLURM_JOB_ID" --redirects 3 --tee 3 \
    --module src.model.tiger.train "${args[@]}"
