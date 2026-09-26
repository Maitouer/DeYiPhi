#!/usr/bin/env bash
# Evaluate the predictive-vocabulary Tiger model on four H100 GPUs.
#
#   sbatch scripts/model/tiger/phi_eval.sbatch <task> <mode> <teacher k> [recent]
#
# The chain is generated as <g* c0 c1 c2 c3> -- the route first (restricted to the user's own
# tokens, docs/v1_algorithm/02_phi.md §10.1), then the four catalog-constrained code digits -- and
# only the code digits reach the metrics, so pass@k stays comparable with the native arms.
#
# The retired soft-image row is gone with the static item-region prior it reordered beams with
# (02_phi.md §16): a route is a per-target label, not an item property, so there is nothing to
# apply a p(g) prior to.
# Output: output/model/tiger/phi/<task>/k<K>_r<R>/seed2026/eval/metrics.json
#SBATCH --partition=gpu
#SBATCH --qos=gpu_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --cpus-per-task=44
#SBATCH --mem=500000M
#SBATCH --time=02:00:00
#SBATCH --job-name=tiger:phi:eval
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
source scripts/model/common_logging.sh tiger/phi_eval
task="${1:?usage: sbatch scripts/model/tiger/phi_eval.sbatch <task> <mode> <k> [recent]}"
mode="${2:?usage: sbatch scripts/model/tiger/phi_eval.sbatch <task> <mode> <k> [recent]}"
teacher_k="${3:?usage: sbatch scripts/model/tiger/phi_eval.sbatch <task> <mode> <k> [recent]}"
case "$task" in short_video|ad|product) ;; *) echo "unknown task $task" >&2; exit 2 ;; esac
case "$mode" in full|recent) ;; *) echo "unknown mode $mode" >&2; exit 2 ;; esac
export LOG_STAGE="model/tiger/phi_eval"
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
if [ "${SLURM_JOB_NAME:-}" = "tiger:phi:eval" ]; then
    scontrol update JobId="${SLURM_JOB_ID}" JobName="tiger:phi:eval:${task}" 2>/dev/null || true
fi
run=$("$PYTHON_BIN" -c "from src.model.tiger import load_config; print(load_config('config/model/tiger.yaml','${task}','${mode}',phi=True,phi_deyi_k=${teacher_k}).run_dir)")
if [ ! -f "$run/model.pt" ]; then
    echo "[phi_eval] no trained weights at $run/model.pt (run phi_sft.sbatch first)" >&2
    exit 2
fi
args=(--config config/model/tiger.yaml --task "$task" --mode "$mode" --phi
      --phi-deyi-k "$teacher_k" --run-dir "$run")
source scripts/model/tiger/guard_monitor.sh
start_tiger_gpu_monitor
exec "$PYTHON_BIN" -u -m torch.distributed.run --standalone \
    --nproc_per_node="${TIGER_NPROC:-4}" \
    --log-dir "log/model/tiger/phi_log/$SLURM_JOB_ID" --redirects 3 --tee 3 \
    --module src.model.tiger.evaluate "${args[@]}"
