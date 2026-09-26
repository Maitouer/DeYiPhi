#!/usr/bin/env bash
# Phi chain: fit the shared predictive vocabulary and the target routes of one frozen DeYi arm.
#
#   sbatch scripts/model/phi/chain.sbatch <task> [k] [recent] [extra CLI flags...]
#   sbatch scripts/model/phi/chain.sbatch product 4 16 --vocabulary 128
#
# The equivalent CLI, if you prefer one stage at a time (see also ``stage.sbatch``):
#
#   PHI_DEYI_K=4 PHI_DEYI_RECENT=16 python -m src.model.phi.cli <stage> --task product
#
# Output: output/model/phi/<dataset>/<task>/k<k>_r<recent>/  (see src/model/phi/artifacts.py).
# Job name: phi:chain:<task>:k<k>:r<recent>
#
# Partition note: the small shared partition is the right home for a one-off asset build; there is
# no job-time guard to stay under. The catalog is 28 GB of fp16 embeddings, so it does not fit
# next to the score blocks there and the estimator falls back to its streaming path; that path is
# correct, just more IO (the config comment on ``estimator.resident`` states the trade-off).
#SBATCH --partition=debug
#SBATCH --qos=debug_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120000M
#SBATCH --time=1-00:00:00
#SBATCH --job-name=phi:chain
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"
source scripts/model/common_logging.sh phi/chain
task="${1:?usage: sbatch scripts/model/phi/chain.sbatch <task> [k] [recent] [flags]}"
k="${2:-4}"
recent="${3:-16}"
shift 3 2>/dev/null || shift $#
case "$task" in short_video|ad|product) ;; *) echo "unknown task $task" >&2; exit 2 ;; esac
if [ "${SLURM_JOB_NAME:-}" = "phi:chain" ]; then
    scontrol update JobId="${SLURM_JOB_ID}" \
        JobName="phi:chain:${task}:k${k}:r${recent}" 2>/dev/null || true
fi
# The frozen teacher this vocabulary belongs to: the same override names the DeYi chain uses for
# its own paths, so a phi run cannot read one arm and record another (config.py applies them).
export PHI_DEYI_K="$k" PHI_DEYI_RECENT="$recent"
export LOG_STAGE="model/phi/chain"
python_bin="${DEYIPHI_PYTHON:-python}"
for stage in plan reservoir estimate codebook router; do
    echo "[$(date '+%H:%M:%S')] phi ${stage}" >&2
    "$python_bin" -u -m src.model.phi.cli "$stage" \
        --config "${PHI_CONFIG:-config/model/phi.yaml}" --task "$task" "$@"
done
for split in train test; do
    echo "[$(date '+%H:%M:%S')] phi tokenize ${split}" >&2
    "$python_bin" -u -m src.model.phi.cli tokenize \
        --config "${PHI_CONFIG:-config/model/phi.yaml}" --task "$task" --split "$split" "$@"
done
echo "[$(date '+%H:%M:%S')] phi route" >&2
"$python_bin" -u -m src.model.phi.cli route \
    --config "${PHI_CONFIG:-config/model/phi.yaml}" --task "$task" --split train "$@"
echo "[$(date '+%H:%M:%S')] phi audit" >&2
"$python_bin" -u -m src.model.phi.cli audit \
    --config "${PHI_CONFIG:-config/model/phi.yaml}" --task "$task" --split train "$@"
