#!/usr/bin/env bash
# Build the shared DeYi item semantic space (00-Reconstruct/00_item_semantic_space.md).
#
#   sbatch scripts/model/deyi/prepare.sbatch [--pca-dim D] [--pca-samples N]
#
# Anything after the script name is forwarded to the CLI, so PCA settings are tunable
# without editing the config.
# Job name: <chain>:<stage>
#SBATCH --partition=debug
#SBATCH --qos=debug_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120000M
#SBATCH --time=1-00:00:00
#SBATCH --job-name=deyi:item-space
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
set -uo pipefail
cd ${DEYIPHI_ROOT:-.}

# Interpreter scratch must not land in the shared project tree. A site-provided disk scratch goes
# first: some nodes delete their per-job /dev/shm directory mid-run.
tmp_scratch=""
tmp_scratch_owned=0
user_name="${USER:-user}"
for candidate in "${DEYIPHI_TMPDIR:-}" "${SLURM_TMPDIR:-}" \
                 "${DEYIPHI_SCRATCH_ROOT:+${DEYIPHI_SCRATCH_ROOT}/${user_name}/slurm-scratch/${SLURM_JOB_ID:-$$}}" \
                 "/dev/shm/deyiphi-${user_name}-${SLURM_JOB_ID:-$$}" \
                 "/tmp/deyiphi-${user_name}-${SLURM_JOB_ID:-$$}"; do
    [ -z "$candidate" ] && continue
    [ -d "$candidate" ] || tmp_scratch_owned=1
    if mkdir -p "$candidate" 2>/dev/null && touch "$candidate/.write-probe" 2>/dev/null; then
        rm -f "$candidate/.write-probe" 2>/dev/null || true
        tmp_scratch="$candidate"
        break
    fi
    tmp_scratch_owned=0
done
if [ -n "$tmp_scratch" ]; then
    export TMPDIR="$tmp_scratch"
    # Only a scratch *this* script chose is removed on exit. A caller-pinned scratch
    # (``DEYIPHI_TMPDIR`` is exported by the outer job and deliberately shared with every nested
    # stage) must never be deleted by an inner stage: if the inner stage momentarily does not see
    # it, it creates it, claims ownership, and then removes it on exit, yanking the directory out
    # from under a sibling that is still running. That failure is invisible at the shell level and
    # surfaces much later as ``FileNotFoundError: <tmpdir>/tmp_xxxx`` from torch's lazy
    # ``TemporaryDirectory`` (observed in earlier campaigns).
    case "$TMPDIR" in
        */deyiphi-*|*/slurm-scratch/*)
            if [ "$tmp_scratch_owned" = "1" ] && [ -z "${DEYIPHI_TMPDIR:-}" ]; then
                trap 'rm -rf "$TMPDIR" 2>/dev/null || true' EXIT
            fi ;;
    esac
fi

RUN_ID="$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-local}"
LOG_ROOT="$PWD/log/model/deyi/fit-pca"
LOG_DIR="$LOG_ROOT/$RUN_ID"
mkdir -p "$LOG_DIR" && ln -sfn "$RUN_ID" "$LOG_ROOT/latest" || true
export DEYI_LOG_DIR="$LOG_DIR"
exec > "$LOG_DIR/console.log" 2> "$LOG_DIR/errors.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [startup] job=${SLURM_JOB_ID:-local} host=$(hostname) log=$LOG_DIR"

if [[ -f /usr/share/modules/init/bash ]]; then source /usr/share/modules/init/bash; fi
if command -v module >/dev/null 2>&1; then
    module purge; module load cuda/12.8
fi
export PYTHONPATH="$PWD"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
exec python -u -m src.model.deyi.cli fit-pca \
  --config "${DEYI_CONFIG:-config/model/deyi.yaml}" "$@"
