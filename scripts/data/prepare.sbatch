#!/usr/bin/env bash
# Recommendation data preparation.
#
#   sbatch scripts/data/prepare.sbatch [--resume] [--max-rows N] [--max-shards N]
#
# Logs are created by the job itself (nothing has to be prepared beforehand):
#   <project>/log/data/prepare/<run-id>/{console.log,progress.json,progress.jsonl,
#                                        plan.json,checks.json,errors.log}
#   <project>/log/data/prepare/latest -> <run-id>
# Watch it with:  tail -f log/data/prepare/latest/console.log
#
# Job name: <chain>:<stage>
#SBATCH --job-name=data:prepare
#SBATCH --partition=debug
#SBATCH --qos=debug_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=120000M
#SBATCH --time=1-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -uo pipefail
cd ${DEYIPHI_ROOT:-.}

# Interpreter scratch stays off the shared project tree, and prefers a site-provided disk scratch:
# on some sites the per-job /dev/shm directory is deleted mid-run. Only the script that created the
# directory removes it, so a nested stage can never wipe it for its siblings.
tmp_scratch=""
tmp_scratch_owned=0
user_name="${USER:-user}"
for candidate in "${DEYIPHI_TMPDIR:-}" \
                 "${SLURM_TMPDIR:-}" \
                 "${DEYIPHI_SCRATCH_ROOT:+${DEYIPHI_SCRATCH_ROOT}/${user_name}/slurm-scratch/${SLURM_JOB_ID:-$$}}" \
                 "/dev/shm/deyiphi-${user_name}-${SLURM_JOB_ID:-$$}" \
                 "/tmp/deyiphi-${user_name}-${SLURM_JOB_ID:-$$}" \
                 "/var/tmp/deyiphi-${user_name}-${SLURM_JOB_ID:-$$}"; do
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
else
    echo "[data/prepare] WARNING: no writable TMPDIR" >&2
fi

RUN_ID="$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-local}"
LOG_ROOT="$PWD/log/data/prepare"
LOG_DIR="$LOG_ROOT/$RUN_ID"
if mkdir -p "$LOG_DIR" 2>/dev/null; then
    ln -sfn "$RUN_ID" "$LOG_ROOT/latest"
else
    LOG_DIR="/tmp/data-prepare-${SLURM_JOB_ID:-$$}"
    mkdir -p "$LOG_DIR"
fi
export DATA_LOG_DIR="$LOG_DIR"
exec > "$LOG_DIR/console.log" 2> "$LOG_DIR/errors.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [startup   ] job=${SLURM_JOB_ID:-local} host=$(hostname) log=$LOG_DIR"

if [[ -f /usr/share/modules/init/bash ]]; then source /usr/share/modules/init/bash; fi
if command -v module >/dev/null 2>&1; then
    module purge
    module load cuda/12.8
fi

export PYTHONPATH="$PWD"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
exec python -u -m src.data.prepare "$@"
