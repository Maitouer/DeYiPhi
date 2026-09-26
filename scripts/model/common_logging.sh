#!/usr/bin/env bash
# Create this job's log directory at startup and redirect stdout/stderr into it.
#
#   source scripts/model/common_logging.sh <stage>
#
# Layout (mirrors scripts/model/<stage>.sbatch):
#   <project>/log/model/<stage>/<run-id>/{console.log,errors.log}
#   <project>/log/model/<stage>/latest -> <run-id>
# Callers point their #SBATCH --output/--error at /dev/null, so a missing directory can
# never make the job fail to start; the job creates it itself.
stage="${1:?usage: source scripts/model/common_logging.sh <stage>}"
# The run id carries the process id as well: several stage scripts can start within the same second
# inside one job (eight parallel prepares), and without the pid they would share one log directory
# and race on progress.json.
run_id="$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-local}-$$"
root="$PWD/log/model/$stage"
dir="$root/$run_id"
if mkdir -p "$dir" 2>/dev/null; then
    ln -sfn "$run_id" "$root/latest"
else
    dir="/tmp/model-log-$(echo "$stage" | tr / _)-${SLURM_JOB_ID:-$$}"
    mkdir -p "$dir"
fi
export MODEL_LOG_DIR="$dir"
exec > "$dir/console.log" 2> "$dir/errors.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [startup   ] job=${SLURM_JOB_ID:-local} stage=$stage host=$(hostname) log=$dir"

# Keep interpreter scratch out of the shared project tree. ``tempfile`` honours TMPDIR, but when
# every candidate is unusable it silently falls back to the *current working directory*, which is
# why torchrun / DataLoader workers occasionally dropped ``torchelastic_*`` / ``pymp-*`` into the
# project root (a full node /tmp is enough to trigger it).
#
# Scratch policy, in order:
#   1. ``DEYIPHI_TMPDIR`` - an outer job pins the scratch so every nested stage shares it;
#   2. ``SLURM_TMPDIR``   - Slurm's own per-job scratch when the site provides one;
#   3. ``DEYIPHI_SCRATCH_ROOT/<user>/slurm-scratch/<jobid>`` - when the site offers a shared
#      disk scratch root and exports it. Disk beats the node-local options on sites whose per-job
#      /dev/shm is deleted mid-run, which kills torch's lazy-import TemporaryDirectory and every
#      parallel prepare; the write volume is small either way;
#   4. /dev/shm, /tmp, /var/tmp - faster, used when nothing above is writable.
# Whatever candidate wins, only the script that *created* the directory removes it on exit, so a
# nested stage can never wipe the scratch out from under its siblings.
tmp_scratch=""
tmp_scratch_owned=0
user_name="${USER:-user}"
# ``DEYIPHI_TMPDIR`` is a *shared* directory by design (nested stages of one job reuse it), but two
# concurrent jobs on the same node must not share it: the job that created the directory removes it
# on exit, which yanks the scratch out from under the other one (torch's lazy-import
# ``TemporaryDirectory`` then dies with FileNotFoundError). Suffix it with the job id so every job
# owns its own copy while its stages still share it.
deyiphi_deyiphi_scratch="${DEYIPHI_TMPDIR:-}"
if [ -n "$deyiphi_deyiphi_scratch" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    case "$deyiphi_deyiphi_scratch" in
        *-"$SLURM_JOB_ID") ;;
        *) deyiphi_deyiphi_scratch="${deyiphi_deyiphi_scratch%/}-${SLURM_JOB_ID}" ;;
    esac
fi
for candidate in "${deyiphi_deyiphi_scratch}" \
                 "${SLURM_TMPDIR:-}" \
                 "${DEYIPHI_SCRATCH_ROOT:+${DEYIPHI_SCRATCH_ROOT}/${user_name}/slurm-scratch/${SLURM_JOB_ID:-$$}}" \
                 "/dev/shm/deyiphi-${user_name}-${SLURM_JOB_ID:-$$}" \
                 "/tmp/deyiphi-${user_name}-${SLURM_JOB_ID:-$$}" \
                 "/var/tmp/deyiphi-${user_name}-${SLURM_JOB_ID:-$$}"; do
    [ -z "$candidate" ] && continue
    # Nested stage scripts share one scratch directory (an outer job exports DEYIPHI_TMPDIR and its
    # inner scripts resolve to the same path). Only the script that actually created the directory
    # may delete it, otherwise the first inner script to exit would wipe the scratch out from under
    # its siblings - which is what silently killed parallel prepares and arms before.
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
    echo "[common_logging] WARNING: no writable TMPDIR; interpreter scratch may land in the cwd" >&2
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [startup   ] TMPDIR=${TMPDIR:-<unset>} (interpreter scratch)"

# ---- disk preflight ---------------------------------------------------------------------------
# A full project volume does not fail the training loudly: the vendored writers abort inside a C
# extension (parquet/checkpoint writers raise ENOSPC where Python cannot catch it), which shows up
# as a mysterious dump with no traceback and costs hours of debugging. Refuse to start instead.
avail_kb=$(df -Pk "${DEYIPHI_ROOT:-.}" 2>/dev/null | awk 'NR==2 {print $4}')
if [ -n "${avail_kb:-}" ] && [ "$avail_kb" -lt 30000000 ]; then
    echo "[startup   ] FATAL: only $((avail_kb / 1048576)) GiB free on the project volume;" \
         "need >= 30 GiB. Free space before rerunning (see the arm cleanup in scripts/)." >&2
    exit 3
fi

# ---- libcuda development symlink ------------------------------------------------------------
# The compute nodes ship the 64-bit driver only as ``libcuda.so.1`` (no ``libcuda.so``) while a
# 32-bit ``/lib/i386-linux-gnu/libcuda.so`` exists, so ``gcc ... -lcuda`` (Triton's JIT for
# vLLM) fails with "cannot find -lcuda" and the whole evaluation dies during engine startup.
# Triton takes ``LIBRARY_PATH`` after ``-L``, so a per-job symlink directory fixes it. Doing it
# here - in the job's own environment, before any child is spawned - also covers processes that
# do not inherit an env dict built later (Ray workers).
if [ ! -e /lib/x86_64-linux-gnu/libcuda.so ] && [ -e /lib/x86_64-linux-gnu/libcuda.so.1 ]; then
    cuda_link="${TMPDIR:-/tmp}/cuda-link"
    mkdir -p "$cuda_link" 2>/dev/null || cuda_link=""
    if [ -n "$cuda_link" ]; then
        [ -e "$cuda_link/libcuda.so" ] || ln -sfn /lib/x86_64-linux-gnu/libcuda.so.1 "$cuda_link/libcuda.so" 2>/dev/null || true
        export LIBRARY_PATH="$cuda_link${LIBRARY_PATH:+:$LIBRARY_PATH}"
        export LD_LIBRARY_PATH="$cuda_link${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [startup   ] libcuda shim=$cuda_link/libcuda.so"
    fi
fi
