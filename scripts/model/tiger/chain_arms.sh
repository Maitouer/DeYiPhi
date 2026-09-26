#!/usr/bin/env bash
# Wave driver for the Tiger arms of one task.
#
#   bash scripts/model/tiger/chain_arms.sh [<task>] [--k K] [--dry-run] [--retry] [--limit N]
#     e.g. bash scripts/model/tiger/chain_arms.sh short_video
#
# Stages, each ``--dependency=afterok`` on the one before it:
#
#   materialize_full  -> sft_full  -> eval_full
#   materialize_recent-> sft_recent-> eval_recent
#   sft_deyi  -> eval_deyi
#   sft_phi   -> eval_phi
#
# Arms (single_channel, ``recent`` = recent_history[task] = 32 for short_video / 16 for product):
#
#   full_r<R>      native model, whole history
#   recent_r<R>    native model, last <R> items
#   deyi_r<R>_k<K> + K continuous interest states (frozen DeYi teacher)
#   phi_r<R>_k<K>  + K shared predictive tokens / route (frozen phi vocabulary)
#
# Why a driver: the account holds only four submitted jobs and every sft/eval stage wants the same
# four-GPU shape, so only one stage runs at a time. The driver reads its own state, submits the
# first missing stages while submission slots are free (each with an ``afterok`` dependency on its
# predecessor), and stops. Call it again later and it advances; a stage that failed cancels the
# stages queued behind it and needs an explicit ``--retry`` once the cause is fixed.
#
# State: log/model/tiger/chain/<dataset>_<task>.tsv   (stage <TAB> jobid)
set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"

dataset=single_channel
task=short_video
k=4
dry=0
retry=0
limit=3          # keep one submission slot free for other chains running on the same account
while [ $# -gt 0 ]; do
    case "$1" in
        --k) k="$2"; shift 2 ;;
        --dry-run) dry=1; shift ;;
        --retry) retry=1; shift ;;
        --limit) limit="$2"; shift 2 ;;
        -*) echo "unknown argument: $1" >&2; exit 2 ;;
        *) task="$1"; shift ;;
    esac
done
case "$task" in short_video|ad|product) ;; *) echo "unknown task $task" >&2; exit 2 ;; esac

state_dir="log/model/tiger/chain"
state="$state_dir/${dataset}_${task}.tsv"
mkdir -p "$state_dir"
touch "$state"

stages=(materialize_full materialize_recent sft_full eval_full sft_recent eval_recent
        sft_deyi eval_deyi sft_phi eval_phi)
scripts=(scripts/model/tiger/materialize.sbatch scripts/model/tiger/materialize.sbatch
         scripts/model/tiger/sft.sbatch         scripts/model/tiger/eval.sbatch
         scripts/model/tiger/sft.sbatch         scripts/model/tiger/eval.sbatch
         scripts/model/tiger/deyi_sft.sbatch    scripts/model/tiger/deyi_eval.sbatch
         scripts/model/tiger/phi_sft.sbatch     scripts/model/tiger/phi_eval.sbatch)
args=("$task full" "$task recent"
      "$task full" "$task full" "$task recent" "$task recent"
      "$task recent $k" "$task recent $k" "$task recent $k" "$task recent $k")
# The r<R> label is read from the config instead of being written as a literal: product is r16
# while short_video is r32, and the run directory the stage scripts compute already follows the
# config. A hardcoded label would make ``squeue`` disagree with the directory a job writes into.
recent=$("${DEYIPHI_PYTHON:-python}" -c \
    "import sys; sys.path.insert(0, '.'); from src.model.tiger import load_config; \
print(int(load_config('config/model/tiger.yaml', '${task}', 'recent').recent_history['${task}']))")
names=(tiger:materialize:${task}:full tiger:materialize:${task}:recent
       tiger:sft:${task}:full_r${recent} tiger:eval:${task}:full_r${recent}
       tiger:sft:${task}:recent_r${recent} tiger:eval:${task}:recent_r${recent}
       tiger:sft:${task}:deyi_r${recent}_k${k} tiger:eval:${task}:deyi_r${recent}_k${k}
       tiger:sft:${task}:phi_r${recent}_k${k} tiger:eval:${task}:phi_r${recent}_k${k})
# Where each stage runs: the two one-off materialisations are asset builds (1 GPU, no training), so
# they take the debug partition -- exactly like the phi chain -- and never compete with the four-GPU
# training stages for the shared gpu_normal allocation.
qs=(debug debug gpu gpu gpu gpu gpu gpu gpu gpu)
DEBUG_EXTRA=(--partition=debug --qos=debug_normal --gres=gpu:1 --cpus-per-task=10 --mem=120000M)

job_of() { awk -F'\t' -v stage="$1" '$1 == stage {print $2}' "$state" | tail -1; }
job_state() { sacct -j "$1" --format=State -n -X 2>/dev/null | head -1 | tr -d ' '; }

gpu_queued=$(squeue -u "${USER}" -h -o '%q' 2>/dev/null | grep -c '^gpu_normal$' || true)
debug_queued=$(squeue -u "${USER}" -h -o '%q' 2>/dev/null | grep -c '^debug_normal$' || true)
prev=""
failed=""
for index in "${!stages[@]}"; do
    stage="${stages[$index]}"
    job=$(job_of "$stage")
    if [ -n "$job" ]; then
        state_now=$(job_state "$job")
        case "$state_now" in
            COMPLETED)
                # Already done: the next stage needs *no* dependency (Slurm purges finished jobs
                # from the controller, and ``afterok`` on a purged id is rejected outright).
                prev=""; continue ;;
            RUNNING|PENDING|CONFIGURING|COMPLETING|REQUEUED|SUSPENDED|RESIZING|"")
                prev="$job"; continue ;;
            *) failed="$stage job=$job state=$state_now"; break ;;
        esac
    fi
    if [ "${qs[$index]}" = "debug" ]; then
        if [ "$debug_queued" -ge 1 ]; then
            echo "debug slot busy ($debug_queued job); nothing more submitted"
            exit 0
        fi
        partition=("${DEBUG_EXTRA[@]}")
    else
        if [ "$gpu_queued" -ge "$limit" ]; then
            echo "gpu submission slots full ($gpu_queued jobs, limit $limit); nothing more submitted"
            exit 0
        fi
        partition=()
    fi
    dependency=()
    [ -n "$prev" ] && dependency=(--dependency=afterok:"$prev")
    if [ "$dry" -eq 1 ]; then
        echo "would submit ${names[$index]} [${qs[$index]}] ${dependency[*]:-} :: ${scripts[$index]} ${args[$index]}"
        prev="<dry-${stage}>"
        continue
    fi
    # shellcheck disable=SC2086  # fixed, space-separated plan
    # shellcheck disable=SC2086  # fixed, space-separated plan
    new=$(sbatch --parsable "${partition[@]}" "${dependency[@]}" --job-name="${names[$index]}" \
                 "${scripts[$index]}" ${args[$index]})
    printf '%s\t%s\n' "$stage" "$new" >> "$state"
    echo "submitted ${stage} job=$new [${qs[$index]}] ${dependency[*]:-}"
    prev="$new"
    if [ "${qs[$index]}" = "debug" ]; then debug_queued=$((debug_queued + 1));
    else gpu_queued=$((gpu_queued + 1)); fi
done

if [ -n "$failed" ]; then
    echo "CHAIN FAILED: $failed" >&2
    for stage in "${stages[@]}"; do
        job=$(job_of "$stage")
        [ -z "$job" ] && continue
        case "$(job_state "$job")" in
            PENDING) scancel "$job" 2>/dev/null && echo "cancelled stranded ${stage} job=$job" >&2 ;;
        esac
    done
    if [ "$retry" -eq 1 ]; then
        cut=$(awk -F'\t' -v s="${failed%% *}" '$1 == s {print NR; exit}' "$state")
        if [ -n "$cut" ]; then
            awk -F'\t' -v cut="$cut" 'NR < cut {print}' "$state" > "$state.tmp"
            mv "$state.tmp" "$state"
            echo "retry: dropped the records from ${failed%% *} on; run the driver again to resubmit" >&2
        fi
    else
        echo "pass --retry once the cause is fixed to resubmit the failed stage" >&2
    fi
    exit 1
fi

[ -z "$(job_of eval_phi)" ] && echo "chain still has unsubmitted stages" || echo "chain fully submitted"
