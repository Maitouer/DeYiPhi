#!/usr/bin/env bash
# Upstream-only k sweep: DeYi teacher -> exported states -> phi vocabulary, for several K on one
# task. **No downstream sft/eval** -- this driver stops when every teacher's material is on disk.
#
#   bash scripts/model/deyi/chain_ksweep.sh [<task>] [--ks "1 2 8 16"] [--recent R] \
#                                           [--dry-run] [--retry]
#
# Why it runs on the debug partition: every stage here is a single-GPU job (the product teacher's
# device candidate table is 2.2 GiB), and the debug QOS is a *separate* budget from gpu_normal. So
# the sweep never competes with the four-GPU downstream chains -- neither for the four GPUs of
# ``gpu_normal`` nor for one of its four submission slots. The price is that the debug QOS runs one
# job at a time, i.e. the sweep progresses strictly serially in the background.
#
# Stages for every K, in order (each ``--dependency=afterok`` on the one before):
#
#   deyi_train_k<K> -> deyi_encode_k<K> -> phi_k<K>
#
# Artifacts:
#   output/model/deyi/<dataset>/<task>/k<K>_r<R>/           checkpoint.pt, run_config.yaml, states/
#   output/model/phi/<dataset>/<task>/k<K>_r<R>/            codebook.pt, tokens/, routes/, audit.json
#
# State: log/model/deyi/chain/<dataset>_<task>_ksweep.tsv   (stage <TAB> jobid)
set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"

PY="${DEYIPHI_PYTHON:-python}"
dataset=single_channel
task=product
ks=()
recent=""
dry=0
retry=0
while [ $# -gt 0 ]; do
    case "$1" in
        --ks) read -r -a ks <<< "$2"; shift 2 ;;
        --recent) recent="$2"; shift 2 ;;
        --dry-run) dry=1; shift ;;
        --retry) retry=1; shift ;;
        -*) echo "unknown argument: $1" >&2; exit 2 ;;
        *) task="$1"; shift ;;
    esac
done
case "$task" in short_video|ad|product) ;; *) echo "unknown task $task" >&2; exit 2 ;; esac
if [ ${#ks[@]} -eq 0 ]; then ks=(1 2 8 16); fi
if [ -z "$recent" ]; then
    recent=$("$PY" -c "from omegaconf import OmegaConf; print(OmegaConf.load('config/model/deyi.yaml').recent['$task'])")
fi

state_dir="log/model/deyi/chain"
state="$state_dir/${dataset}_${task}_ksweep.tsv"
mkdir -p "$state_dir"
touch "$state"

stages=()
scripts=()
args=()
names=()
qs=()
for k in "${ks[@]}"; do
    stages+=("deyi_train_k${k}" "deyi_encode_k${k}" "phi_k${k}")
    scripts+=("scripts/model/deyi/train.sbatch" "scripts/model/deyi/encode.sbatch"
              "scripts/model/phi/chain.sbatch")
    args+=("$task $k $recent" "$task $k $recent" "$task $k $recent")
    names+=("deyi:train:${task}:k${k}:r${recent}" "deyi:encode:${task}:k${k}:r${recent}"
            "phi:chain:${task}:k${k}:r${recent}")
    qs+=("debug" "debug" "debug")
done

# The single-GPU shapes these stages need on ``debug`` (the scripts default to the gpu partition).
# Plain ``--gres=gpu:1`` on purpose: the debug node also advertises small MIG slices, but asking
# for the slice type explicitly is refused with "reserved for jobs in higher priority
# partitions". The untyped request is what lets these stages grab a full GPU the moment one
# frees, which is how every earlier stage of this sweep ran.
# The debug node holds a single MIG slice shared with every other user, so it is regularly blocked
# (``Nodes required ... reserved`` / ``AssocGrpGRES``). ``KSWEEP_PARTITION``/``KSWEEP_QOS`` let the
# same sweep be driven on the four-GPU partition instead; the default stays the debug budget.
KSWEEP_PARTITION="${KSWEEP_PARTITION:-debug}"
KSWEEP_QOS="${KSWEEP_QOS:-debug_normal}"
DEBUG_EXTRA=(--partition="$KSWEEP_PARTITION" --qos="$KSWEEP_QOS" --gres=gpu:1
             --cpus-per-task=10 --mem=120000M)

job_of() { awk -F'\t' -v stage="$1" '$1 == stage {print $2}' "$state" | tail -1; }
job_state() { sacct -j "$1" --format=State -n -X 2>/dev/null | head -1 | tr -d ' '; }
in_flight=$(squeue -u "${USER}" -h -o '%q' 2>/dev/null | grep -c "^${KSWEEP_QOS}$" || true)

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
    if [ "$in_flight" -ge 1 ]; then
        echo "one sweep stage already queued on ${KSWEEP_QOS}; nothing more submitted"
        exit 0
    fi
    dependency=()
    [ -n "$prev" ] && dependency=(--dependency=afterok:"$prev")
    if [ "$dry" -eq 1 ]; then
        echo "would submit ${names[$index]} ${dependency[*]:-} :: ${scripts[$index]} ${args[$index]}"
        prev="<dry-${stage}>"
        continue
    fi
    # Every stage carries the same shape on the command line, which wins over each script's own
    # #SBATCH block (``train.sbatch`` defaults to gpu/gpu_normal, ``phi/chain.sbatch`` to debug).
    new=$(sbatch --parsable "${DEBUG_EXTRA[@]}" "${dependency[@]}" \
                 --job-name="${names[$index]}" "${scripts[$index]}" ${args[$index]})
    printf '%s\t%s\n' "$stage" "$new" >> "$state"
    echo "submitted ${stage} job=$new ${dependency[*]:-}"
    prev="$new"
    in_flight=$((in_flight + 1))
done

if [ -n "$failed" ]; then
    echo "SWEEP FAILED: $failed" >&2
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
            echo "retry: dropped the records from ${failed%% *} on; run again to resubmit" >&2
        fi
    else
        echo "pass --retry once the cause is fixed" >&2
    fi
    exit 1
fi

last="phi_k${ks[${#ks[@]}-1]}"
[ -z "$(job_of "$last")" ] && echo "sweep still has unsubmitted stages" || echo "sweep fully submitted"
