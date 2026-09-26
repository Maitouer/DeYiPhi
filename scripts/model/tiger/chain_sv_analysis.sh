#!/usr/bin/env bash
# The explicit-memory analysis table on TIGER + single_channel/short_video (Block A/B/C, K = 4).
#
#   bash scripts/model/tiger/chain_sv_analysis.sh [--limit N] [--dry-run] [--retry]
#
# Preparation (one GPU each):
#   pooling   mean of the unit item vectors over K equal time segments of H_old   (compress/pooling)
#   cause     level-1 SID categories, most recent K                            (compress/cause)
#   hicogen   hierarchical semantic clusters                                    (compress/hicogen)
#   chronicle temporally anchored predictive states                             (compress/chronicle)
#   euclidean / cosine  alternative codebooks of the *same* frozen DeYi states   (phi_variants/*)
#
# Arms (four GPUs each, mode = full, K = 4):
#   full_r32_dummy4                      native model + K shared learnable tokens
#   deyi_full_r32_k4/{pooling,cause,hicogen,chronicle}
#   phi_full_r32_k4/{euclidean,cosine}
#
# Idempotent: a stage whose artifact exists is skipped, a failure cancels the stages queued behind
# it and ``--retry`` drops those records so the chain can continue.
#
# State: log/model/tiger/chain/single_channel_short_video_analysis.tsv   (stage <TAB> jobid)
set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"

task=short_video
mode=full
k=4
recent=32
limit=4
dry=0
retry=0
only=""          # optional: keep just these stages (the rest of the table is run elsewhere)
while [ $# -gt 0 ]; do
    case "$1" in
        --limit) limit="$2"; shift 2 ;;
        --only) only="$2"; shift 2 ;;
        --dry-run) dry=1; shift ;;
        --retry) retry=1; shift ;;
        -*) echo "unknown argument: $1" >&2; exit 2 ;;
        *) echo "unexpected argument: $1" >&2; exit 2 ;;
    esac
done

PY="${DEYIPHI_PYTHON:-python}"
state_dir="log/model/tiger/chain"
state="$state_dir/single_channel_${task}_analysis.tsv"
mkdir -p "$state_dir"
touch "$state"

COMPRESS=output/model/compress
VQ_ROOT=output/model/phi_variants
ONE_GPU=(--partition=gpu --qos=gpu_normal --gres=gpu:nvidia_h100_80gb_hbm3:1
         --cpus-per-task=10 --mem=120000M)

run_dummy=$(TIGER_DUMMY_MEMORY=1 TIGER_DUMMY_SLOTS=$k "$PY" -c "from src.model.tiger import load_config; print(load_config('config/model/tiger.yaml','$task','$mode').run_dir)")
run_of_method() {  # <method>
    "$PY" -c "from src.model.tiger import load_config; print(load_config('config/model/tiger.yaml','$task','$mode',deyi_k=$k,deyi_root='$COMPRESS/$1').run_dir)"
}
run_of_vq() {      # <metric>
    # On the first pass the vocabulary does not exist yet, and load_config refuses to name an arm
    # whose codebook is missing; fall back to the same naming rule the config applies
    # (<output_dir>/<dataset>/<task>/phi_<mode>_r<R>_k<K>_<vocabulary directory>/seed<seed>).
    local metric="$1" dir
    dir=$(PHI_OUTPUT_DIR="$VQ_ROOT/$metric" "$PY" -c "from src.model.tiger import load_config; print(load_config('config/model/tiger.yaml','$task','$mode',phi=True,phi_deyi_k=$k).run_dir)" 2>/dev/null) || dir=""
    [ -n "$dir" ] || dir="output/model/tiger/single_channel/$task/phi_${mode}_r${recent}_k${k}_${metric}/seed2026"
    printf '%s\n' "$dir"
}

labels=() scripts=() sargs=() envs=() arts=() names=() extras=() reqs=() kinds=()
# ``add <label> <script> <args> <env> <artifact> <name> [one-gpu] [requires]``: ``requires`` names
# the upstream artifact this stage consumes, so preparation stages can run in parallel while every
# arm still waits for its own frozen bank / vocabulary.
add() { labels+=("$1"); scripts+=("$2"); sargs+=("$3"); envs+=("$4"); arts+=("$5"); names+=("$6"); extras+=("${7:-}"); reqs+=("${8:-}"); kinds+=("${9:-other}"); }

# A preempted sft leaves a mid-run ``train_summary.json`` behind, so a plain file test would call
# the stage finished and the evaluation would then refuse an incomplete run. Every sft stage is
# therefore gated on the summary's ``training_complete`` flag (the same rule sv_full_arms.sh uses).
stage_done() {   # <index>
    local art="${arts[$1]}"
    [ -f "$art" ] || return 1
    if [ "${kinds[$1]}" = "sft" ]; then
        grep -q '"training_complete": true' "$art" || return 1
    fi
    return 0
}

# ---- preparation -------------------------------------------------------------------------------
add prep_hicogen   scripts/model/baseline/train_states.sbatch "hicogen $task $recent $k"   "" "$COMPRESS/hicogen/single_channel/$task/k${k}_r${recent}/states/test.pt"   "baseline:hicogen"
add prep_chronicle scripts/model/baseline/train_states.sbatch "chronicle $task $recent $k" "" "$COMPRESS/chronicle/single_channel/$task/k${k}_r${recent}/states/test.pt" "baseline:chronicle"
add prep_pooling   scripts/model/baseline/pooling.sbatch "$task $recent $k"               "" "$COMPRESS/pooling/single_channel/$task/k${k}_r${recent}/states/test.pt"   "baseline:pooling"
add prep_cause     scripts/model/baseline/cause.sbatch "$task $recent $k"                 "" "$COMPRESS/cause/single_channel/$task/k${k}_r${recent}/states/test.pt"     "baseline:cause" "1"
add prep_euclidean scripts/model/phi/variant.sbatch "euclidean $task $k $recent"          "" "$VQ_ROOT/euclidean/single_channel/$task/k${k}_r${recent}/tokens/test.pt" "phi:variant:euclidean"
add prep_cosine    scripts/model/phi/variant.sbatch "cosine $task $k $recent"             "" "$VQ_ROOT/cosine/single_channel/$task/k${k}_r${recent}/tokens/test.pt"    "phi:variant:cosine"

# ---- arms --------------------------------------------------------------------------------------
add sft_dummy  scripts/model/tiger/sft.sbatch "$task $mode" "TIGER_DUMMY_MEMORY=1,TIGER_DUMMY_SLOTS=$k" "$run_dummy/train_summary.json" "tiger:sft:$task:dummy$k" "" "" sft
add eval_dummy scripts/model/tiger/eval.sbatch "$task $mode" "TIGER_DUMMY_MEMORY=1,TIGER_DUMMY_SLOTS=$k" "$run_dummy/eval/metrics.json"  "tiger:eval:$task:dummy$k"
for method in pooling cause hicogen chronicle; do
    run=$(run_of_method "$method")
    add "sft_${method}"  scripts/model/tiger/deyi_sft.sbatch "$task $mode $k" "TIGER_DEYI_ROOT=$COMPRESS/$method" "$run/train_summary.json" "tiger:sft:$task:$method" "" "$COMPRESS/$method/single_channel/$task/k${k}_r${recent}/states/test.pt" sft
    add "eval_${method}" scripts/model/tiger/deyi_eval.sbatch "$task $mode $k" "TIGER_DEYI_ROOT=$COMPRESS/$method" "$run/eval/metrics.json"  "tiger:eval:$task:$method"
done
for metric in euclidean cosine; do
    run=$(run_of_vq "$metric")
    add "sft_${metric}"  scripts/model/tiger/phi_sft.sbatch "$task $mode $k" "PHI_OUTPUT_DIR=$VQ_ROOT/$metric" "$run/train_summary.json" "tiger:sft:$task:$metric" "" "$VQ_ROOT/$metric/single_channel/$task/k${k}_r${recent}/tokens/test.pt" sft
    add "eval_${metric}" scripts/model/tiger/phi_eval.sbatch "$task $mode $k" "PHI_OUTPUT_DIR=$VQ_ROOT/$metric" "$run/eval/metrics.json"  "tiger:eval:$task:$metric"
done

if [ -n "$only" ]; then
    sel_labels=(); sel_scripts=(); sel_sargs=(); sel_envs=(); sel_arts=(); sel_names=(); sel_extras=(); sel_reqs=(); sel_kinds=()
    for index in "${!labels[@]}"; do
        for want in $only; do
            if [ "${labels[$index]}" = "$want" ]; then
                sel_labels+=("${labels[$index]}"); sel_scripts+=("${scripts[$index]}")
                sel_sargs+=("${sargs[$index]}");   sel_envs+=("${envs[$index]}")
                sel_arts+=("${arts[$index]}");     sel_names+=("${names[$index]}")
                sel_extras+=("${extras[$index]}"); sel_reqs+=("${reqs[$index]}")
                sel_kinds+=("${kinds[$index]}")
                break
            fi
        done
    done
    labels=("${sel_labels[@]}"); scripts=("${sel_scripts[@]}"); sargs=("${sel_sargs[@]}")
    envs=("${sel_envs[@]}"); arts=("${sel_arts[@]}"); names=("${sel_names[@]}")
    extras=("${sel_extras[@]}"); reqs=("${sel_reqs[@]}"); kinds=("${sel_kinds[@]}")
fi

job_of() { awk -F'\t' -v stage="$1" '$1 == stage {print $2}' "$state" | tail -1; }
job_state() { sacct -j "$1" --format=State -n -X 2>/dev/null | head -1 | tr -d ' '; }

gpu_queued=$(squeue -u "$USER" -h -o '%q' 2>/dev/null | grep -c '^gpu_normal$' || true)
prev=""
failed=""
echo "sv analysis: ${#labels[@]} stages, task=${task}, mode=${mode}, K=${k}, recent=${recent}, budget ${gpu_queued}/${limit}"
for index in "${!labels[@]}"; do
    label="${labels[$index]}"
    job=$(job_of "$label")
    if [ -n "$job" ]; then
        state_now=$(job_state "$job")
        case "$state_now" in
            COMPLETED) continue ;;
            RUNNING|PENDING|CONFIGURING|COMPLETING|REQUEUED|SUSPENDED|RESIZING|"")
                continue ;;
            *) failed="$label job=$job state=$state_now"; break ;;
        esac
    fi
    if stage_done "$index"; then
        echo "   [done]      $label"
        continue
    fi
    if [ -n "${reqs[$index]}" ] && [ ! -e "${reqs[$index]}" ]; then
        echo "   [wait]      $label needs ${reqs[$index]}"
        exit 0
    fi
    if [ "$gpu_queued" -ge "$limit" ]; then
        echo "   [wait]      submission budget full ($gpu_queued/$limit)"
        exit 0
    fi
    extra=()
    [ "${extras[$index]}" = "1" ] && extra=("${ONE_GPU[@]}")
    # One explicit rule: only an arm's evaluation is dependent, and only on its own training job.
    # Preparation stages are independent (they are gated by ``requires`` above), so producers and
    # the arms of other methods overlap instead of forming one long chain.
    dependency=()
    case "$label" in eval_*)
        sft_job=$(job_of "${labels[$((index - 1))]}")
        case "$(job_state "$sft_job")" in
            RUNNING|PENDING|CONFIGURING|COMPLETING|REQUEUED|SUSPENDED|RESIZING)
                dependency=(--dependency=afterok:"$sft_job") ;;
        esac ;;
    esac
    export_arg="ALL"
    [ -n "${envs[$index]}" ] && export_arg="ALL,${envs[$index]}"
    if [ "$dry" -eq 1 ]; then
        echo "   [would]     $label ${dependency[*]:-} env=${export_arg} :: ${scripts[$index]} ${sargs[$index]} ${extra[*]:-}"
        continue
    fi
    new=$(sbatch --parsable --export="$export_arg" "${extra[@]}" "${dependency[@]}" \
                 --job-name="${names[$index]}" "${scripts[$index]}" ${sargs[$index]})
    printf '%s\t%s\n' "$label" "$new" >> "$state"
    echo "   [submitted] $label job=$new ${dependency[*]:-}"
    gpu_queued=$((gpu_queued + 1))
done

if [ -n "$failed" ]; then
    echo "CHAIN FAILED: $failed" >&2
    # No blanket cancellation here: preemptions can hit any stage at any time, and cancelling every
    # other queued stage would also kill unrelated work that could have continued or resumed. A
    # stage that must not run on its own is already gated by ``requires`` or an ``afterok``.
    if [ "$retry" -eq 1 ]; then
        cut=$(awk -F'\t' -v s="${failed%% *}" '$1 == s {print NR; exit}' "$state")
        [ -n "$cut" ] && awk -F'\t' -v cut="$cut" 'NR < cut {print}' "$state" > "$state.tmp" && mv "$state.tmp" "$state"
        echo "retry: dropped the records from ${failed%% *} on; run the driver again" >&2
    fi
    exit 1
fi

remaining=0
for index in "${!arts[@]}"; do stage_done "$index" || remaining=1; done
[ "$remaining" -eq 0 ] && echo "ALL DONE" || echo "chain still has unsubmitted stages"
