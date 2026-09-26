#!/usr/bin/env bash
# The four controlled arms of the explicit-memory analysis table (TIGER + Product, mode=full).
#
#   bash scripts/model/tiger/chain_analysis_arms.sh [--limit N] [--dry-run] [--retry]
#
#   Block A   Dummy    native model + one shared trainable token  (TIGER_DUMMY_MEMORY=1)
#             Pooling  one mean-pooled continuous summary of H_old (compress/pooling, K=1)
#   Block C   Uniform VQ    same frozen DeYi K=1 states, Euclidean codebook   (phi_variants/euclidean)
#             Geometric VQ  same states, cosine codebook                     (phi_variants/cosine)
#
# Every stage is a four-GPU job on the gpu partition, so the arms run strictly one at a time; the
# driver keeps the submission budget busy, skips a stage whose artifact exists, cancels the stages
# queued behind a failure and continues with ``--retry`` once the cause is fixed.
#
# State: log/model/tiger/chain/single_channel_product_analysis.tsv   (stage <TAB> jobid)
set -euo pipefail
cd "${DEYIPHI_ROOT:-.}"

task=product
mode=full
k=1
limit=4
dry=0
retry=0
while [ $# -gt 0 ]; do
    case "$1" in
        --limit) limit="$2"; shift 2 ;;
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

POOL_ROOT=output/model/compress/pooling
VQ_ROOT=output/model/phi_variants

run_dummy=$(TIGER_DUMMY_MEMORY=1 "$PY" -c "from src.model.tiger import load_config; print(load_config('config/model/tiger.yaml','$task','$mode').run_dir)")
run_pool=$("$PY" -c "from src.model.tiger import load_config; print(load_config('config/model/tiger.yaml','$task','$mode',deyi_k=$k,deyi_root='$POOL_ROOT').run_dir)")
run_euc=$(PHI_OUTPUT_DIR="$VQ_ROOT/euclidean" "$PY" -c "from src.model.tiger import load_config; print(load_config('config/model/tiger.yaml','$task','$mode',phi=True,phi_deyi_k=$k).run_dir)")
run_cos=$(PHI_OUTPUT_DIR="$VQ_ROOT/cosine" "$PY" -c "from src.model.tiger import load_config; print(load_config('config/model/tiger.yaml','$task','$mode',phi=True,phi_deyi_k=$k).run_dir)")

labels=() scripts=() sargs=() envs=() arts=() names=()
add() { labels+=("$1"); scripts+=("$2"); sargs+=("$3"); envs+=("$4"); arts+=("$5"); names+=("$6"); }

add sft_dummy  scripts/model/tiger/sft.sbatch "product $mode"      "TIGER_DUMMY_MEMORY=1"            "$run_dummy/train_summary.json" "tiger:sft:product:dummy"
add eval_dummy scripts/model/tiger/eval.sbatch "product $mode"     "TIGER_DUMMY_MEMORY=1"            "$run_dummy/eval/metrics.json"  "tiger:eval:product:dummy"
add sft_pooling  scripts/model/tiger/deyi_sft.sbatch "product $mode $k"  "TIGER_DEYI_ROOT=$POOL_ROOT"        "$run_pool/train_summary.json" "tiger:sft:product:pooling"
add eval_pooling scripts/model/tiger/deyi_eval.sbatch "product $mode $k" "TIGER_DEYI_ROOT=$POOL_ROOT"        "$run_pool/eval/metrics.json"  "tiger:eval:product:pooling"
add sft_uniform_vq  scripts/model/tiger/phi_sft.sbatch "product $mode $k"  "PHI_OUTPUT_DIR=$VQ_ROOT/euclidean" "$run_euc/train_summary.json" "tiger:sft:product:uniform_vq"
add eval_uniform_vq scripts/model/tiger/phi_eval.sbatch "product $mode $k" "PHI_OUTPUT_DIR=$VQ_ROOT/euclidean" "$run_euc/eval/metrics.json"  "tiger:eval:product:uniform_vq"
add sft_geometric_vq  scripts/model/tiger/phi_sft.sbatch "product $mode $k"  "PHI_OUTPUT_DIR=$VQ_ROOT/cosine"   "$run_cos/train_summary.json" "tiger:sft:product:geometric_vq"
add eval_geometric_vq scripts/model/tiger/phi_eval.sbatch "product $mode $k" "PHI_OUTPUT_DIR=$VQ_ROOT/cosine"   "$run_cos/eval/metrics.json"  "tiger:eval:product:geometric_vq"

job_of() { awk -F'\t' -v stage="$1" '$1 == stage {print $2}' "$state" | tail -1; }
job_state() { sacct -j "$1" --format=State -n -X 2>/dev/null | head -1 | tr -d ' '; }

gpu_queued=$(squeue -u "$USER" -h -o '%q' 2>/dev/null | grep -c '^gpu_normal$' || true)
prev=""
failed=""
echo "analysis arms: ${#labels[@]} stages, task=${task}, mode=${mode}, K=${k}, budget ${gpu_queued}/${limit}"
for index in "${!labels[@]}"; do
    label="${labels[$index]}"
    job=$(job_of "$label")
    if [ -n "$job" ]; then
        state_now=$(job_state "$job")
        case "$state_now" in
            COMPLETED) prev=""; continue ;;
            RUNNING|PENDING|CONFIGURING|COMPLETING|REQUEUED|SUSPENDED|RESIZING|"")
                prev="$job"; continue ;;
            *) failed="$label job=$job state=$state_now"; break ;;
        esac
    fi
    if [ -f "${arts[$index]}" ]; then
        echo "   [done]      $label"
        prev=""
        continue
    fi
    if [ "$gpu_queued" -ge "$limit" ]; then
        echo "   [wait]      submission budget full ($gpu_queued/$limit)"
        exit 0
    fi
    dependency=()
    [ -n "$prev" ] && dependency=(--dependency=afterok:"$prev")
    export_arg="ALL"
    [ -n "${envs[$index]}" ] && export_arg="ALL,${envs[$index]}"
    if [ "$dry" -eq 1 ]; then
        echo "   [would]     $label ${dependency[*]:-} env=${export_arg} :: ${scripts[$index]} ${sargs[$index]}"
        prev="<dry-$label>"
        continue
    fi
    new=$(sbatch --parsable --export="$export_arg" "${dependency[@]}" \
                 --job-name="${names[$index]}" "${scripts[$index]}" ${sargs[$index]})
    printf '%s\t%s\n' "$label" "$new" >> "$state"
    echo "   [submitted] $label job=$new ${dependency[*]:-}"
    prev="$new"
    gpu_queued=$((gpu_queued + 1))
done

if [ -n "$failed" ]; then
    echo "CHAIN FAILED: $failed" >&2
    for label in "${labels[@]}"; do
        job=$(job_of "$label"); [ -z "$job" ] && continue
        case "$(job_state "$job")" in
            PENDING) scancel "$job" 2>/dev/null && echo "   cancelled stranded $label job=$job" >&2 ;;
        esac
    done
    if [ "$retry" -eq 1 ]; then
        cut=$(awk -F'\t' -v s="${failed%% *}" '$1 == s {print NR; exit}' "$state")
        [ -n "$cut" ] && awk -F'\t' -v cut="$cut" 'NR < cut {print}' "$state" > "$state.tmp" && mv "$state.tmp" "$state"
        echo "retry: dropped the records from ${failed%% *} on; run the driver again" >&2
    fi
    exit 1
fi

remaining=0
for art in "${arts[@]}"; do [ -f "$art" ] || remaining=1; done
[ "$remaining" -eq 0 ] && echo "ALL DONE" || echo "chain still has unsubmitted stages"
