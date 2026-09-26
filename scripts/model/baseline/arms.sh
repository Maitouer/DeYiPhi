#!/usr/bin/env bash
# Submit the three controlled-baseline arms of 04_baseline.md on the eight-GPU node.
#
#   bash scripts/model/baseline/arms.sh [task] [recent] [K] [full|recent] [--dry-run]
#
# ``mode`` is the history view the Tiger backbone sees (full = whole history, recent = the
# last ``recent`` interactions). The frozen memory bank is the same file either way: the
# bank is resolved from the method root, the task and K, exactly like the DeYi teacher.
#
# Each arm conditions TIGER on one frozen memory bank (`output/model/compress/<method>`) through
# the ordinary continuous-memory path, so the three rows differ from the DeYi row only in how
# H_old -> M was built. sft runs first and its eval is chained with ``afterok``.
#
# Why gpu_heavy/gpu_priv: the whole comparison is a handful of short jobs, and the account holds a
# preempting QOS, so the arms take the whole node as a single eight-GPU allocation instead of
# queueing behind
# multi-day jobs on the four-GPU partition.
set -euo pipefail
# Resolve this project from the script's own location. The hard-coded default used to be the *other*
# account's project, so a caller without DEYIPHI_ROOT in its environment exported that path into the
# submitted jobs; they then cd'd there, could not write a single stage artifact and died in seconds
# with no log directory (Slurm swallows it: these batches set --output/--error to /dev/null).
root="${DEYIPHI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export DEYIPHI_ROOT="$root"
cd "$root"
task="${1:-product}"
recent="${2:-16}"
k="${3:-4}"
mode="${4:-recent}"
dry=0
for arg in "$@"; do [ "$arg" = "--dry-run" ] && dry=1; done
case "$mode" in
    full|recent) ;;
    --dry-run) mode=recent ;;
    *) echo "unknown mode: $mode (choose full or recent)" >&2; exit 2 ;;
esac
methods=(cause chronicle hicogen)
python_bin="${DEYIPHI_PYTHON:-python}"

# Quota aware and idempotent: the account may hold only a handful of queued jobs, so the loop stops
# cleanly when the budget is full (rerun the script to pick up the rest) and never resubmits a
# method whose arm already finished training.
queued=$(squeue -u "${USER}" -h -o "%i" 2>/dev/null | wc -l)
submit_budget=$(( ${MAX_SUBMIT:-4} - queued ))
echo "[arms] task=${task} mode=${mode} k=${k} queued=${queued} budget=${submit_budget}"
for method in "${methods[@]}"; do
    deyi_root="output/model/compress/${method}"
    if [ ! -f "${deyi_root}/single_channel/${task}/k${k}_r${recent}/states/train.pt" ]; then
        echo "skip ${method}: ${deyi_root}/single_channel/${task}/k${k}_r${recent}/states/train.pt is missing"
        continue
    fi
    run=$("${python_bin}" -c "from src.model.tiger import load_config; print(load_config('config/model/tiger.yaml','${task}','${mode}',deyi_k=${k},deyi_root='${deyi_root}').run_dir)")
    # The trainer saves an intermediate train_summary.json *and* checkpoint.pt every save interval
    # with "training_complete": false (observed: step 1000 of 1490), so the existence of those files
    # is not proof that the arm finished. Only the flag is -- otherwise this script would submit the
    # evaluation of a half-trained model.
    trained=0
    if [ -f "${run}/train_summary.json" ] && "${python_bin}" -c \
        'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("training_complete") else 1)' \
        "${run}/train_summary.json"; then
        trained=1
    fi
    if [ "$trained" = 1 ] && [ -f "${run}/eval/metrics.json" ]; then
        echo "skip ${method}: ${run} already trained and evaluated"
        continue
    fi
    if squeue -u "${USER}" -h -o "%j" | grep -qx "baseline:${method}:${mode}:sft"; then
        echo "skip ${method}: already queued"
        continue
    fi
    if [ "$submit_budget" -lt 2 ]; then
        echo "stop: submit budget exhausted before ${method}; rerun when jobs finish"
        break
    fi
    # Pin the interpreter scratch for the whole wave: every stage of the job then shares one
    # writable directory instead of falling back to a per-node default.
    TMPDIR_ROOT="${DEYIPHI_TMPDIR:-$PWD/tmp/deyiphi}"
    exports="ALL,TIGER_NPROC=8,TIGER_DEYI_ROOT=${deyi_root},DEYIPHI_ROOT=${root},DEYIPHI_PYTHON=${python_bin},DEYIPHI_TMPDIR=${TMPDIR_ROOT}"
    common=(--partition=gpu_heavy --qos=gpu_priv --gres=gpu:nvidia_h100_80gb_hbm3:8
            --cpus-per-task=88 --mem=1000000M --time=1:00:00 --export="${exports}")
    if [ "$dry" -eq 1 ]; then
        echo "would submit baseline:${method}:${mode}:sft  -> ${deyi_root}"
        echo "would submit baseline:${method}:${mode}:eval (afterok)"
        continue
    fi
    # An arm that is already trained only needs its evaluation: chaining the second job on a
    # dependency that no longer exists would leave it pending forever.
    sft=""
    dep=()
    if [ "$trained" != 1 ]; then
        sft=$(sbatch --parsable "${common[@]}" --job-name="baseline:${method}:${mode}:sft" \
                     scripts/model/tiger/deyi_sft.sbatch "$task" "$mode" "$k")
        echo "submitted baseline:${method}:${mode}:sft job=$sft"
        dep=(--dependency=afterok:"$sft")
        submit_budget=$(( submit_budget - 1 ))
    fi
    evaluation=$(sbatch --parsable "${common[@]}" ${dep[@]+"${dep[@]}"} \
                        --job-name="baseline:${method}:${mode}:eval" \
                        scripts/model/tiger/deyi_eval.sbatch "$task" "$mode" "$k")
    echo "submitted baseline:${method}:${mode}:eval job=$evaluation${sft:+ (afterok:${sft})}"
    submit_budget=$(( submit_budget - 1 ))
done
