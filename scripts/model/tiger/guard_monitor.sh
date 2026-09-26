#!/usr/bin/env bash

# Audit actual assigned-GPU behavior; this never changes the compute workload.
start_tiger_gpu_monitor() {
    [[ "${TIGER_GPU_MONITOR:-1}" == "1" ]] || return 0
    [[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]] || return 0
    command -v nvidia-smi >/dev/null 2>&1 || return 0

    local log_dir="${TIGER_GPU_LOG_DIR:-$PWD/log/gpu-monitor}"
    local log_file="${log_dir}/gpu-${SLURM_JOB_ID}.csv"
    mkdir -p "$log_dir"
    nvidia-smi --id="$CUDA_VISIBLE_DEVICES" \
        --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits -l 15 >>"$log_file" 2>&1 &
    TIGER_GPU_MONITOR_PID=$!
    trap 'kill "${TIGER_GPU_MONITOR_PID:-}" 2>/dev/null || true' EXIT
    echo "[tiger/guard] sampling assigned GPUs every 15 seconds: $log_file"
}
