"""Device choice, seeding, logging and progress for the phi chain.

The device rule is deliberately conservative: a job that did not receive a GPU allocation
must not quietly take one, because the phi stages run next to training jobs on a shared node.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def has_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    # Slurm exports the allocated devices only when the job really holds a GRES allocation;
    # a CPU-partition job on a GPU node must not grab one.
    return (os.environ.get("SLURM_JOB_GPUS") is not None
            or os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""))


def device(wanted: str = "auto") -> torch.device:
    """``auto`` uses the GPU only when this job actually holds one; ``cuda`` stays authoritative."""
    name = str(wanted or "auto").lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device: cuda was requested but no CUDA device is visible")
        return torch.device("cuda")
    return torch.device("cuda" if has_gpu() else "cpu")


def free_device_bytes(target: torch.device) -> int:
    """How much of ``target`` this process may cache, leaving room for the score blocks."""
    if target.type != "cuda":
        return 0
    try:
        free, _total = torch.cuda.mem_get_info(target)
        # The resident catalog is a cache, not a requirement, so only part of the free memory
        # is offered: the streamed path stays correct when it does not fit.
        return int(free * 0.55)
    except Exception:                                     # pragma: no cover - driver dependent
        return int(torch.cuda.get_device_properties(0).total_memory * 0.55)


def compute_dtype(target: torch.device) -> torch.dtype:
    """bf16 on a GPU (the dot products are memory-bound), fp32 on the CPU."""
    return torch.bfloat16 if target.type == "cuda" else torch.float32


def log_directory(stage: str = "phi"):
    from src.common.progress import run_directory

    configured = os.environ.get("MODEL_LOG_DIR")
    if configured:
        path = Path(configured)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return run_directory(Path("log/model/phi") / stage, os.environ.get("SLURM_JOB_ID"))


def reporter(stage: str, total: int = 1, interval: float = 5.0):
    from src.common.progress import Progress

    return Progress(log_directory(stage), total, job_id=os.environ.get("SLURM_JOB_ID"),
                    interval=interval)
