"""Small deterministic runtime helpers for DeYi."""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def save_pt(payload, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary); os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record, sort_keys=True) + "\n")


def progress(label: str, done: int, total: int, started: float, extra: str = "") -> None:
    elapsed = time.time() - started; eta = elapsed / done * (total - done) if done else 0.0
    suffix = f" | {extra}" if extra else ""
    print(f"[{label}] {done}/{total} ({100 * done / total:.1f}%) elapsed {elapsed:.0f}s eta {eta:.0f}s{suffix}", flush=True)


def cosine_scheduler(optimizer, steps: int, warmup_ratio: float):
    warmup = max(1, int(steps * warmup_ratio))
    def schedule(step: int) -> float:
        if step < warmup: return (step + 1) / warmup
        ratio = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * ratio))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


# --- unified progress (same style as the data pipeline and fit-PCA) ------------------------
# The definitions below intentionally shadow the legacy helpers above: every existing
# ``progress(label, done, total, started)`` call now also feeds the shared Progress reporter,
# so deyi stages get console.log + progress.json + heartbeat + a closing summary for free.
# The run directory comes from $MODEL_LOG_DIR (set by scripts/model/common_logging.sh) or is
# created on the fly under log/model/deyi/<stage>/.
import atexit as _atexit
import os as _os
from pathlib import Path as _Path

_PROGRESS = None


def log_directory(stage: str = "train"):
    from src.common.progress import run_directory

    configured = _os.environ.get("MODEL_LOG_DIR")
    if configured:
        path = _Path(configured)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return run_directory(_Path("log/model/deyi") / stage, _os.environ.get("SLURM_JOB_ID"))


def attach_progress(progress):
    global _PROGRESS
    _PROGRESS = progress
    return progress


def active_progress(stage: str = "train"):
    global _PROGRESS
    if _PROGRESS is None:
        from src.common.progress import Progress

        _PROGRESS = Progress(log_directory(stage), 1, interval=5.0)
        _atexit.register(lambda: _PROGRESS.finish("deyi %s finished" % stage))
    return _PROGRESS


def progress(label, done=0, total=0, started=0.0, extra=""):
    reporter = active_progress()
    if total:
        reporter.stage = label
        reporter.tick(int(done), int(total), extra or "")
    else:
        reporter.line(label, extra or "", force=True)
