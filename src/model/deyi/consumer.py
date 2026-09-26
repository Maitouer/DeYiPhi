"""Shared loader for frozen DeYi states consumed by downstream models."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import DeYi


def recent_of(cfg) -> int:
    """The ``r<R>`` of the arm to read.

    The DeYi old/recent boundary and the consumer's explicit recent history are **one**
    setting, not two: the consumer's ``recent_history`` *is* the ``r<R>``. A consumer that
    keeps 32 recent items reads the ``k<K>_r32`` arm -- there is no second knob that could
    drift away from it.
    """
    table = cfg.get("recent_history", None) if hasattr(cfg, "get") else None
    if not table:
        raise ValueError(
            "config must declare recent_history as a task -> recent mapping; it is the r<R> "
            "part of the DeYi arm name")
    if cfg.task not in table:
        raise ValueError("recent_history has no entry for task %r" % cfg.task)
    return int(table[cfg.task])


def dataset_of(cfg) -> str:
    """Which dataset the arm was trained on (``product`` alone is ambiguous)."""
    value = cfg.get("dataset", None) if hasattr(cfg, "get") else None
    if value is None:
        raise ValueError(
            "config must declare dataset: the dataset this DeYi arm was trained on")
    return str(value)


def arm_dir(root, dataset, task, num_interests, recent) -> Path:
    return (Path(root) / str(dataset) / str(task)
            / ("k%d_r%d" % (int(num_interests), int(recent))))


def state_path(root, dataset, task, num_interests, split, recent) -> Path:
    if split not in ("train", "test"):
        raise ValueError(f"unsupported DeYi split: {split}")
    return arm_dir(root, dataset, task, num_interests, recent) / "states" / f"{split}.pt"


def load_states(root, dataset, task, num_interests, split, source_rows,
                recent) -> tuple[torch.Tensor, torch.Tensor]:
    """Load states and verify their row contract once at each dataset boundary."""
    path = state_path(root, dataset, task, num_interests, split, recent)
    if not path.is_file():
        raise FileNotFoundError(f"DeYi {split} states are missing: {path}")
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    required = {"format", "states", "mask", "source_row_idx", "num_interests", "state_dim"}
    if required - payload.keys() or payload["format"] != DeYi.STATE_FORMAT:
        raise ValueError(f"unsupported DeYi state artifact: {path}")
    rows = torch.as_tensor(source_rows, dtype=torch.long)
    states = payload["states"]
    mask = payload["mask"].bool()
    expected = (rows.numel(), int(num_interests), int(payload["state_dim"]))
    if tuple(states.shape) != expected or tuple(mask.shape) != expected[:2]:
        raise ValueError(f"DeYi state shape mismatch at {path}: {tuple(states.shape)}")
    if int(payload["num_interests"]) != int(num_interests):
        raise ValueError(f"DeYi interest count mismatch at {path}")
    if not torch.equal(payload["source_row_idx"].long(), rows):
        raise ValueError(f"DeYi source rows do not match canonical data at {path}")
    return states, mask
