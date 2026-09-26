"""Artifact paths for the DeYi chain.

Layout (see ``00-Reconstruct/00_item_semantic_space.md`` and ``01_deyi_encoder.md``)::

    output/model/deyi/
    └── <dataset>/<task>/k<K>_r<R>/
        ├── checkpoint.pt, run_config.yaml, train_log.jsonl
        └── states/{train,test}.pt

``recent`` is part of the arm name because it decides how old and recent history are cut
and therefore which model is trained; two runs that differ only in ``recent`` must not share
an output directory.
"""

from __future__ import annotations

import json
from pathlib import Path


CHECKPOINT_FILENAME = "checkpoint.pt"
# The single format string for the item-space manifest: ``pca.py`` writes it and
# ``item_space.completed`` compares against it, so both must read this one constant.
ITEM_SPACE_FORMAT = "deyi_item_semantic_pca"
ARM_PREFIX = "k"


def arm_name(cfg) -> str:
    return "%s%d_r%d" % (ARM_PREFIX, int(cfg.deyi.num_interests), int(recent(cfg)))


def dataset(cfg) -> str:
    """Which dataset this arm reads. ``cfg.dataset`` (CLI override) wins over the task entry."""
    override = cfg.get("dataset") if hasattr(cfg, "get") else None
    if override:
        return str(override)
    spec = cfg.tasks[cfg.task]
    name = spec.get("dataset") if hasattr(spec, "get") else None
    if not name:
        raise ValueError(
            "tasks.%s.dataset must name the dataset that owns this task" % cfg.task)
    return str(name)


def recent(cfg) -> int:
    """The one place ``recent`` is read from, so CLI overrides always win."""
    table = cfg.get("recent") if hasattr(cfg, "get") else None
    if not table or cfg.task not in table:
        raise ValueError("recent must map task -> window; missing entry for %r" % cfg.task)
    return int(table[cfg.task])


def output_root(cfg) -> Path:
    return Path(str(cfg.deyi.output_root))


def root(cfg) -> Path:
    """One trained arm: ``<output_root>/<dataset>/<task>/k<K>_r<R>``.

    The dataset is part of the path because ``product`` exists in both datasets; without it
    two different arms would write to the same directory.
    """
    return output_root(cfg) / dataset(cfg) / str(cfg.task) / arm_name(cfg)


def item_space(cfg) -> Path:
    configured = cfg.deyi.get("item_space")
    if not configured:
        raise ValueError("deyi.item_space must be configured explicitly")
    return Path(str(configured))


def pca(cfg) -> Path:
    return item_space(cfg) / "pca.pt"


def embeddings(cfg) -> Path:
    """float16 ``(N, d)`` PCA coordinates ``p(i)``; consumers L2-normalize to get ``e(i)``."""
    return item_space(cfg) / "embeddings.npy"


def item_space_manifest(cfg) -> Path:
    return item_space(cfg) / "manifest.json"


def item_space_stats(cfg) -> Path:
    return item_space(cfg) / "stats.json"


def checkpoint(cfg) -> Path:
    return root(cfg) / CHECKPOINT_FILENAME


def run_config(cfg) -> Path:
    return root(cfg) / "run_config.yaml"


def train_log(cfg) -> Path:
    return root(cfg) / "train_log.jsonl"


def states(cfg, split: str) -> Path:
    if split not in ("train", "test"):
        raise ValueError(f"unsupported DeYi split: {split}")
    return root(cfg) / "states" / f"{split}.pt"


def read_manifest(path: Path):
    if not Path(path).is_file():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
