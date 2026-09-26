"""The effective phi configuration.

``config/model/deyi.yaml`` is the base: the task list, the dataset that owns each task, the
old/recent cut (``recent``) and the history cap are the *same* settings the frozen teacher was
trained with, so phi cannot read a different dataset or a different recent window than the
teacher it consumes. ``config/model/phi.yaml`` adds phi's own sections on top.

Environment overrides live here, in one place, because an override that reaches the tokenizer
but not the reservoir is how a run silently mixes two configurations.
"""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import OmegaConf

from src.model.deyi import artifacts as deyi_artifacts
from src.model.deyi.config import load_config as load_deyi_config

DEFAULT = "config/model/phi.yaml"

REQUIRED = ("output_dir", "vocabulary", "reservoir", "estimator", "codebook", "router")


def _apply_env(cfg) -> None:
    """``PHI_*`` overrides, applied after the config and before validation."""
    for name, key, cast in (
        ("PHI_OUTPUT_DIR", "output_dir", str),
        ("PHI_VOCABULARY", "vocabulary", int),
        ("PHI_RESERVOIR_SIZE", "reservoir.size", int),
    ):
        value = os.environ.get(name)
        if value in (None, ""):
            continue
        node, parts = cfg.phi, key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = cast(value)
    # Teacher overrides: which frozen DeYi arm phi reads. The deyi keys are read by the shared
    # ``deyi.data``/``deyi.item_space`` helpers, so overriding them here is what keeps every
    # stage on the same arm. ``PHI_TASK`` is applied first because the recent override is
    # per task.
    if os.environ.get("PHI_TASK"):
        cfg.task = str(os.environ["PHI_TASK"])
    if os.environ.get("PHI_DEYI_K"):
        cfg.deyi.num_interests = int(os.environ["PHI_DEYI_K"])
    if os.environ.get("PHI_DEYI_RECENT"):
        # ``recent`` stays the task -> window *table* the rest of the project reads; the override
        # replaces this task's entry. Writing a scalar here would break every shared reader
        # (``deyi.artifacts.recent`` / ``deyi.data.load_samples``) that expects the table.
        if cfg.task not in cfg.recent:
            raise ValueError("PHI_DEYI_RECENT needs a config entry for task %r" % cfg.task)
        cfg.recent[cfg.task] = int(os.environ["PHI_DEYI_RECENT"])
    if os.environ.get("PHI_DEYI_DATASET"):
        cfg.dataset = str(os.environ["PHI_DEYI_DATASET"])


def validate(cfg) -> None:
    """Cheap checks that fail before a job burns GPU hours."""
    phi = cfg.phi
    if int(phi.vocabulary) < 2:
        raise ValueError("vocabulary (L) must be at least 2")
    if int(cfg.deyi.num_interests) < 1:
        raise ValueError("deyi.num_interests (K) must be positive")
    if int(phi.reservoir.size) < int(phi.vocabulary):
        raise ValueError("reservoir.size must be at least vocabulary (one state per token)")
    estimator = phi.estimator
    if int(estimator.head) < 1 or int(estimator.tail) < 1:
        raise ValueError("estimator.head (M) and estimator.tail (N) must be positive")
    if not 0.0 <= float(estimator.proposal_lambda) <= 1.0:
        raise ValueError("estimator.proposal_lambda must be in [0, 1]")
    if int(estimator.calibration) < 0:
        raise ValueError("estimator.calibration (S_cal) must be >= 0 (0 = skip calibration)")
    if int(estimator.get("moment_block", 1024)) < 1:
        raise ValueError("estimator.moment_block must be positive")
    if not 0.0 <= float(estimator.get("fallback_max_fraction", 0.05)) <= 1.0:
        raise ValueError("estimator.fallback_max_fraction must be in [0, 1]")
    if int(phi.codebook.iterations) < 1:
        raise ValueError("codebook.iterations must be positive")
    metric = str(phi.codebook.get("metric", "predictive"))
    if metric not in ("predictive", "euclidean", "cosine"):
        raise ValueError("codebook.metric must be predictive|euclidean|cosine, got %r" % metric)
    phi.codebook.metric = metric
    if not 0.0 <= float(phi.codebook.held_out_fraction) < 0.5:
        raise ValueError("codebook.held_out_fraction must be in [0, 0.5)")
    if str(phi.estimator.device) not in ("auto", "cpu", "cuda"):
        raise ValueError("estimator.device must be one of auto/cpu/cuda")
    if float(cfg.deyi.item_temperature) <= 0:
        raise ValueError("deyi.item_temperature (tau_p) must be positive")
    if int(phi.router.layers) < 2:
        raise ValueError("router.layers must be at least 2")
    if int(phi.router.batch_size) < 1 or int(phi.tokenize.batch_size) < 1:
        raise ValueError("batch sizes must be positive")
    if not 0.0 <= float(phi.router.validation_fraction) < 0.5:
        raise ValueError("router.validation_fraction must be in [0, 0.5)")


def load(path: str | Path = DEFAULT):
    """The DeYi config as the base plus phi's own sections, resolved and validated."""
    spec = OmegaConf.load(Path(path))
    base = str(spec.get("deyi_config", "config/model/deyi.yaml"))
    cfg = load_deyi_config(base)
    cfg.phi = OmegaConf.create(OmegaConf.to_container(spec, resolve=True))
    cfg.source = str(Path(path).resolve())
    _apply_env(cfg)
    # The table stays a table; the selected task's window is mirrored once into ``deyi.recent``,
    # the scalar every phi artifact path is keyed by (the same mirror ``deyi.validate`` makes).
    cfg.deyi.recent = int(deyi_artifacts.recent(cfg))
    OmegaConf.resolve(cfg)
    validate(cfg)
    return cfg
