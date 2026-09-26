"""Load and validate the DeYi configuration (see ``00-Reconstruct/01_deyi_encoder.md``).

Validation runs after every CLI override, so a flag value is checked exactly like a config
value. ``recent`` is stored once at the top level and mirrored into ``deyi.recent`` so the
model code and the output path can never disagree.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

REQUIRED = ("recent", "max_history", "deyi", "pca", "tasks", "data")


def validate(cfg) -> None:
    if cfg.task not in cfg.tasks:
        raise ValueError(f"unknown DeYi task: {cfg.task}")
    if int(cfg.deyi.num_interests) < 1:
        raise ValueError("num_interests must be positive")
    if int(cfg.pca.dim) != int(cfg.deyi.state_dim):
        raise ValueError(
            "pca.dim (%d) must equal deyi.state_dim (%d): the encoder consumes the PCA "
            "coordinates directly" % (int(cfg.pca.dim), int(cfg.deyi.state_dim))
        )
    if cfg.task not in cfg.recent:
        raise ValueError("recent must map task -> window; missing entry for %r" % cfg.task)
    recent = int(cfg.recent[cfg.task])
    if recent < 1:
        raise ValueError("recent must be positive")
    cfg.deyi.recent = recent
    if cfg.task not in cfg.max_history:
        raise ValueError("max_history must map task -> cap; missing entry for %r" % cfg.task)
    max_history = int(cfg.max_history[cfg.task])
    if max_history < recent:
        raise ValueError("max_history must be at least recent")
    cfg.deyi.max_history = max_history
    if int(cfg.deyi.history_layers) < 1 or int(cfg.deyi.interest_layers) < 1:
        raise ValueError("history_layers and interest_layers must be positive")
    if int(cfg.deyi.heads) < 1 or int(cfg.deyi.ffn) < 1:
        raise ValueError("heads and ffn must be positive")
    if int(cfg.deyi.temporal_buckets) < 1:
        raise ValueError("temporal_buckets must be positive")
    for name in ("item_temperature", "route_temperature"):
        if float(getattr(cfg.deyi, name)) <= 0:
            raise ValueError("%s must be positive" % name)
    for name in ("lambda_user", "lambda_div"):
        if float(getattr(cfg.deyi, name)) < 0:
            raise ValueError("%s must be non-negative" % name)
    if int(cfg.deyi.negatives) < 1:
        raise ValueError("negatives must be positive")
    if int(cfg.deyi.user_negatives) < 0:
        raise ValueError("user_negatives must be non-negative")
    if not 0.0 <= float(cfg.deyi.validation_fraction) < 0.5:
        raise ValueError("validation_fraction must be in [0, 0.5)")
    if int(cfg.deyi.min_epochs) < 1:
        raise ValueError("min_epochs must be positive")
    if int(cfg.deyi.early_stop_patience) < 1:
        raise ValueError("early_stop_patience must be positive")
    if int(cfg.pca.samples) < 0:
        raise ValueError("pca.samples must be >= 0 (0 = fit on the whole catalog)")
    # The channel list is owned by the built dataset, not duplicated here; what this config
    # must say is which dataset the task is read from.
    method = str(cfg.deyi.get("method", "deyi"))
    if method not in ("deyi", "chronicle", "hicogen"):
        raise ValueError("deyi.method must be one of deyi|chronicle|hicogen, got %r" % method)
    cfg.deyi.method = method
    if method == "hicogen" and float(cfg.deyi.get("cluster_temperature", 1.0)) <= 0:
        raise ValueError("deyi.cluster_temperature must be positive")
    spec = cfg.tasks[cfg.task]
    if not spec.get("dataset"):
        raise ValueError(f"tasks.{cfg.task}.dataset must name the dataset that owns this task")


def load_config(path: str | Path):
    cfg = OmegaConf.load(Path(path))
    cfg.source = str(Path(path).resolve())
    OmegaConf.resolve(cfg)
    for name in REQUIRED:
        if name not in cfg:
            raise ValueError("config %s is missing the %r section" % (path, name))
    validate(cfg)
    return cfg
