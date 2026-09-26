"""CLI for the phi chain.

    python -m src.model.phi.cli plan      --task product
    python -m src.model.phi.cli reservoir --task product
    python -m src.model.phi.cli estimate  --task product
    python -m src.model.phi.cli codebook  --task product --vocabulary 256
    python -m src.model.phi.cli router    --task product
    python -m src.model.phi.cli tokenize  --task product --split train
    python -m src.model.phi.cli route     --task product --split train
    python -m src.model.phi.cli audit     --task product

Every knob that changes the vocabulary is a flag here, so an arm can be searched without editing
the config; the environment (``PHI_*``) is applied first and a flag always wins.
"""

from __future__ import annotations

import argparse
import json

from . import config as phi_config
from . import pipeline

NUMERIC = {
    "vocabulary": ("phi.vocabulary", int),
    "reservoir": ("phi.reservoir.size", int),
    "head": ("phi.estimator.head", int),
    "tail": ("phi.estimator.tail", int),
    "calibration": ("phi.estimator.calibration", int),
    "iterations": ("phi.codebook.iterations", int),
    "router-epochs": ("phi.router.epochs", int),
    "router-hidden": ("phi.router.hidden", int),
}


def _set(cfg, dotted: str, value) -> None:
    node = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=pipeline.STAGES)
    parser.add_argument("--config", default=phi_config.DEFAULT)
    parser.add_argument("--task", choices=("short_video", "ad", "product"))
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        help="override phi.estimator.device")
    # Analysis II (05_analysis_design.md §11-12): one clustering framework, three distortion
    # criteria. Only the state -> token partition changes; the reservoir and the downstream
    # interface stay identical.
    parser.add_argument("--metric", choices=("predictive", "euclidean", "cosine"),
                        help="override phi.codebook.metric")
    for flag, (path, cast) in NUMERIC.items():
        parser.add_argument("--" + flag, type=cast, dest=flag.replace("-", "_"),
                            help="override %s" % path)
    args = parser.parse_args()

    cfg = phi_config.load(args.config)
    if args.task:
        cfg.task = args.task
    if args.device:
        cfg.phi.estimator.device = args.device
    if args.metric:
        cfg.phi.codebook.metric = args.metric
    for flag, (path, _cast) in NUMERIC.items():
        value = getattr(args, flag.replace("-", "_"))
        if value is not None:
            _set(cfg, path, value)
    phi_config.validate(cfg)
    result = pipeline.run(cfg, args.stage, args.split)
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()

