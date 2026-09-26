"""CLI for the DeYi chain.

Every knob that changes what is trained or how it is scored is a flag here, so an experiment
arm can be launched without editing the config:

    python -m src.model.deyi.cli train --task ad --num-interests 4 --recent 32 \
        --item-temperature 0.15 --lambda-user 0.1 --lambda-div 0.01 --epochs 20

``recent`` additionally enters the output path (``k4_r32``), because two arms that differ only
in ``recent`` train different models and must not share an output directory.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

from src.model.deyi.config import load_config, validate
from src.model.deyi.encode import encode
from src.model.deyi.pca import fit_pca
from src.model.deyi.train import train

STAGES: dict[str, Callable] = {"fit-pca": fit_pca, "train": train, "encode": encode}

# flag -> config path. One table drives the flags, the defaults and the run snapshot.
NUMERIC = {
    "num-interests": ("deyi.num_interests", int),
    "history-layers": ("deyi.history_layers", int),
    "interest-layers": ("deyi.interest_layers", int),
    "heads": ("deyi.heads", int),
    "ffn": ("deyi.ffn", int),
    "dropout": ("deyi.dropout", float),
    "temporal-buckets": ("deyi.temporal_buckets", int),
    "route-dim": ("deyi.route_dim", int),
    "route-temperature": ("deyi.route_temperature", float),
    "item-temperature": ("deyi.item_temperature", float),
    "lambda-user": ("deyi.lambda_user", float),
    "lambda-div": ("deyi.lambda_div", float),
    "negatives": ("deyi.negatives", int),
    "device-candidate-chunk": ("deyi.device_candidate_chunk", int),
    "user-negatives": ("deyi.user_negatives", int),
    "epochs": ("deyi.epochs", int),
    "batch-size": ("deyi.batch_size", int),
    "micro-batch-size": ("deyi.micro_batch_size", int),
    "learning-rate": ("deyi.learning_rate", float),
    "weight-decay": ("deyi.weight_decay", float),
    "validation-fraction": ("deyi.validation_fraction", float),
    "min-epochs": ("deyi.min_epochs", int),
    "early-stop-patience": ("deyi.early_stop_patience", int),
    "pca-dim": ("pca.dim", int),
    "pca-samples": ("pca.samples", int),
    "block-rows": ("pca.block_rows", int),
    "project-batch-size": ("pca.project_batch_size", int),
    "cluster-temperature": ("deyi.cluster_temperature", float),
}


def _set(cfg, dotted: str, value) -> None:
    node = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        if part not in node or node[part] is None:
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=tuple(STAGES))
    parser.add_argument("--config", default="config/model/deyi.yaml")
    parser.add_argument("--task", choices=("short_video", "ad", "product"))
    parser.add_argument("--dataset", help="override tasks.<task>.dataset")
    # ``recent`` and ``max_history`` are per-task tables, so a flag targets the *selected* task
    # only; ``recent`` is also the r<R> of the arm path (k<K>_r<R>), which is why the launchers
    # pass it explicitly instead of relying on the config default.
    parser.add_argument("--recent", type=int,
                        help="old/recent window of the selected task (also the arm's r<R>)")
    parser.add_argument("--max-history", type=int,
                        help="history cap of the selected task")
    # Which H_old -> M mapping this arm instantiates (04_baseline.md): the production encoder or
    # one of the learned controlled baselines. It also picks the output root, so an arm can never
    # overwrite the DeYi one.
    parser.add_argument("--method", choices=("deyi", "chronicle", "hicogen"),
                        help="memory method; writes under output/model/compress/<method>")
    parser.add_argument("--output-root",
                        help="override deyi.output_root (defaults to output/model/compress/<method>)")
    for flag, (path, cast) in NUMERIC.items():
        parser.add_argument("--" + flag, type=cast, dest=flag.replace("-", "_"),
                            help="override %s" % path)
    parser.add_argument("--resume", action="store_true",
                        help="continue a chunked training run from its checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.task:
        cfg.task = args.task
    if args.dataset:
        cfg.dataset = args.dataset
    if args.max_history is not None:
        cfg.max_history[cfg.task] = int(args.max_history)
    if args.recent is not None:
        # ``recent`` is per task, so a CLI override targets the selected task only.
        cfg.recent[cfg.task] = int(args.recent)
    if args.method:
        cfg.deyi.method = args.method
        # One arm per method: the three baselines and DeYi never share an output directory.
        cfg.deyi.output_root = ("output/model/compress/%s" % args.method
                                if args.method != "deyi" else str(cfg.deyi.output_root))
    if args.output_root:
        cfg.deyi.output_root = args.output_root
    for flag, (path, _cast) in NUMERIC.items():
        value = getattr(args, flag.replace("-", "_"))
        if value is not None:
            _set(cfg, path, value)
    validate(cfg)
    if args.stage == "train":
        result = train(cfg, resume=args.resume)
    else:
        result = STAGES[args.stage](cfg)
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
