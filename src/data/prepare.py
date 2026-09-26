"""Build ``output/data``: one shared item table plus the single- and multi-channel datasets.

    sbatch scripts/data/prepare.sbatch [--resume] [--max-rows N] [--max-shards N]

Stages, in order:

    plan        read the config, the source tables and the embedding footers; print the plan
    item-table  scan the embedding shards, assign physical rows, write pids.npy and text.npy
    splits      filter, truncate, map PIDs to rows and write <task>/<split>.bin per dataset
    manifest    write each dataset.json (catalog_rows.npy is written with the splits)
    selfcheck   invariants plus sampled parity against the released tables

Every artifact is written with ``.tmp`` + atomic rename.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from src.common.progress import Progress, fmt_bytes, run_directory

from . import item_table as item_table_stage
from . import selfcheck as selfcheck_stage
from . import source, splits as splits_stage
from .config import load_config
from .runtime import configure_runtime

STAGES = ("plan", "item-table", "splits", "manifest", "selfcheck")


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def prepare(config_path="config/data.yaml", out_override=None, resume=False,
            max_rows=None, max_shards=None, log_root=None, workers=None):
    config = load_config(config_path)
    out_root = Path(out_override or config.output)
    item_root = out_root / config.item_dir
    runtime = config.runtime
    job_id = os.environ.get("SLURM_JOB_ID")
    if os.environ.get("DATA_LOG_DIR"):
        log_dir = Path(os.environ["DATA_LOG_DIR"])
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = run_directory(Path(log_root or config.log_root) / "prepare", job_id)
    progress = Progress(log_dir, len(STAGES), job_id=job_id,
                        interval=float(runtime.get("progress_interval", 5)))
    workers = int(workers or runtime.get("workers", max(1, min(16, (os.cpu_count() or 8) - 2))))
    write_workers = int(runtime.get("write_workers", 6))
    configure_runtime(int(runtime.get("arrow_threads", 8)))
    started = time.time()

    # ------------------------------------------------------------------ stage 0
    progress.stage_begin(1, "plan", "source tables + embedding footers")
    shards = item_table_stage.shard_files(config.embeddings)
    if max_shards:
        shards = shards[:int(max_shards)]
    footers = item_table_stage.scan_footers(shards)
    wide = source.read_train(config)
    samples, rows_per_split = {}, {}
    for dataset in config.datasets:
        rows_per_split[dataset.name] = {}
        for task in dataset.tasks:
            samples[(dataset.name, task.name, "train")] = source.build_train_table(
                wide, task, config.min_primary_history, max_rows)
            samples[(dataset.name, task.name, "test")] = source.build_test_table(
                config, task, config.min_primary_history, max_rows)
            rows_per_split[dataset.name][task.name] = {
                split: int(len(samples[(dataset.name, task.name, split)]))
                for split in ("train", "test")
            }
    del wide
    requested = source.requested_pids(list(samples.values()))
    progress.banner([
        "=" * 96,
        "recommendation data  |  job=%s  workers=%d  write_workers=%d"
        % (job_id, workers, write_workers),
        "  source      : %s" % config.source,
        "  embeddings  : %s (%d shards, %d rows, %s)"
        % (config.embeddings, footers["shards"], footers["rows"], fmt_bytes(footers["bytes"])),
        "  output      : %s" % out_root,
        "  logs        : %s" % log_dir,
        "  rules       : min_primary_history=%d, drop rows without an item vector"
        % config.min_primary_history,
        "  datasets    : %s" % ", ".join(
            "%s[%s]" % (dataset.name, " ".join(
                "%s(%s)" % (task.name, "+".join(task.channels)) for task in dataset.tasks))
            for dataset in config.datasets),
        "  plan        : " + " -> ".join(
            "%d.%s" % (index + 1, name) for index, name in enumerate(STAGES)),
        "=" * 96,
    ])
    _atomic_json(log_dir / "plan.json", {
        "config": config_path, "output": str(out_root), "resume": bool(resume),
        "max_rows": max_rows, "max_shards": max_shards, "footers": footers,
        "requested_items": int(len(requested)), "rows_per_split": rows_per_split,
    })
    kept_rows = sum(count for dataset_rows in rows_per_split.values()
                    for task_rows in dataset_rows.values()
                    for count in task_rows.values())
    progress.stage_end("items=%d rows=%d" % (len(requested), kept_rows))

    # ------------------------------------------------------------------ stage 1
    progress.stage_begin(2, "item-table", "scan %d embedding shards" % len(shards))
    item_payload = item_table_stage.build(config, shards, requested, item_root,
                                          log_dir / "item_index", workers, progress,
                                          resume=resume)
    progress.stage_end("stored=%d missing=%d" % (item_payload["stored_items"],
                                                 item_payload["missing_items"]))

    # ------------------------------------------------------------------ stage 2
    split_stats = []
    for dataset in config.datasets:
        dataset_root = out_root / dataset.name
        dataset_root.mkdir(parents=True, exist_ok=True)
        tags = {"dataset": dataset.name, "min_primary_history": config.min_primary_history}
        progress.stage_begin(3, "splits", dataset.name)
        stats = splits_stage.write_dataset(dataset, samples, tags,
                                           item_root / "pids.npy", dataset_root,
                                           write_workers, progress)
        catalog, task_catalogs = splits_stage.write_catalogs(dataset_root, stats)
        for entry in stats:
            entry["dataset"] = dataset.name
            entry["root"] = str(dataset_root)
        split_stats.extend(stats)
        _atomic_json(dataset_root / "stats.json",
                     {"dataset": dataset.name, "catalog_rows": int(len(catalog)),
                      "task_catalog_rows": task_catalogs, "splits": stats})
        progress.line("splits", "%s catalog=%d rows" % (dataset.name, len(catalog)), force=True)
        progress.stage_end("%s splits=%d" % (dataset.name, len(stats)))

    # ------------------------------------------------------------------ stage 3
    progress.stage_begin(4, "manifest", "dataset.json")
    # Which dataset a bare task name means. Declared in config/data.yaml under ``owners``;
    # readers that want the other dataset name it explicitly.
    _atomic_json(out_root / "tasks.json", dict(config.task_owners))
    by_key = {(entry["dataset"], entry["task"], entry["split"]): entry
              for entry in split_stats}
    for dataset in config.datasets:
        dataset_root = out_root / dataset.name
        tasks = {}
        for task in dataset.tasks:
            tasks[task.name] = {
                "channels": list(task.channels),
                "primary": task.primary,
                "target_limit": int(task.target_limit),
                "behaviors": list(task.behaviors),
                "catalog": "%s/catalog_rows.npy" % task.name,
                "histories": {history.channel: {"field": history.field, "limit": history.limit}
                              for history in task.histories},
                "splits": {
                    split: {
                        "file": "%s/%s.bin" % (task.name, split),
                        "rows": int(by_key[(dataset.name, task.name, split)]["rows"]),
                        "widths": {name: int(width) for name, width
                                   in by_key[(dataset.name, task.name, split)]["widths"].items()},
                        "removed_missing_item_vector":
                            int(by_key[(dataset.name, task.name, split)]
                                ["removed_missing_item_vector"]),
                    } for split in ("train", "test")
                },
            }
        _atomic_json(dataset_root / "dataset.json", {
            "kind": "recommendation_dataset",
            "name": dataset.name,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source": str(config.source),
            "item_table": str(Path(config.item_dir) / "item_table.json"),
            "catalog": "catalog_rows.npy",
            "padding": {"item_row": 0, "behavior_unobserved": -1},
            "rules": {"min_primary_history": config.min_primary_history,
                      "drop_rows_without_item_vector": True},
            "tasks": tasks,
        })
    progress.stage_end("datasets=%d" % len(config.datasets))

    # ------------------------------------------------------------------ stage 4
    progress.stage_begin(5, "selfcheck", "invariants + sampled parity")
    from .dataset import open_store

    store = open_store(out_root, config.item_dir)
    checks = selfcheck_stage.run_checks(store, split_stats, samples, progress)
    _atomic_json(log_dir / "checks.json", checks)
    for dataset in config.datasets:
        manifest_path = out_root / dataset.name / "dataset.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["checks"] = {"passed": checks["passed"], "failed": checks["failed"],
                              "see": str(log_dir / "checks.json")}
        _atomic_json(manifest_path, manifest)
    progress.stage_end("checks=%d/%d" % (checks["passed"],
                                         checks["passed"] + checks["failed"]))

    total_files = 3 + sum(3 + 2 * len(dataset.tasks) for dataset in config.datasets)
    progress.finish("files=%d" % total_files, total_seconds=time.time() - started)
    return {"checks": checks, "log_dir": str(log_dir), "seconds": time.time() - started,
            "rows_per_split": rows_per_split}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/data.yaml")
    parser.add_argument("--output", help="override the output directory (smoke runs)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-rows", type=int, help="smoke: keep only the first N rows")
    parser.add_argument("--max-shards", type=int, help="smoke: scan only the first N shards")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--log-root")
    args = parser.parse_args()
    prepare(args.config, args.output, args.resume, args.max_rows, args.max_shards,
            args.log_root, args.workers)


if __name__ == "__main__":
    main()
