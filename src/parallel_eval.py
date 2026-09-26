"""Single-node evaluation replicas, preserving the original per-GPU batches."""

from __future__ import annotations

import importlib
import json
import math
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path


def worker_count():
    count = int(os.environ.get("DEYIPHI_EVAL_GPUS", "1"))
    if count < 1:
        raise ValueError("DEYIPHI_EVAL_GPUS must be positive")
    return count


def is_worker():
    return "DEYIPHI_EVAL_RANK" in os.environ


def batch_starts(rows, batch_size):
    if batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    rank = int(os.environ.get("DEYIPHI_EVAL_RANK", "0"))
    world = worker_count() if is_worker() else 1
    if not 0 <= rank < world:
        raise ValueError("invalid evaluation rank")
    return list(range(rank * batch_size, rows, world * batch_size))


def local_rows(rows, batch_size):
    return sum(min(batch_size, rows - start) for start in batch_starts(rows, batch_size))


def shard_mapping(data, batch_size):
    items = list(data.items())
    return dict(
        item
        for start in batch_starts(len(items), batch_size)
        for item in items[start : start + batch_size]
    )


def output_dir(default):
    return Path(os.environ["DEYIPHI_EVAL_OUTPUT"]) if is_worker() else Path(default)


def validate_devices(count):
    import torch

    if torch.cuda.device_count() != count:
        raise ValueError(f"eval requested {count} GPUs, visible={torch.cuda.device_count()}")
    if any(torch.cuda.get_device_properties(i).total_memory < 70 * 2**30 for i in range(count)):
        raise ValueError("formal evaluation requires complete H100 GPUs")


def _worker(module, function, args, kwargs, log):
    _cache = configure_compile_cache()
    with open(log, "a", buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        getattr(importlib.import_module(module), function)(*args, **kwargs)


def launch(module, args, output, *, kwargs=None):
    """Return successful shard directories in parent, None in single-card/worker mode.

    Each attempt is isolated. Failed shards are retained for diagnosis; no partial
    aggregate is published and no training/checkpoint files are changed.
    """
    if is_worker() or worker_count() == 1:
        return None
    count = worker_count()
    validate_devices(count)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(count)]
    if len(devices) != count:
        raise ValueError("CUDA_VISIBLE_DEVICES does not match evaluation replicas")
    attempt = Path(output) / "shards" / uuid.uuid4().hex[:12]
    attempt.mkdir(parents=True)
    # Worker logs stay under the stage log directory (``MODEL_LOG_DIR`` is what
    # scripts/model/common_logging.sh exports); never fall back to a project-level output/log.
    logs = (
        Path(
            os.environ.get("DEYIPHI_EVAL_LOG_DIR")
            or os.environ.get("MODEL_LOG_DIR")
            or os.environ.get("EXPERIMENT_LOG_DIR")
            or "log/model/tiger/eval/shard-workers"
        )
        / attempt.name
    )
    logs.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    processes, directories = [], []
    keys = ("CUDA_VISIBLE_DEVICES", "DEYIPHI_EVAL_RANK", "DEYIPHI_EVAL_OUTPUT")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for rank, device in enumerate(devices):
            directory = attempt / f"rank{rank}"
            directory.mkdir()
            directories.append(directory)
            os.environ.update(
                CUDA_VISIBLE_DEVICES=device,
                DEYIPHI_EVAL_RANK=str(rank),
                DEYIPHI_EVAL_OUTPUT=str(directory),
            )
            process = context.Process(
                target=_worker,
                args=(module, "evaluate", args, kwargs or {}, str(logs / f"rank{rank}.log")),
            )
            process.start()
            processes.append(process)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        while any(p.is_alive() for p in processes):
            failed = [
                (i, p.exitcode) for i, p in enumerate(processes) if p.exitcode not in (None, 0)
            ]
            if failed:
                raise RuntimeError(f"evaluation workers failed: {failed}; logs={logs}")
            time.sleep(0.5)
        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError(f"evaluation worker failed; logs={logs}")
        return directories
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join()


def merge_jsonl(shards, relative, destination, expected):
    """Validate exact source coverage, then publish in original dataset order."""
    expected = [int(x) for x in expected]
    if len(set(expected)) != len(expected):
        raise ValueError("evaluation source IDs are not unique")
    records = {}
    for shard in shards:
        with (Path(shard) / relative).open() as stream:
            for line in stream:
                row = json.loads(line)
                source = int(row["source_row_idx"])
                if source in records:
                    raise ValueError(f"duplicate evaluation source_row_idx={source}")
                records[source] = row
    if set(records) != set(expected):
        raise ValueError("evaluation shards have missing or unexpected source rows")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".tmp")
    with temp.open("w") as stream:
        for source in expected:
            stream.write(json.dumps(records[source], separators=(",", ":")) + "\n")
    os.replace(temp, destination)
    return records


def merge_metrics(
    shards,
    output,
    expected,
    *,
    rows_file="row_metrics.jsonl",
    generations="test_generated.jsonl",
    ks,
):
    expected = [int(x) for x in expected]
    names = [
        f"{prefix}{metric}@{k}"
        for k in ks
        for prefix in ("", "pid_")
        for metric in ("pass", "recall", "ndcg")
    ]
    # Validate all rows before writing an authoritative aggregate.
    rows = merge_jsonl(shards, rows_file, Path(output) / rows_file, expected)
    if generations:
        merge_jsonl(shards, generations, Path(output) / generations, expected)
    aggregate = {}
    for name in names:
        values = [float(rows[source][name]) for source in expected]
        if not values or not all(math.isfinite(x) for x in values):
            raise ValueError(f"empty/non-finite evaluation metric {name}")
        aggregate[name] = math.fsum(values) / len(values)
    summaries = [json.loads((Path(shard) / "metrics.json").read_text()) for shard in shards]
    checkpoints = {summary["checkpoint"] for summary in summaries}
    if len(checkpoints) != 1:
        raise ValueError("evaluation shards used different checkpoints")
    result = {**summaries[0], **aggregate, "rows": len(expected), "eval_gpus": len(shards)}
    target = Path(output) / "metrics.json"
    temp = target.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temp, target)
    return result


def merge_generated(shards, relative, destination, expected):
    """Merge the official benchmark samples without averaging shard summaries."""
    records, sources = {}, set()
    template = None
    for shard in shards:
        payload = json.loads((Path(shard) / relative).read_text())
        template = payload if template is None else template
        for key, sample in payload["samples"].items():
            metadata = sample["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            source = int(metadata["source_row_idx"])
            if key in records or source in sources:
                raise ValueError("duplicate generated sample in evaluation shards")
            sources.add(source)
            records[key] = sample
    if sources != set(map(int, expected)) or len(sources) != len(expected):
        raise ValueError("generated evaluation shards have incomplete coverage")
    template["samples"] = records
    # Timing and per-shard summary fields must not masquerade as global results.
    payload = {
        **{k: template[k] for k in ("model_name", "task_name", "split") if k in template},
        "samples": records,
        "eval_gpus": len(shards),
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload) + "\n")
    os.replace(temp, destination)


def configure_compile_cache():
    # Replica compilation must not race on the shared NFS kernel metadata.
    import atexit
    import tempfile
    cache = tempfile.TemporaryDirectory(prefix="deyiphi-eval-compile-")
    atexit.register(cache.cleanup)
    for variable, folder in (("TRITON_CACHE_DIR", "triton"),
                             ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                             ("VLLM_CACHE_ROOT", "vllm")):
        os.environ[variable] = str(Path(cache.name) / folder)
    return cache
