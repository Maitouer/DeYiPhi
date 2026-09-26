"""Encode chronological train/test rows with the generic DeYi encoder.

    plan -> encode-train -> encode-test -> verify

Style follows the data pipeline and fit-pca stages: the job creates its own log directory
(console.log / progress.json / progress.jsonl), the artifact carries a manifest, and a
completed manifest makes the stage idempotent.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from time import monotonic, perf_counter

import numpy as np
import torch

from src.common.progress import Progress, fmt_bytes, fmt_seconds, run_directory
from src.model.deyi import artifacts
from src.model.deyi.data import collate, load_samples, projected_catalog
from src.model.deyi.model import DeYi, model_config
from src.model.deyi.runtime import device, save_pt, seed_all

STAGES = ("plan", "encode-train", "encode-test", "verify")
FORMAT = "deyi-states-v2"
CHECK_ROWS = 8192


def log_directory(stage="deyi/encode"):
    configured = os.environ.get("MODEL_LOG_DIR")
    if configured:
        path = Path(configured)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return run_directory(Path("log/model") / stage, os.environ.get("SLURM_JOB_ID"))


def manifest_path(cfg):
    return artifacts.states(cfg, "train").parent / "manifest.json"


def dataset_root(cfg):
    """The dataset that owns ``cfg.task``, declared by the pipeline's own ``tasks.json``."""
    from src.data.dataset import open_dataset

    # One schema: ``data.root`` is the pipeline root and ``tasks.<task>.dataset`` (or the CLI
    # ``--dataset`` override) names the dataset, exactly like every other stage.
    return open_dataset(str(cfg.data.root), str(cfg.task), artifacts.dataset(cfg)).root


def canonical_identity(cfg):
    """Rows per split straight from the dataset manifest (single source of truth, no reload)."""
    meta = json.loads((dataset_root(cfg) / "dataset.json").read_text(encoding="utf-8"))
    counts = {split: int(meta["tasks"][cfg.task]["splits"][split]["rows"])
              for split in ("train", "test")}
    return {"dataset": str(meta.get("name")), "created": str(meta.get("created")),
            "rows": counts}, counts


def _reusable(cfg, canonical, counts):
    """Reuse a finished encode only while its口径 (canonical identity + K/dim) still holds."""
    path = manifest_path(cfg)
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    splits = saved.get("splits", {})
    if (saved.get("format") != FORMAT
            or saved.get("canonical") != canonical
            or int(saved.get("num_interests", -1)) != int(cfg.deyi.num_interests)
            or int(saved.get("state_dim", -1)) != int(cfg.deyi.state_dim)
            or set(splits) != set(counts)):
        return None
    for split, rows in counts.items():
        if int(splits[split].get("rows", -1)) != rows or not Path(splits[split]["path"]).exists():
            return None
    return saved


def _autocast(target):
    return torch.autocast("cuda", dtype=torch.bfloat16) if target.type == "cuda" else nullcontext()


def _encode_split(cfg, model, samples, catalog, split, target, progress):
    rows = len(samples)
    shape = (rows, int(cfg.deyi.num_interests), int(cfg.deyi.state_dim))
    output_path = artifacts.states(cfg, split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending = output_path.with_name(f".{output_path.name}.{os.getpid()}.states.pending.npy")
    states = np.lib.format.open_memmap(pending, mode="w+", dtype=np.float16, shape=shape)
    mask = torch.zeros(shape[:2], dtype=torch.bool)
    source_rows = torch.empty(rows, dtype=torch.long)
    batch_size = int(cfg.deyi.encode_batch_size)
    spans = [(start, min(start + batch_size, rows)) for start in range(0, rows, batch_size)]
    workers = max(1, int(cfg.deyi.get("encode_workers", 4)))
    started = monotonic()
    gather_seconds = forward_seconds = 0.0
    finite = True
    model.eval()
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    with torch.inference_mode():
        # Global pooling already gave every row its final causal state, so the export only
        # reads history; the next batches gather while the GPU runs the current one.
        pool = ThreadPoolExecutor(max_workers=workers)
        futures, next_to_submit = {}, 1
        for batch_index, (start, stop) in enumerate(spans, 1):
            while next_to_submit <= len(spans) and len(futures) < workers:
                s, e = spans[next_to_submit - 1]
                futures[next_to_submit] = pool.submit(collate, samples[s:e], catalog,
                                                      dtype=torch.float16)
                next_to_submit += 1
            gather_started = perf_counter()
            batch = futures.pop(batch_index).result()
            gather_seconds += perf_counter() - gather_started
            forward_started = perf_counter()
            with _autocast(target):
                # Z depends on old history only; recent history is a training-time readout
                # input and never enters the exported memory.
                encoded = model.encode(batch["old_content"].to(target),
                                       batch["old_behavior"].to(target),
                                       batch["old_recency"].to(target),
                                       batch["old_mask"].to(target))
            forward_seconds += perf_counter() - forward_started
            states[start:stop] = encoded["states"].float().cpu().numpy().astype(np.float16)
            # ``DeYi.encode`` names the slot validity ``state_mask``; the export carries it as
            # ``mask`` (that is the key ``deyi.consumer.load_states`` requires).
            mask[start:stop] = encoded["state_mask"].cpu()
            source_rows[start:stop] = batch["source_row_idx"]
            if finite and not bool(torch.isfinite(encoded["states"]).all()):
                finite = False
            if batch_index % 10 == 0 or batch_index == len(spans):
                progress.tick(batch_index, len(spans),
                              "%s rows=%d/%d gather=%.0fs forward=%.0fs %s"
                              % (split, stop, rows, gather_seconds, forward_seconds,
                                 fmt_bytes(stop * shape[1] * shape[2] * 2)),
                              counters={"rows_encoded": stop})
        pool.shutdown(wait=True)
    states.flush()
    del states
    dense = torch.from_numpy(np.load(pending, mmap_mode="r+"))
    save_pt({"format": DeYi.STATE_FORMAT, "states": dense, "mask": mask,
             "source_row_idx": source_rows,
             "slot_id": torch.arange(int(cfg.deyi.num_interests)),
             "num_interests": int(cfg.deyi.num_interests),
             "state_dim": int(cfg.deyi.state_dim), "design": DeYi.DESIGN,
             "radius_alignment": DeYi.RADIUS_ALIGNMENT}, output_path)
    pending.unlink(missing_ok=True)
    seconds = monotonic() - started
    progress.line(split, "DONE rows=%d seconds=%.1f (gather %.1fs forward %.1fs) -> %s"
                  % (rows, seconds, gather_seconds, forward_seconds, output_path), force=True)
    result = {"path": str(output_path.resolve()), "rows": rows, "shape": list(shape),
              "dtype": "float16", "seconds": round(seconds, 1),
              "gather_seconds": round(gather_seconds, 1),
              "forward_seconds": round(forward_seconds, 1), "finite": finite,
              "source_row_min": int(source_rows.min()), "source_row_max": int(source_rows.max()),
              # private: used by verify, stripped before the manifest is written
              "_source_row_idx": source_rows}
    if target.type == "cuda":
        result["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated(target) / 2 ** 30, 3)
    return result


def _verify(cfg, counts, records, progress):
    """Minimum self-check: one strictly ordered row per canonical row, shapes, finiteness.

    Everything is taken from what the encode pass already produced, so verify never re-reads
    the canonical split nor copies the 1.2 GB of states back into memory.
    """
    problems = []
    for split, record in records.items():
        rows = int(record["rows"])
        index = record["_source_row_idx"]
        if len(index) != rows or rows != counts[split]:
            problems.append(f"{split}: row count mismatch")
        if rows > 1 and not bool((index[1:] > index[:-1]).all()):
            problems.append(f"{split}: source_row_idx is not one strictly ordered row per row")
        if not record.get("finite", False):
            problems.append(f"{split}: non-finite states produced during encode")
        progress.line("verify", "%s rows=%d source_rows=[%d, %d] finite=%s"
                      % (split, rows, record["source_row_min"], record["source_row_max"],
                         record["finite"]), force=True)
    if problems:
        raise ValueError("deyi encode self-check failed: %s" % "; ".join(problems))


def encode(cfg):
    progress = Progress(log_directory(), len(STAGES), interval=5.0)
    started = monotonic()
    progress.banner([
        "=" * 96,
        "DeYi encode | job=%s task=%s K=%d dim=%d"
        % (os.environ.get("SLURM_JOB_ID"), cfg.task, int(cfg.deyi.num_interests),
           int(cfg.deyi.state_dim)),
        "  checkpoint: %s" % artifacts.checkpoint(cfg),
        "  output    : %s" % artifacts.states(cfg, "train").parent,
        "  plan      : " + " -> ".join("%d.%s" % (i + 1, name) for i, name in enumerate(STAGES)),
        "=" * 96,
    ])
    progress.stage_begin(1, "plan", "checkpoint + canonical rows + reuse guard")
    checkpoint_path = artifacts.checkpoint(cfg)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"DeYi checkpoint is missing: {checkpoint_path}; run scripts/model/deyi/train.sbatch "
            f"for task {cfg.task} first")
    # Row counts come from the canonical manifest (data-pipeline selfcheck already guarantees
    # they match the split files), so plan never touches the multi-GB sample payloads.
    canonical, counts = canonical_identity(cfg)
    progress.line("plan", "dataset %s created=%s rows=%s"
                  % (canonical["dataset"], canonical["created"], canonical["rows"]), force=True)
    saved = _reusable(cfg, canonical, counts)
    if saved is not None:
        progress.line("plan", "completed encode reused (%s)" % manifest_path(cfg), force=True)
        progress.finish("encode reused rows=%s" % counts)
        return saved
    progress.stage_end("rows=%s" % counts)
    target = device()
    seed_all(int(cfg.seed))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != DeYi.CHECKPOINT_FORMAT:
        raise ValueError("incompatible generic DeYi checkpoint format")
    model = DeYi(model_config(cfg)).to(target)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    catalog = projected_catalog(cfg)
    records = {}
    for index, split in enumerate(("train", "test"), start=2):
        progress.stage_begin(index, STAGES[index - 1], "split=%s" % split)
        samples = load_samples(cfg, split, history_only=True, recent=artifacts.recent(cfg))
        records[split] = _encode_split(cfg, model, samples, catalog, split, target, progress)
        progress.stage_end("split=%s rows=%d" % (split, records[split]["rows"]))
    progress.stage_begin(4, "verify", "manifest + alignment + finiteness")
    _verify(cfg, counts, records, progress)
    public = {split: {key: value for key, value in record.items() if not key.startswith("_")}
              for split, record in records.items()}
    manifest = {"format": FORMAT, "task": cfg.task, "num_interests": int(cfg.deyi.num_interests),
                "dataset": artifacts.dataset(cfg),
                "state_dim": int(cfg.deyi.state_dim), "seed": int(cfg.seed),
                "recent": artifacts.recent(cfg), "arm": artifacts.arm_name(cfg),
                "radius_alignment": DeYi.RADIUS_ALIGNMENT,
                "state_semantics": "unit-norm predictive memory token",
                "checkpoint": str(artifacts.checkpoint(cfg)),
                "canonical": canonical,
                "item_space": str(artifacts.item_space(cfg)),
                "align": "one row per canonical source_row_idx", "splits": public,
                "seconds": round(monotonic() - started, 1)}
    path = manifest_path(cfg)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    progress.stage_end("checks 4/4")
    progress.finish("states ready (%s)" % fmt_seconds(monotonic() - started))
    return manifest
