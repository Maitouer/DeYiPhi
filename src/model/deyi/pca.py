"""Build the shared item semantic space (``00-Reconstruct/00_item_semantic_space.md``).

    Qwen3 text  ->  L2 normalize  ->  centered PCA  ->  p(i) = U_d^T (q_bar(i) - mu)

No whitening, no trainable adapter. ``embeddings.npy`` stores the Euclidean coordinate
``p(i)`` used for residual quantization; consumers that need cosine-style matching apply
``e(i) = normalize(p(i))`` themselves.

Stages: plan -> covariance -> fit -> project -> manifest.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.common.progress import Progress, fmt_bytes, fmt_seconds, run_directory
from src.data.dataset import open_store
from src.model.deyi import artifacts, covariance, item_space, projection
from src.model.deyi.runtime import save_pt, seed_all

STAGES = ("plan", "covariance", "fit", "project", "manifest")
# One format string for the item-space manifest (see artifacts.ITEM_SPACE_FORMAT).
PCA_FORMAT = artifacts.ITEM_SPACE_FORMAT


def _log_directory(cfg, job_id):
    if os.environ.get("DEYI_LOG_DIR"):
        path = Path(os.environ["DEYI_LOG_DIR"])
        path.mkdir(parents=True, exist_ok=True)
        return path
    return run_directory(Path(str(cfg.log_root)) / "fit-pca", job_id)


def _write_json(path: Path, payload) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def fit_pca(cfg):
    started = time.time()
    job_id = os.environ.get("SLURM_JOB_ID")
    log_dir = _log_directory(cfg, job_id)
    runtime = cfg.get("runtime") or {}
    workers = int(runtime.get("workers", 6))
    progress = Progress(log_dir, len(STAGES), job_id=job_id,
                        interval=float(runtime.get("progress_interval", 5)))
    seed_all(int(cfg.seed))

    progress.stage_begin(1, "plan", "item table + reuse guard")
    space = open_store(str(cfg.data.root)).item_space()
    problems = space.validate()
    if problems:
        raise ValueError("item table failed validation: %s" % "; ".join(problems))
    tag = item_space.canonical_tag(cfg, space)
    dim = int(cfg.pca.dim)
    samples = int(cfg.pca.get("samples", 0))
    progress.banner([
        "=" * 96,
        "DeYi item semantic space  |  job=%s  workers=%d" % (job_id, workers),
        "  item table  : %s  created=%s  rows=%d  scope=%s"
        % (cfg.data.root, tag["created"], tag["rows"], tag["scope"]),
        "  pca         : dim=%d  fit=%s  block_rows=%d  project_batch=%d"
        % (dim, "catalog(exact)" if samples <= 0 else "sampled %d" % samples,
           int(cfg.pca.get("block_rows", 16384)), int(cfg.pca.project_batch_size)),
        "  pipeline    : Qwen3 -> L2 normalize -> centered PCA (no whitening)",
        "  output      : %s" % artifacts.item_space(cfg),
        "  logs        : %s" % log_dir,
        "  plan        : " + " -> ".join("%d.%s" % (i + 1, name) for i, name in enumerate(STAGES)),
        "=" * 96,
    ])
    saved = item_space.completed(cfg, tag)
    if saved is not None:
        progress.line("plan", "reuse=hit (%s)" % artifacts.item_space_manifest(cfg), force=True)
        progress.finish("item space reused")
        return saved
    progress.line("plan", "reuse=miss catalog=%d" % len(space.catalog), force=True)
    progress.stage_end("rows=%d catalog=%d" % (tag["rows"], len(space.catalog)))

    progress.stage_begin(2, "covariance", "normalize + mean/Gram")
    blocks = covariance.row_blocks(cfg, np.asarray(space.catalog))
    mean, matrix, stats = covariance.accumulate(cfg, space, blocks, progress)
    if stats["zero_norm_rows"]:
        raise ValueError("%d catalog rows have a zero-norm Qwen3 vector"
                         % stats["zero_norm_rows"])
    progress.stage_end("rows=%d exact=%s" % (stats["rows"], bool(stats["exact"])))

    progress.stage_begin(3, "fit", "eigh(4096) -> top %d" % dim)
    fit_started = time.time()
    eigenvalues, components = covariance.fit(matrix, dim)
    total_variance = float(torch.trace(matrix))
    ratio = float(eigenvalues.sum()) / total_variance if total_variance > 0 else 0.0
    stats["explained_variance_ratio"] = ratio
    stats["fit_seconds"] = round(time.time() - fit_started, 1)
    save_pt({"format": PCA_FORMAT, "dim": dim,
             "mean": mean.detach().cpu(),
             "components": components.detach().cpu(),
             "eigenvalues": eigenvalues.detach().cpu(),
             "normalize_input": True, "whitening": False}, artifacts.pca(cfg))
    progress.stage_end("explained_variance=%.4f (%.1fs)" % (ratio, stats["fit_seconds"]))

    progress.stage_begin(4, "project", "rows=%d dim=%d" % (len(space.items), dim))
    projection_stats = projection.project_all(
        cfg, space, mean.detach().cpu().numpy(), components.detach().cpu().numpy(),
        artifacts.embeddings(cfg), workers, progress)
    stats["project"] = projection_stats
    progress.stage_end("rows=%d %s" % (projection_stats["rows"],
                                       fmt_bytes(projection_stats["rows"] * dim * 2)))

    progress.stage_begin(5, "manifest", "manifest + stats + selfcheck")
    checks = item_space.selfcheck(cfg, space, mean.detach().cpu().numpy(),
                                  components.detach().cpu().numpy(), artifacts.embeddings(cfg),
                                  stats, progress, log_dir / "checks.json")
    payload = {
        "format": PCA_FORMAT,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rows": int(len(space.items)),
        "dim": dim,
        "embedding_dtype": "float16",
        "row0": "sentinel row, never a real item",
        # Frozen definition of the two views of the same space. Consumers read this instead
        # of assuming what ``embeddings.npy`` already contains.
        "semantics": {
            "input": "pca.mean is subtracted from the L2-normalized Qwen3 text vector",
            "embeddings.npy": "p(i) = U_d^T (q_bar(i) - mu); Euclidean coordinate, NOT normalized",
            "consumer_note": "apply e(i) = p(i)/||p(i)||_2 for cosine-style matching",
            "whitening": False,
        },
        "canonical": tag,
        "stats": stats,
        "checks": {"passed": checks["passed"], "failed": checks["failed"],
                   "see": str(log_dir / "checks.json")},
        "seconds": round(time.time() - started, 1),
    }
    _write_json(artifacts.item_space_stats(cfg), payload)
    item_space.write_manifest(cfg, payload)
    progress.stage_end("checks=%d/%d" % (checks["passed"],
                                         checks["passed"] + checks["failed"]))

    progress.finish("item space ready (4 files) total %s"
                    % fmt_seconds(time.time() - started))
    return payload
