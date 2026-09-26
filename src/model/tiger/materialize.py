"""One-shot materialisation of the Tiger training tensors.

    plan -> scan -> write -> verify

Reads the canonical data through ``TigerDataset`` exactly like training does, then writes one
self-contained tensor file per (task, mode, split) so the training loop never parses history
again (the reference DeYiPhi-v1 Tiger pipeline does the same with ``sequences_*.pt``).

    <project>/output/model/tiger/data/<task>/<mode>/<split>.pt
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from src.common.progress import Progress, fmt_seconds, run_directory
from . import load_config, parser
from .data import TigerDataset

STAGES = ("plan", "scan", "write", "verify")
CHUNK = 4096


def log_directory(stage="tiger/materialize"):
    configured = os.environ.get("MODEL_LOG_DIR")
    if configured:
        path = Path(configured)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return run_directory(Path("log/model") / stage, os.environ.get("SLURM_JOB_ID"))


def destination(cfg, split):
    return Path("output/model/tiger/data") / cfg.dataset / cfg.task / cfg.mode / f"{split}.pt"


def materialize(cfg, split, progress):
    data = TigerDataset(cfg, split)
    rows = len(data)
    width = int(max(int(value) for value in data.lengths))
    targets = int(data.samples.target.shape[1])
    progress.line("plan", "rows=%d width=%d targets=%d split=%s"
                  % (rows, width, targets, split), force=True)
    history = torch.zeros((rows, width, 4), dtype=torch.int16)
    lengths = torch.zeros(rows, dtype=torch.int32)
    target_digits = torch.zeros((rows, targets, 4), dtype=torch.int16)
    target_mask = torch.zeros((rows, targets), dtype=torch.bool)
    source = torch.from_numpy(np.asarray(data.samples.source_rows, dtype=np.int64).copy())
    positives = 0
    for start in range(0, rows, CHUNK):
        stop = min(start + CHUNK, rows)
        selection = list(range(start, stop))
        batch = data.batch(selection, torch.device("cpu"), training=True)
        block = batch["history_raw"].to(torch.int16)
        history[start:stop, :block.shape[1]] = block
        lengths[start:stop] = batch["history_mask"].sum(1).to(torch.int32)
        target_rows = np.asarray(data.samples.target[start:stop], dtype=np.int64)
        target_len = np.asarray(data.samples.target_lengths[start:stop], dtype=np.int64)
        mask = np.arange(targets)[None, :] < target_len[:, None]
        codes = np.zeros((stop - start, targets, 4), dtype=np.int16)
        if mask.any():
            codes[mask] = np.asarray(data.codes[target_rows[mask]], dtype=np.int16)
        target_digits[start:stop] = torch.from_numpy(codes)
        target_mask[start:stop] = torch.from_numpy(mask)
        positives += int(mask.sum())
        progress.tick(stop, rows, "rows=%d/%d positives=%d" % (stop, rows, positives),
                      counters={"rows_written": stop})
    path = destination(cfg, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Record the channel order the sequence was built with: the cache is keyed by
    # (task, mode, split) and only checks width/rows, so an order change is otherwise
    # invisible to a reader (see the 2026-09-23 ad incident).
    history_order = ",".join(str(name) for name in data.samples.channels)
    payload = {"format": "tiger-sequences-v1", "task": cfg.task, "mode": cfg.mode,
               "split": split, "codebook_width": int(cfg.codebook.width),
               "history_order": history_order,
               "rows": rows, "history_width": width, "target_width": targets,
               "history_digits": history, "history_lengths": lengths,
               "target_digits": target_digits, "target_mask": target_mask,
               "source_row_idx": source}
    pending = path.with_suffix(".pending.pt")
    torch.save(payload, pending)
    pending.replace(path)
    progress.line("write", "%s (%.1f MiB)" % (path, path.stat().st_size / 2**20), force=True)
    check = torch.load(path, map_location="cpu", weights_only=False)
    problems = []
    if int(check["history_digits"].shape[0]) != rows:
        problems.append("row count mismatch")
    if int(check["history_lengths"].max()) > width:
        problems.append("history length exceeds width")
    if bool((check["target_mask"].sum(1) <= 0).any()):
        problems.append("row without positive target")
    if len(problems):
        raise ValueError("materialised sequences failed: %s" % "; ".join(problems))
    progress.line("verify", "history=%s targets=%d" % (tuple(check["history_digits"].shape),
                                                       int(check["target_mask"].sum())), force=True)
    return {"path": str(path), "rows": rows, "positives": positives,
            "bytes": int(path.stat().st_size)}


def main():
    args = parser().parse_args()
    cfg = load_config(args.config, args.task, args.mode, args.run_dir)
    progress = Progress(log_directory(), len(STAGES), interval=5.0)
    started = monotonic()
    progress.banner([
        "=" * 96,
        "Tiger sequences | job=%s task=%s mode=%s codebook=%s" % (
            os.environ.get("SLURM_JOB_ID"), cfg.task, cfg.mode, cfg.codebook.width),
        "  data    : %s   codes: %s" % (cfg.data_dir, cfg.prepared_dir),
        "  output  : output/model/tiger/data/%s/%s/%s"
        % (cfg.dataset, cfg.task, cfg.mode),
        "  plan    : " + " -> ".join("%d.%s" % (i + 1, name) for i, name in enumerate(STAGES)),
        "=" * 96,
    ])
    progress.stage_begin(1, "plan", "open canonical + metadata")
    records = {}
    progress.stage_end("task=%s mode=%s" % (cfg.task, cfg.mode))
    for index, split in enumerate(("train", "test"), start=2):
        progress.stage_begin(index, "scan" if index == 2 else "write", "split=%s" % split)
        records[split] = materialize(cfg, split, progress)
        progress.stage_end("split=%s rows=%d" % (split, records[split]["rows"]))
    progress.stage_begin(4, "verify", "publish index")
    index_path = Path("output/model/tiger/data") / cfg.dataset / cfg.task / cfg.mode / "index.json"
    index_path.write_text(json.dumps({"task": cfg.task, "mode": cfg.mode,
                                      "codebook_width": int(cfg.codebook.width),
                                      "splits": records}, indent=2) + "\n")
    progress.stage_end("index=%s" % index_path)
    progress.finish("sequences ready (%s)" % fmt_seconds(monotonic() - started))


if __name__ == "__main__":
    main()
