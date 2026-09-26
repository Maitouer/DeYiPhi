"""Pooling memory: one continuous summary of the old history.

``docs/v1_algorithm/06_analysis_baseline.md`` §2 (Block A). Deterministic, training-free:

    H_old --(split into K equal time segments, mean of each segment's unit item vectors)--> M

K states per user, no learned parameters and no predictive supervision: the control that asks
whether a *generic summary* of the earlier history already explains the gain of explicit
predictive memory. Unlike DeYi the states are not optimised against future prediction and carry no
semantic or categorical organisation -- the only structure is time. Slot 0 is the most recent
segment (the ordering convention of ``04_baseline_decisions.md``); K = 1 reduces to one global mean
over H_old.

The artifact is written in exactly the format the DeYi encoder exports (``DeYi.STATE_FORMAT``), so
TIGER consumes it through its ordinary continuous-memory path:

    python -m src.model.baseline.pooling --task product --recent 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import open_store
from src.model.baseline.cause import catalog_table
from src.model.deyi.model import DeYi


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="output/data")
    parser.add_argument("--dataset", default="single_channel")
    parser.add_argument("--task", default="product")
    parser.add_argument("--item-space", default="output/model/item/semantic_space")
    parser.add_argument("--out", default="output/model/compress/pooling")
    parser.add_argument("--recent", type=int, default=16)
    parser.add_argument("--interests", type=int, default=1, help="memory slots (time segments)")
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--splits", default="train,test")
    return parser.parse_args(argv)


def memories(hist, valid, length, table, position_of_row, recent, k, chunk):
    """``k`` mean-pooled states per row: the old history split into ``k`` equal time segments.

    ``hist`` holds physical item rows, ``valid`` marks the real (non-padding) events of a row and
    ``length`` is its history length. Slot 0 is the most recent segment. A row with fewer than
    ``k`` old events fills only the newest slots and leaves the rest mask=False, which the consumer
    reads as an unfilled slot (the partial-fill convention of ``04_baseline_decisions.md``).
    """
    rows, width = hist.shape
    dim = table.shape[1]
    states = np.zeros((rows, k, dim), dtype=np.float32)
    mask = np.zeros((rows, k), dtype=bool)
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        block = slice(start, stop)
        events = hist[block].astype(np.int64)
        keep = valid[block]
        rank = np.cumsum(keep, axis=1, dtype=np.int64) - 1
        old = keep & (rank < (length[block] - recent)[:, None])
        count = old.sum(axis=1, dtype=np.int64)
        # Position inside H_old, oldest event first; equal-count segments map onto slots with
        # slot 0 = newest, so the slot embedding keeps one meaning across rows.
        position = np.where(old, np.cumsum(old, axis=1, dtype=np.int64) - 1, 0)
        segment = np.minimum(k - 1, (position * k) // np.maximum(count, 1)[:, None])
        slot = (k - 1) - segment
        unit = table[position_of_row[events]]
        unit /= np.maximum(np.linalg.norm(unit, axis=-1, keepdims=True), 1e-12)
        for index in range(k):
            bucket = old & (slot == index)
            filled = bucket.sum(axis=1, dtype=np.int64)
            mask[block, index] = filled > 0
            pooled = np.einsum("rw,rwd->rd", bucket.astype(np.float32), unit)
            states[block, index] = pooled / np.maximum(filled, 1)[:, None]
    return states, mask


def build_split(dataset, task, split, recent, k, chunk, table, position):
    data = dataset.split(task, split)
    history, valid = data.history_rows(np.arange(data.rows), mode="full")
    length = valid.sum(axis=1, dtype=np.int64)
    states, mask = memories(history, valid, length, table, position, recent, k, chunk)
    return states, mask, np.asarray(data.source_row_idx, dtype=np.int64), length


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    store = open_store(args.data_root)
    dataset = store.dataset(args.dataset)
    arm = Path(args.out) / args.dataset / args.task / ("k%d_r%d" % (int(args.interests), int(args.recent)))
    (arm / "states").mkdir(parents=True, exist_ok=True)
    catalog, table, position = catalog_table(args.item_space, dataset, args.task)
    print("[pooling] task catalog %d rows, dense table %.2f GiB"
          % (catalog.size, table.nbytes / 2 ** 30), flush=True)
    summary = {"format": DeYi.STATE_FORMAT, "method": "pooling",
               "design": "mean_of_unit_item_vectors_over_equal_time_segments_of_old_history",
               "dataset": args.dataset, "task": args.task, "recent": int(args.recent),
               "num_interests": int(args.interests), "state_dim": None, "splits": {}}
    for split in [name.strip() for name in args.splits.split(",") if name.strip()]:
        states, mask, source_rows, length = build_split(
            dataset, args.task, split, args.recent, args.interests, args.chunk, table, position)
        payload = {
            "format": DeYi.STATE_FORMAT,
            "states": torch.from_numpy(states.astype(np.float16)),
            "mask": torch.from_numpy(mask),
            "source_row_idx": torch.from_numpy(source_rows),
            "num_interests": int(args.interests),
            "state_dim": int(states.shape[-1]),
            "method": "pooling",
        }
        torch.save(payload, arm / "states" / ("%s.pt" % split))
        summary["state_dim"] = int(states.shape[-1])
        summary["splits"][split] = {
            "rows": int(states.shape[0]),
            "mean_filled_slots": float(mask.sum(axis=1).mean()),
            "mean_old_history": float(length.mean() - args.recent),
            "seconds": round(time.time() - started, 1),
        }
        print("[pooling] %s rows=%d filled_slots=%.3f mean_old=%.1f"
              % (split, states.shape[0], summary["splits"][split]["mean_filled_slots"],
                 summary["splits"][split]["mean_old_history"]))
    (arm / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[pooling] wrote %s in %.1fs" % (arm, time.time() - started))


if __name__ == "__main__":
    main()
