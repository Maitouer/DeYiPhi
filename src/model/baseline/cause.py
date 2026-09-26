"""CAUSE-Core memory states: categorical aggregation over the old history.

``docs/v1_algorithm/04_baseline.md`` §4. Deterministic, with no learned parameters and no
training at all:

    H_old --(bucket by the level-1 TIGER SID, keep the 4 most recently active buckets,
            mean-pool the unit item vectors of each bucket)--> M = {m_1..m_4}

The states are written in exactly the artifact format the DeYi encoder exports
(``DeYi.STATE_FORMAT``), so TIGER consumes them through its ordinary continuous-memory path.
Only ``H_old`` enters: the last ``recent`` valid events are the retained context ``R`` and are
never touched here.

    python -m src.model.baseline.cause --task product --recent 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import open_store
from src.model.deyi.model import DeYi


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="output/data")
    parser.add_argument("--dataset", default="single_channel")
    parser.add_argument("--task", default="product")
    parser.add_argument("--item-space", default="output/model/item/semantic_space")
    parser.add_argument("--codebook", default="output/model/item/tiger_codebook/codes.npy")
    parser.add_argument("--out", default="output/model/compress/cause")
    parser.add_argument("--recent", type=int, default=16)
    parser.add_argument("--interests", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--splits", default="train,test")
    return parser.parse_args(argv)


def memories(hist, valid, length, category_of_row, table, position_of_row, recent, k, chunk):
    """Four most recently active categories per row, each mean-pooled into one state.

    ``hist`` holds physical item rows, ``valid`` marks the real (non-padding) events of a row and
    ``length`` is its history length. Everything is vectorised per chunk; the per-row loop that a
    naive implementation would need is replaced by a "latest position per category" table.
    """
    rows, width = hist.shape
    dim = table.shape[1]
    states = np.zeros((rows, k, dim), dtype=np.float32)
    mask = np.zeros((rows, k), dtype=bool)
    columns = 1 + max(int(category_of_row.max(initial=0)), 0) + 1
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        block = slice(start, stop)
        # Positions are rebuilt per chunk: a chunk-wide row subset cannot be indexed with the
        # full table (and broadcasting the whole table again per chunk costs nothing).
        position = np.broadcast_to(np.arange(width, dtype=np.int64), (stop - start, width))
        events = hist[block].astype(np.int64)
        keep = valid[block]
        rank = np.cumsum(keep, axis=1, dtype=np.int64) - 1
        old = keep & (rank < (length[block] - recent)[:, None])
        if not old.any():
            continue
        print("[cause]   chunk %d/%d" % (start // chunk + 1, (rows + chunk - 1) // chunk), flush=True)
        category = np.where(old, category_of_row[events], -1)
        # latest position of every category inside H_old; column 0 is "absent", column c+1 is
        # category c, so a plain argsort of the negated table is the recency ranking of §4.3.
        latest = np.zeros((stop - start, columns), dtype=np.int64)
        np.maximum.at(latest, (np.nonzero(old)[0], category[old] + 1), position[old] + 1)
        selected_categories = np.argsort(-latest[:, 1:], axis=1, kind="stable")[:, :k]

        unit = table[position_of_row[events]]
        unit /= np.maximum(np.linalg.norm(unit, axis=-1, keepdims=True), 1e-12)
        for slot in range(k):
            bucket = old & (category == selected_categories[:, slot:slot + 1])
            count = bucket.sum(axis=1, dtype=np.int64)
            mask[block, slot] = count > 0
            pooled = np.einsum("rw,rwd->rd", bucket.astype(np.float32), unit)
            states[block, slot] = pooled / np.maximum(count, 1)[:, None]
    return states, mask


def catalog_table(item_space, dataset, task):
    """Dense unit-vector table over the task catalog plus the global row -> position map.

    Gathering straight out of the 28 GB fp16 matrix costs ~7.6M random reads per split; the
    catalog of one task is 1.13M rows, so materialising it once turns that into a cache-friendly
    gather and keeps every state identical (the rows are the same vectors).
    """
    catalog = np.asarray(dataset.task_catalog_rows(task), dtype=np.int64)
    table = np.asarray(np.load(Path(item_space) / "embeddings.npy", mmap_mode="r")[catalog],
                       dtype=np.float32)
    table /= np.maximum(np.linalg.norm(table, axis=-1, keepdims=True), 1e-12)
    position = np.full(int(catalog.max()) + 1, -1, dtype=np.int64)
    position[catalog] = np.arange(catalog.size, dtype=np.int64)
    return catalog, table, position


def build_split(dataset, task, split, codebook, recent, k, chunk, table, position):
    data = dataset.split(task, split)
    history, valid = data.history_rows(np.arange(data.rows), mode="full")
    length = valid.sum(axis=1, dtype=np.int64)
    codes = np.load(codebook, mmap_mode="r")
    states, mask = memories(history, valid, length, codes[:, 0].astype(np.int64),
                            table, position, recent, k, chunk)
    return states, mask, np.asarray(data.source_row_idx, dtype=np.int64), length


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    store = open_store(args.data_root)
    dataset = store.dataset(args.dataset)
    arm = Path(args.out) / args.dataset / args.task / ("k%d_r%d" % (args.interests, args.recent))
    (arm / "states").mkdir(parents=True, exist_ok=True)
    catalog, table, position = catalog_table(args.item_space, dataset, args.task)
    print("[cause] task catalog %d rows, dense table %.2f GiB"
          % (catalog.size, table.nbytes / 2 ** 30), flush=True)
    summary = {"format": DeYi.STATE_FORMAT, "method": "cause",
               "design": "level1_sid_buckets_most_recent_4_mean_pooled",
               "dataset": args.dataset, "task": args.task, "recent": int(args.recent),
               "num_interests": int(args.interests), "state_dim": None, "splits": {}}
    for split in [name.strip() for name in args.splits.split(",") if name.strip()]:
        states, mask, source_rows, length = build_split(
            dataset, args.task, split, args.codebook,
            args.recent, args.interests, args.chunk, table, position)
        payload = {
            "format": DeYi.STATE_FORMAT,
            "states": torch.from_numpy(states.astype(np.float16)),
            "mask": torch.from_numpy(mask),
            "source_row_idx": torch.from_numpy(source_rows),
            "num_interests": int(args.interests),
            "state_dim": int(states.shape[-1]),
            "method": "cause",
        }
        torch.save(payload, arm / "states" / ("%s.pt" % split))
        summary["state_dim"] = int(states.shape[-1])
        summary["splits"][split] = {
            "rows": int(states.shape[0]),
            "mean_filled_slots": float(mask.sum(axis=1).mean()),
            "rows_with_four_slots": float((mask.sum(axis=1) == args.interests).mean()),
            "mean_old_history": float(length.mean() - args.recent),
            "seconds": round(time.time() - started, 1),
        }
        print("[cause] %s rows=%d filled_slots=%.3f four_slots=%.3f"
              % (split, states.shape[0], summary["splits"][split]["mean_filled_slots"],
                 summary["splits"][split]["rows_with_four_slots"]))
    (arm / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[cause] wrote %s in %.1fs" % (arm, time.time() - started))


if __name__ == "__main__":
    main()
