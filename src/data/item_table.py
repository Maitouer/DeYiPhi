"""Build the shared item table: the PID list plus the 4096-d text matrix.

Two passes over the embedding shards, both borrowed from the previous pipeline because the
shape of the problem has not changed:

* pass A reads only ``pid`` and the validity of ``text_emb``, so the final row numbers are
  known before a single vector is decoded and the destination can be pre-allocated exactly;
* pass B decodes contiguous shard ranges in parallel and writes each worker's rows straight
  into the pre-allocated matrix, so there is no shuffle, no merge step and no duplicated
  copy. Decoding, not the disk, is the bottleneck (measured 274 MiB/s through this path).

Row 0 is the zero sentinel. A PID owns the row of its first occurrence that actually carries
a text vector; PIDs without any text get no row and stay 0, which downstream filters read as
"missing item".
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TEXT_DIM = 4096
BATCH_ROWS = 131072


def shard_files(embeddings_dir: Path):
    files = sorted(Path(embeddings_dir).resolve().rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("no embedding shards under %s" % embeddings_dir)
    return files


def scan_footers(files) -> dict:
    """Read only parquet footers: how many shards, rows and bytes we are dealing with."""
    rows = row_groups = size = 0
    for path in files:
        metadata = pq.ParquetFile(path).metadata
        rows += metadata.num_rows
        row_groups += metadata.num_row_groups
        size += path.stat().st_size
    return {"shards": len(files), "rows": rows, "row_groups": row_groups, "bytes": size}


def _shard_worker(job):
    """Decode ``pid`` + text validity for a contiguous shard range; write a sidecar."""
    index, paths, out_path = job
    pids, valid, counts = [], [], []
    for path in paths:
        rows = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS,
                                                       columns=["pid", "text_emb"]):
            pids.append(batch.column("pid").to_numpy(zero_copy_only=False).astype(np.int64))
            valid.append(batch.column("text_emb").is_valid().to_numpy(zero_copy_only=False))
            rows += batch.num_rows
        counts.append(rows)
    np.savez(out_path, pid=np.concatenate(pids), valid=np.concatenate(valid),
             counts=np.asarray(counts, dtype=np.int64))
    return index, str(out_path), int(sum(counts))


def build_index(files, requested, work_dir, workers, progress):
    """Return ``(pids, keep_index, shard_rows)`` for the shared item space.

    ``pids`` starts with the 0 sentinel. ``keep_index`` holds one entry per scanned row: the
    global row it was stored at, or -1 when the row was not kept. ``shard_rows`` lets pass B
    address scanned rows by shard without re-reading anything.
    """
    workers = max(1, min(int(workers), len(files)))
    # Many small chunks rather than one per worker: progress lines arrive every few seconds and
    # shards stay balanced by their real size.
    chunk_count = max(1, min(len(files), workers * 20))
    chunks = np.array_split(np.arange(len(files)), chunk_count)
    index_dir = Path(work_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(index, [files[i] for i in chunk], index_dir / ("range-%03d.npz" % index))
            for index, chunk in enumerate(chunks)]

    results, done_shards = [], 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for completed, result in enumerate(pool.map(_shard_worker, jobs), start=1):
            results.append(result)
            done_shards += len(jobs[completed - 1][1])
            progress.tick(completed, len(jobs),
                          "chunks=%d/%d shards=%d/%d" % (completed, len(jobs), done_shards,
                                                         len(files)),
                          counters={"shards_scanned": done_shards})
    results.sort(key=lambda item: item[0])

    pid_parts, valid_parts, shard_rows = [], [], []
    for _index, path, _count in results:
        data = np.load(path)
        pid_parts.append(data["pid"])
        valid_parts.append(data["valid"])
        shard_rows.extend(int(value) for value in data["counts"])
    pid = np.concatenate(pid_parts)
    valid = np.concatenate(valid_parts)

    requested_sorted = np.asarray(requested, dtype=np.int64)
    position = np.clip(np.searchsorted(requested_sorted, pid), 0, len(requested_sorted) - 1)
    candidates = np.flatnonzero(valid & (requested_sorted[position] == pid))
    keys = pid[candidates]
    _, first = np.unique(keys, return_index=True)
    kept_rows = np.sort(candidates[first])

    keep_index = np.full(len(pid), -1, dtype=np.int32)
    keep_index[kept_rows] = np.arange(1, len(kept_rows) + 1, dtype=np.int32)
    pids = np.concatenate((np.zeros(1, dtype=np.int64), pid[kept_rows]))
    progress.counters_update(scanned_rows=int(len(pid)), kept_items=int(len(kept_rows)))
    np.savez(Path(work_dir) / "keep_index.npz", keep=keep_index,
             shard_rows=np.asarray(shard_rows, dtype=np.int64))
    return pids, keep_index, np.asarray(shard_rows, dtype=np.int64)


def _create_text_matrix(path, rows):
    """Pre-allocate ``text.npy``; row 0 stays the zero sentinel.

    ``open_memmap`` leaves the file sparse, so unwritten regions read back as zeros and we do
    not spend minutes zero-filling hundreds of gigabytes.
    """
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                       shape=(int(rows), TEXT_DIM))
    matrix.flush()


def _text_worker(job):
    out_path, paths, keep_path, scanned_offset = job
    keep_all = np.load(keep_path)["keep"]
    destination = np.load(out_path, mmap_mode="r+")
    written = runs = 0
    cursor = int(scanned_offset)
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS,
                                                       columns=["text_emb"]):
            rows = batch.num_rows
            keep = keep_all[cursor:cursor + rows]
            cursor += rows
            target = keep >= 0
            if not target.any():
                continue
            text = batch.column("text_emb")
            values = text.values.to_numpy(zero_copy_only=False).reshape(-1, TEXT_DIM)
            starts = (text.offsets.to_numpy()[:-1] // TEXT_DIM)[target]
            destination[keep[target].astype(np.int64)] = values[starts]
            runs += 1
            written += len(starts)
    destination.flush()
    return written, runs


def write_text(files, keep_path, out_path, workers, progress, total_rows):
    _create_text_matrix(out_path, total_rows)
    workers = max(1, min(int(workers), len(files)))
    chunk_count = max(1, min(len(files), workers * 20))
    chunks = np.array_split(np.arange(len(files)), chunk_count)
    shard_rows = np.load(keep_path)["shard_rows"]
    shard_offsets = np.r_[0, np.cumsum(shard_rows)]
    jobs = [(str(out_path), [files[i] for i in chunk], str(keep_path),
             int(shard_offsets[chunk[0]])) for index, chunk in enumerate(chunks)]

    started = time.time()
    written = done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rows, _runs in pool.map(_text_worker, jobs):
            done += 1
            written += rows
            elapsed = max(time.time() - started, 1e-6)
            bytes_done = written * TEXT_DIM * 4
            progress.tick(done, len(jobs),
                          "chunks=%d/%d rows=%d %.1fGiB %.0fMiB/s"
                          % (done, len(jobs), written, bytes_done / 2 ** 30,
                             bytes_done / 2 ** 20 / elapsed),
                          counters={"rows_written": written, "bytes_written": bytes_done})
    seconds = time.time() - started
    return {"rows_written": written, "seconds": seconds,
            "mib_per_s": written * TEXT_DIM * 4 / 2 ** 20 / max(seconds, 1e-6)}


def build(config, files, requested, item_root, work_dir, workers, progress, resume=False):
    """Materialise ``items/`` and return the item-table statistics."""
    item_root = Path(item_root)
    item_root.mkdir(parents=True, exist_ok=True)
    pids_path = item_root / "pids.npy"
    text_path = item_root / "text.npy"
    keep_path = Path(work_dir) / "keep_index.npz"
    footers = scan_footers(files)
    progress.line("item-table", "shards=%d rows=%d payload=%.1fGiB"
                  % (footers["shards"], footers["rows"], footers["bytes"] / 2 ** 30),
                  force=True)

    if resume and pids_path.is_file() and keep_path.is_file():
        pids = np.load(pids_path)
        keep_index = np.load(keep_path)["keep"]
        progress.line("item-table", "reused index rows=%d" % len(pids), force=True)
    else:
        started = time.time()
        pids, keep_index, _ = build_index(files, requested, work_dir, workers, progress)
        temporary = pids_path.with_name(pids_path.name + ".tmp")
        with open(temporary, "wb") as stream:
            np.save(stream, pids)
        temporary.replace(pids_path)
        progress.line("item-table", "index items=%d scanned=%d (%.1fs)"
                      % (len(pids) - 1, len(keep_index), time.time() - started), force=True)

    if resume and text_path.is_file():
        text_stats = {"reused": True}
        progress.line("item-table", "reused text matrix", force=True)
    else:
        text_stats = write_text(files, keep_path, text_path, workers, progress, len(pids))
        progress.line("item-table", "text rows=%d %.1fGiB %.0fMiB/s (%.1fs)"
                      % (text_stats["rows_written"],
                         text_stats["rows_written"] * TEXT_DIM * 4 / 2 ** 30,
                         text_stats["mib_per_s"], text_stats["seconds"]), force=True)

    payload = {
        "kind": "item_table",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": str(config.embeddings),
        "text": {"file": text_path.name, "dtype": "float32",
                 "shape": [int(len(pids)), TEXT_DIM], "row0": "zero-padding-sentinel"},
        "pids": {"file": pids_path.name, "dtype": "int64", "shape": [int(len(pids))]},
        "scanned_rows": int(footers["rows"]),
        "shards": int(footers["shards"]),
        "requested_items": int(len(requested)),
        "stored_items": int(len(pids) - 1),
        "missing_items": int(len(requested) - (len(pids) - 1)),
        "text_build": text_stats,
    }
    destination = item_root / "item_table.json"
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(destination)
    return payload
