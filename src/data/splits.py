"""Write the per-split tensor files and each dataset's candidate universe.

Conventions carried over unchanged from the previous pipeline:

* arrays hold **physical item rows** (an index into ``items/``), right padded with 0;
* a sample is dropped entirely when any history or target entry has no text vector, which
  shows up as row 0;
* ``behavior`` exists only for tasks that declare behaviours, ``-1`` means "not observed";
* ``source_row_idx`` keeps the row identity in the released tables.

New here: the channel order is written into the file's own header, so a reader can never
recover it from a tensor-key scan.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
from safetensors.numpy import load_file, save_file


def row_lookup(pids):
    """Return ``(sorted_pids, rows)`` so that a PID maps to its physical row."""
    order = np.argsort(pids[1:], kind="stable")
    return pids[1:][order], (order + 1).astype(np.int32)


def map_pids(values, sorted_pids, rows):
    """PID -> physical row (0 = missing).

    The release contains a handful of genuine PID 0 entries. They are handled correctly by
    construction: the sentinel owns row 0, a real PID 0 owns its own row >= 1, and this
    lookup resolves 0 to that real row because ``sorted_pids`` contains it. Padding is
    distinguished by ``lengths``, never by the row value.
    """
    position = np.clip(np.searchsorted(sorted_pids, values), 0, len(sorted_pids) - 1)
    ok = sorted_pids[position] == values
    return np.where(ok, rows[position], 0).astype(np.int32)


def _coordinates(offsets):
    lengths = np.diff(offsets)
    row = np.repeat(np.arange(len(lengths)), lengths)
    column = np.arange(offsets[-1]) - np.repeat(offsets[:-1], lengths)
    return row, column


def _padded_rows(array, sorted_pids, rows):
    """Return ``(matrix, lengths)`` for one list column, with PIDs mapped to item rows."""
    array = array.combine_chunks() if hasattr(array, "combine_chunks") else array
    offsets = array.offsets.to_numpy()
    lengths = np.diff(offsets).astype(np.int32)
    width = max(1, int(lengths.max(initial=0)))
    matrix = np.zeros((len(lengths), width), dtype=np.int32)
    if offsets[-1]:
        values = pc.fill_null(array.values, -1).to_numpy(zero_copy_only=False)
        mapped = map_pids(values, sorted_pids, rows)
        row, column = _coordinates(offsets)
        matrix[row, column] = mapped
    return matrix, lengths


def _padded_behaviors(table, behaviors, width):
    """Behaviour channels share the reference history channel's width and are clipped to it."""
    matrix = np.full((len(table), width, len(behaviors)), -1, dtype=np.int8)
    for channel, name in enumerate(behaviors):
        array = table["behavior." + name].combine_chunks()
        offsets = array.offsets.to_numpy()
        row, column = _coordinates(offsets)
        inside = column < width
        values = pc.fill_null(array.values, -1).to_numpy(zero_copy_only=False)
        matrix[row[inside], column[inside], channel] = values[inside]
    return matrix


def write_split(task, tags, split, table, item_path, destination):
    """Write one ``<task>/<split>.bin`` and return its statistics."""
    pids = np.load(item_path)
    sorted_pids, rows = row_lookup(pids)
    channels = list(task.channels)

    matrices, lengths = {}, {}
    missing = np.zeros(len(table), dtype=bool)
    for name in [*channels, "target"]:
        matrix, length = _padded_rows(table[name], sorted_pids, rows)
        matrices[name], lengths[name] = matrix, length
        real = np.arange(matrix.shape[1])[None, :] < length[:, None]
        missing |= ((matrix == 0) & real).any(axis=1)

    behaviors = list(task.behaviors)
    width = max(1, int(lengths[channels[0]].max(initial=0)))
    if behaviors:
        matrices["behavior"] = _padded_behaviors(table, behaviors, width)

    keep = ~missing
    tensors = {"history." + name: np.ascontiguousarray(matrices[name][keep])
               for name in channels}
    tensors["target"] = np.ascontiguousarray(matrices["target"][keep])
    for name in channels:
        tensors["lengths." + name] = np.ascontiguousarray(lengths[name][keep]).astype(np.int32)
    tensors["target_length"] = np.ascontiguousarray(lengths["target"][keep]).astype(np.int32)
    if behaviors:
        tensors["behavior"] = np.ascontiguousarray(matrices["behavior"][keep])
    tensors["source_row_idx"] = table["source_row_idx"].to_numpy(
        zero_copy_only=False)[keep].astype(np.int64)
    tensors["uid"] = np.asarray(table["uid"].to_numpy(
        zero_copy_only=False))[keep].astype(np.int64)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    save_file(tensors, str(temporary), metadata={
        "kind": "split_file",
        "dataset": str(tags["dataset"]),
        "task": str(task.name),
        "split": str(split),
        "history_channels": ",".join(channels),
        "primary_channel": str(task.primary),
        "min_primary_history": str(int(tags["min_primary_history"])),
    })
    temporary.replace(destination)
    return {
        "task": task.name,
        "split": split,
        "file": destination.name,
        "channels": channels,
        "primary": task.primary,
        "rows": int(keep.sum()),
        "removed_missing_item_vector": int(missing.sum()),
        "widths": {name: int(matrices[name].shape[1]) for name in [*channels, "target"]},
        "lengths": {name: {"max": int(lengths[name].max(initial=0)),
                           "mean": round(float(lengths[name][keep].mean()) if keep.any() else 0.0, 1)}
                    for name in [*channels, "target"]},
    }


def _split_job(job):
    task, tags, split, table, item_path, destination = job
    return write_split(task, tags, split, table, item_path, destination)


def write_dataset(dataset, samples, tags, item_path, out_root, workers, progress):
    """Write every task/split of one dataset in parallel; return the split statistics."""
    jobs = [(task, tags, split, samples[(dataset.name, task.name, split)], str(item_path),
             str(Path(out_root) / task.name / ("%s.bin" % split)))
            for task in dataset.tasks for split in ("train", "test")]
    workers = max(1, min(int(workers), len(jobs)))
    stats = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(_split_job, jobs), start=1):
            stats.append(result)
            progress.tick(index, len(jobs), "%s/%s rows=%d removed=%d"
                          % (result["task"], result["split"], result["rows"],
                             result["removed_missing_item_vector"]))
    return stats


def _referenced_rows(root, entries):
    """Every physical item row these splits reference, deduplicated and without the sentinel."""
    referenced = []
    for item in entries:
        tensors = load_file(str(Path(root) / item["task"] / item["file"]))
        for name, value in tensors.items():
            if name.startswith("history.") or name == "target":
                array = np.asarray(value).reshape(-1)
                referenced.append(array[array != 0])
    if not referenced:
        return np.zeros(0, dtype=np.int32)
    return np.unique(np.concatenate(referenced)).astype(np.int32)


def _save_rows(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as stream:
        np.save(stream, rows)
    temporary.replace(path)


def write_catalogs(out_root, stats):
    """One candidate universe per task, plus the dataset's as their union.

    A task is evaluated on its own item universe, so ``<task>/catalog_rows.npy`` is the set
    that task can legally rank over; ``<dataset>/catalog_rows.npy`` stays the union over its
    tasks for consumers that need the whole scene (the DeYi item space, the Tiger codebook).
    """
    out_root = Path(out_root)
    per_task = {}
    for item in stats:
        per_task.setdefault(item["task"], []).append(item)
    rows, sizes = [], {}
    for task in sorted(per_task):
        catalog = _referenced_rows(out_root, per_task[task])
        _save_rows(out_root / task / "catalog_rows.npy", catalog)
        rows.append(catalog)
        sizes[task] = int(len(catalog))
    dataset = (np.unique(np.concatenate(rows)).astype(np.int32) if rows
               else np.zeros(0, dtype=np.int32))
    _save_rows(out_root / "catalog_rows.npy", dataset)
    return dataset, sizes
