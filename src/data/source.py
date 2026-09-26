"""Read the RecIF release tables and materialise one Arrow table per task and split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import PipelineConfig, Task


def _as_int_lists(table: pa.Table, fields) -> pa.Table:
    """Cast history list columns to ``list<int64>``.

    The release stores the ad, goods and long-view histories as ``list<double>`` while the
    targets are ``list<int64>``. Values stay exact, but mixing the two dtypes across a join is
    how silent mismatches start, so the pipeline normalises once, at read time.
    """
    for name in fields:
        column = table[name]
        if pa.types.is_floating(column.type.value_type):
            index = table.schema.get_field_index(name)
            table = table.set_column(index, name, pc.cast(column, pa.list_(pa.int64())))
    return table


def tail(array, limit: int):
    """Keep the most recent ``limit`` entries of every row (chronological order)."""
    if hasattr(array, "combine_chunks"):
        array = array.combine_chunks()
    if limit <= 0:
        raise ValueError("history limit must be positive")
    offsets = array.offsets.to_numpy()
    lengths = np.diff(offsets)
    starts = np.maximum(offsets[:-1], offsets[1:] - limit)
    row = np.repeat(np.arange(len(lengths)), lengths)
    position = np.arange(offsets[-1])
    selected = position >= starts[row]
    kept = pc.take(array.values, pa.array(position[selected]))
    new_offsets = np.r_[0, np.cumsum(np.minimum(lengths, limit))]
    return pa.ListArray.from_arrays(pa.array(new_offsets), kept)


def read_train(config: PipelineConfig) -> pa.Table:
    """Read the release wide table once, with every column any task needs."""
    fields = sorted({history.field for dataset in config.datasets for task in dataset.tasks
                     for history in task.histories}
                    | {task.target_field for dataset in config.datasets for task in dataset.tasks})
    behaviors = sorted({name for dataset in config.datasets for task in dataset.tasks
                        for name in task.behaviors})
    columns = ["uid", "split", *fields, *("hist_video_%s" % name for name in behaviors)]
    table = pq.read_table(Path(config.source) / "onerec_bench_release.parquet", columns=columns)
    return _as_int_lists(table, fields)


def build_train_table(table: pa.Table, task: Task, min_primary_history: int,
                      max_rows: int | None = None) -> pa.Table:
    """Select, truncate and filter one task's training rows out of the wide table."""
    keep = pc.equal(table["split"], 0)
    columns = {"source_row_idx": pa.array(np.arange(len(table), dtype=np.int64)),
               "uid": table["uid"]}
    columns["target"] = table[task.target_field]
    keep = pc.and_(keep, pc.greater(pc.list_value_length(columns["target"]), 0))
    for history in task.histories:
        values = tail(table[history.field], history.limit)
        columns[history.channel] = values
        if history.channel == task.primary:
            keep = pc.and_(keep, pc.greater_equal(pc.list_value_length(values),
                                                  min_primary_history))
    for name in task.behaviors:
        columns["behavior." + name] = tail(table["hist_video_" + name],
                                           task.histories[0].limit)
    selected = pa.table(columns).filter(keep)
    if max_rows:
        selected = selected.slice(0, int(max_rows))
    return selected


def build_test_table(config: PipelineConfig, task: Task, min_primary_history: int,
                     max_rows: int | None = None) -> pa.Table:
    """Build one task's test table from the released benchmark file.

    The released test files already cap the cross-domain video channel and the product
    channel at 100, so ``tail`` is a no-op there; video (512) and ad (200) are the two caps
    that still bind. Behaviour channels do not exist on the test side and are written as
    ``-1`` ("not observed"), exactly like the source implies.
    """
    table = pq.read_table(Path(config.source) / task.test_file)
    metadata = [json.loads(value) for value in table["metadata"].to_pylist()]
    columns = {
        "source_row_idx": pa.array(np.arange(len(table), dtype=np.int64)),
        "uid": pa.array([item.get("uid") for item in metadata], type=pa.int64()),
        "target": pa.array([item[task.test_target] for item in metadata],
                           type=pa.list_(pa.int64())),
    }
    keep = pc.greater(pc.list_value_length(columns["target"]), 0)
    for history in task.histories:
        values = tail(table[task.test_history[history.channel]], history.limit)
        columns[history.channel] = values
        if history.channel == task.primary:
            keep = pc.and_(keep, pc.greater_equal(pc.list_value_length(values),
                                                  min_primary_history))
    if task.behaviors:
        reference = columns[task.histories[0].channel]
        lengths = pc.list_value_length(reference).to_numpy(zero_copy_only=False)
        columns.update({
            "behavior." + name: pa.array([[-1] * int(length) for length in lengths],
                                         type=pa.list_(pa.int8()))
            for name in task.behaviors
        })
    selected = pa.table(columns).filter(keep)
    if max_rows:
        selected = selected.slice(0, int(max_rows))
    return selected


def requested_pids(samples) -> np.ndarray:
    """Union of every PID any task or split references (histories, targets and behaviours)."""
    parts = []
    for table in samples:
        for name in table.column_names:
            if name in ("source_row_idx", "uid"):
                continue
            for chunk in table[name].chunks:
                parts.append(np.asarray(chunk.flatten().to_numpy(zero_copy_only=False),
                                        dtype=np.int64))
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(parts))
