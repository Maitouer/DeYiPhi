"""Consistency checks for a freshly written ``output/data``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def run_checks(store, split_stats, samples, progress, sample_rows=200, seed=2026):
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        progress.line("selfcheck", "%s %s %s" % ("PASS" if ok else "FAIL", name, detail),
                      force=True)

    items = store.items
    text, pids = items.text, items.pids
    rows = len(items)
    check("text/pids length match", text.shape[0] == rows and pids.shape[0] == rows,
          "text=%d pids=%d" % (text.shape[0], pids.shape[0]))
    check("text row 0 is the zero sentinel", not np.asarray(text[0]).any())
    check("pids[0] == 0", int(pids[0]) == 0)
    check("pids[1:] are unique", len(np.unique(pids[1:])) == rows - 1)
    # The release contains a few genuine PID 0 entries. The sentinel owns row 0 and a real
    # PID 0 owns its own row, so the lookup must resolve 0 to that real row rather than to
    # the sentinel; padding is decided by ``lengths``, never by the row value.
    real_zero = np.flatnonzero(np.asarray(pids[1:]) == 0)
    if len(real_zero):
        resolved = int(items.rows_for_pids([0])[0])
        check("real PID 0 resolves past the sentinel",
              len(real_zero) == 1 and resolved == int(real_zero[0]) + 1,
              "rows=%s resolved=%d" % (real_zero.tolist(), resolved))
    else:
        check("real PID 0 resolves past the sentinel", True, "no real PID 0 in scope")
    sample = np.asarray(text[1:1 + 64])
    check("text sample is non-zero", bool(np.abs(sample).sum() > 0))
    check("text has no NaN/Inf", bool(np.isfinite(np.asarray(text[::97][:512])).all()))

    referenced = []
    for stats in split_stats:
        dataset_root = Path(stats["root"])
        tensors = load_file(str(dataset_root / stats["task"] / ("%s.bin" % stats["split"])))
        for name, value in tensors.items():
            if name.startswith("history.") or name == "target":
                referenced.append(np.asarray(value).reshape(-1))
    values = np.concatenate(referenced) if referenced else np.zeros(1, dtype=np.int32)
    check("history/target values in range",
          bool(len(values) == 0 or (values.min() >= 0 and values.max() < rows)),
          "max=%d" % (int(values.max()) if len(values) else -1))

    for name in store.datasets():
        dataset = store.dataset(name)
        catalog = np.asarray(dataset.catalog_rows)
        catalog_ok = bool(len(catalog) and catalog.min() > 0 and np.all(np.diff(catalog) > 0))
        check("%s catalog ascending, non-empty and without 0" % name, catalog_ok,
              "n=%d" % len(catalog))
        check("%s catalog within issued rows" % name,
              bool(len(catalog) == 0 or catalog.max() < rows),
              "max=%d" % (int(catalog.max()) if len(catalog) else -1))
        mine = [stats for stats in split_stats if stats["dataset"] == name]
        used = []
        for stats in mine:
            tensors = load_file(str(Path(stats["root"]) / stats["task"]
                                    / ("%s.bin" % stats["split"])))
            for key, value in tensors.items():
                if key.startswith("history.") or key == "target":
                    array = np.asarray(value).reshape(-1)
                    used.append(array[array != 0])
        used = np.unique(np.concatenate(used)) if used else np.zeros(0, dtype=np.int64)
        check("%s catalog covers every referenced row" % name,
              bool(len(used) == 0 or np.isin(used, catalog).all()),
              "referenced=%d" % int(len(used)))
        problems = dataset.validate()
        check("%s declarative validation" % name, not problems, "; ".join(problems))

        # Each task is evaluated on its own item universe, so its catalog must be a real
        # candidate set -- not merely a slice of the dataset's.
        task_catalogs = []
        for task in dataset.tasks():
            rows = np.asarray(dataset.task_catalog_rows(task))
            task_catalogs.append(rows)
            check("%s/%s catalog ascending, non-empty and without 0" % (name, task),
                  bool(len(rows) and rows.min() > 0 and np.all(np.diff(rows) > 0)),
                  "n=%d" % len(rows))
            used = []
            for stats in split_stats:
                if stats["dataset"] != name or stats["task"] != task:
                    continue
                tensors = load_file(str(Path(stats["root"]) / stats["task"]
                                        / ("%s.bin" % stats["split"])))
                for key, value in tensors.items():
                    if key.startswith("history.") or key == "target":
                        array = np.asarray(value).reshape(-1)
                        used.append(array[array != 0])
            used = np.unique(np.concatenate(used)) if used else np.zeros(0, dtype=np.int64)
            check("%s/%s catalog covers every referenced row" % (name, task),
                  bool(len(used) == 0 or np.isin(used, rows).all()),
                  "referenced=%d" % len(used))
        union = (np.unique(np.concatenate(task_catalogs)).astype(np.int32) if task_catalogs
                 else np.zeros(0, dtype=np.int32))
        check("%s dataset catalog is the union of its task catalogs" % name,
              bool(np.array_equal(union, catalog)), "dataset=%d" % len(catalog))

    for stats in split_stats:
        tensors = load_file(str(Path(stats["root"]) / stats["task"]
                                / ("%s.bin" % stats["split"])))
        count = len(tensors["source_row_idx"])
        shapes_ok = all(tensors[name].shape[0] == count for name in tensors)
        padding_ok = True
        for name, value in tensors.items():
            if name.startswith("lengths.") or name == "target_length":
                key = "target" if name == "target_length" else "history." + name.split(".", 1)[1]
                matrix = np.asarray(tensors[key])
                real = np.arange(matrix.shape[1])[None, :] < np.asarray(value)[:, None]
                padding_ok &= bool((matrix[~real] == 0).all())
        check("%s/%s/%s shapes and padding" % (stats["dataset"], stats["task"], stats["split"]),
              shapes_ok and padding_ok)

    generator = np.random.default_rng(seed)
    parity_ok, detail = True, ""
    for stats in split_stats:
        table = samples[(stats["dataset"], stats["task"], stats["split"])]
        tensors = load_file(str(Path(stats["root"]) / stats["task"]
                                / ("%s.bin" % stats["split"])))
        count = len(tensors["source_row_idx"])
        if count == 0:
            continue
        picked = np.sort(generator.choice(count, size=min(sample_rows, count), replace=False))
        # The table is already filtered, so ``source_row_idx`` is not a positional index into
        # it; resolving it through a sorted lookup first is the whole point.
        identity = np.asarray(table["source_row_idx"].to_numpy(zero_copy_only=False))
        order = np.argsort(identity)
        wanted = np.asarray(tensors["source_row_idx"])[picked]
        positions = order[np.searchsorted(identity[order], wanted)]
        for name in tensors:
            if not name.startswith("history."):
                continue
            channel = name.split(".", 1)[1]
            length = np.asarray(tensors["lengths." + channel])[picked]
            from_pids = np.asarray(pids)[np.asarray(tensors[name])[picked]]
            source = table[channel].combine_chunks()
            offsets = source.offsets.to_numpy()
            for index, position in enumerate(positions):
                start, stop = int(offsets[position]), int(offsets[position + 1])
                expected = np.asarray(
                    source.values.to_numpy(zero_copy_only=False)[start:stop])
                got = from_pids[index][:int(length[index])]
                if len(expected) != len(got) or not np.array_equal(expected, got):
                    parity_ok = False
                    detail = "%s/%s/%s row=%d channel=%s" % (
                        stats["dataset"], stats["task"], stats["split"], position, channel)
                    break
            if not parity_ok:
                break
        if not parity_ok:
            break
    check("sampled %d rows match the source parquet" % sample_rows, parity_ok, detail)

    passed = sum(1 for item in checks if item["ok"])
    return {"passed": passed, "failed": len(checks) - passed, "checks": checks}
