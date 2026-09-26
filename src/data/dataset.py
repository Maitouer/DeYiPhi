"""The single read entry point for ``output/data``.

    store = open_store("output/data")           # shared item table + both datasets
    dataset = store.dataset("multi_channel")
    split = dataset.split("ad", "train")
    rows, mask = split.history_rows(indices, mode="recent", recent=32)
    vectors = store.items.vectors(rows)         # only if a model needs item embeddings

Channel order is read from the split file's own ``history_channels`` header and must agree
with the dataset manifest. There is no fallback to the tensor key order: safetensors writes
keys alphabetically, which silently turned a declared ``[video, ad]`` into ``[ad, video]``
and made the "recent N" window take the tail of the wrong channel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file


class ItemTable:
    """The shared item table: physical row -> PID, plus the 4096-d text matrix."""

    def __init__(self, root, meta):
        self.root = Path(root)
        self.meta = meta
        self.text_path = self.root / meta["text"]["file"]
        self.pids_path = self.root / meta["pids"]["file"]
        self.text = np.load(self.text_path, mmap_mode="r")
        self.pids = np.load(self.pids_path, mmap_mode="r")
        order = np.argsort(self.pids[1:], kind="stable")
        self._sorted_pids = self.pids[1:][order]
        self._sorted_rows = (order + 1).astype(np.int32)

    def __len__(self):
        return int(self.pids.shape[0])

    def rows_for_pids(self, pids):
        values = np.asarray(pids, dtype=np.int64)
        position = np.clip(np.searchsorted(self._sorted_pids, values), 0,
                           len(self._sorted_pids) - 1)
        ok = self._sorted_pids[position] == values
        return np.where(ok, self._sorted_rows[position], 0).astype(np.int32)

    def vectors(self, rows):
        return np.asarray(self.text[np.asarray(rows)])

    def pids_of(self, rows):
        return np.asarray(self.pids[np.asarray(rows)])


class Split:
    def __init__(self, path, spec, manifest):
        self.path = Path(path)
        self.meta = manifest
        self.spec = spec
        with safe_open(str(self.path), framework="numpy") as handle:
            header = dict(handle.metadata() or {})
        stamped = header.get("history_channels")
        if not stamped:
            raise ValueError(
                "%s has no history_channels header; the file cannot state its own channel "
                "order and guessing it from tensor key names is not allowed" % self.path
            )
        declared = tuple(part.strip() for part in str(stamped).split(",") if part.strip())
        expected = tuple(str(name) for name in spec["channels"])
        if declared != expected:
            raise ValueError(
                "%s declares channels %s but the dataset manifest declares %s"
                % (self.path, declared, expected)
            )
        self.channels = declared
        self.primary = str(header.get("primary_channel") or spec["primary"])
        self.tensors = load_file(str(self.path))
        self.rows = int(self.tensors["source_row_idx"].shape[0])
        self.target = self.tensors["target"]
        self.target_length = self.tensors["target_length"]
        self.source_row_idx = self.tensors["source_row_idx"]
        self.uid = self.tensors["uid"]
        self.behavior = self.tensors.get("behavior")

    def history(self, channel):
        return self.tensors["history." + channel]

    def lengths(self, channel):
        return self.tensors["lengths." + channel]

    def history_rows(self, indices, mode="recent", recent=32):
        """Concatenate the declared channel blocks, then keep the last ``recent`` items.

        Channel blocks contain right padding and do not encode a cross-channel timestamp
        order, so valid positions are compacted before the tail is selected.

        Returns ``(rows, mask)`` with ``rows`` the physical item rows (0 = padding) and
        ``mask`` the per-slot validity.
        """
        indices = np.asarray(indices, dtype=np.int64)
        blocks, masks = [], []
        for name in self.channels:
            values = np.asarray(self.history(name)[indices], dtype=np.int64)
            lengths = np.asarray(self.lengths(name)[indices], dtype=np.int64)
            blocks.append(values)
            masks.append(np.arange(values.shape[1])[None, :] < lengths[:, None])
        values = np.concatenate(blocks, axis=1)
        valid = np.concatenate(masks, axis=1)
        lengths = valid.sum(axis=1, dtype=np.int64)
        if mode != "recent" or recent <= 0:
            return values, valid
        kept = np.minimum(lengths, int(recent))
        width = int(kept.max()) if len(kept) else 0
        output = np.zeros((len(indices), width), dtype=np.int64)
        mask = np.arange(width)[None, :] < kept[:, None]
        if width:
            ranks = np.cumsum(valid, axis=1, dtype=np.int64) - 1
            offsets = ranks - (lengths - kept)[:, None]
            selected = valid & (offsets >= 0)
            owners, positions = np.nonzero(selected)
            output[owners, offsets[owners, positions]] = values[owners, positions]
        return output, mask


class Dataset:
    def __init__(self, root, store):
        self.root = Path(root)
        self.store = store
        self.meta = json.loads((self.root / "dataset.json").read_text(encoding="utf-8"))
        if self.meta.get("kind") != "recommendation_dataset":
            raise ValueError("%s is not a recommendation dataset manifest" % self.root)
        self.name = str(self.meta["name"])
        self.catalog_path = self.root / str(self.meta["catalog"])
        self._catalog = None

    @property
    def items(self):
        return self.store.items

    @property
    def catalog_rows(self):
        if self._catalog is None:
            self._catalog = np.load(self.catalog_path, mmap_mode="r")
        return self._catalog

    def task_catalog_rows(self, task):
        """The candidate universe of one task (a subset of this dataset's catalog)."""
        return np.load(self.root / str(self.meta["tasks"][task]["catalog"]), mmap_mode="r")

    def tasks(self):
        return tuple(self.meta["tasks"])

    def channels(self, task):
        return tuple(str(name) for name in self.meta["tasks"][task]["channels"])

    def split(self, task, split):
        spec = self.meta["tasks"][task]["splits"][split]
        return Split(self.root / spec["file"],
                     {"channels": self.channels(task),
                      "primary": str(self.meta["tasks"][task]["primary"])},
                     self.meta)

    def identity(self):
        """Declarative identity used by downstream artifacts to detect a stale build."""
        return {"dataset": self.name, "created": str(self.meta["created"]),
                "items": int(len(self.items))}

    def validate(self):
        """Cheap consistency checks that catch half-written or mismatched outputs."""
        problems = []
        item_rows = len(self.items)
        if self.items.text.shape[0] != item_rows:
            problems.append("text/pids length mismatch")
        if int(self.items.pids[0]) != 0:
            problems.append("pids[0] must be the 0 sentinel")
        catalog = np.asarray(self.catalog_rows)
        if len(catalog) and (catalog.min() <= 0 or not np.all(np.diff(catalog) > 0)):
            problems.append("catalog must be ascending and exclude 0")
        if len(catalog) and int(catalog.max()) >= item_rows:
            problems.append("catalog row out of range")
        for task, spec in self.meta["tasks"].items():
            for split, split_spec in spec["splits"].items():
                data = self.split(task, split)
                if data.rows != int(split_spec["rows"]):
                    problems.append("%s/%s declared rows mismatch" % (task, split))
                if data.channels != tuple(str(name) for name in spec["channels"]):
                    problems.append("%s/%s channel order mismatch" % (task, split))
                for name in data.channels:
                    matrix = data.history(name)
                    if int(matrix.max(initial=0)) >= item_rows:
                        problems.append("%s/%s history row out of range" % (task, split))
        return problems


class Store:
    """The pipeline output root: one shared item table plus one or more datasets."""

    def __init__(self, root, item_dir="items"):
        self.root = Path(root)
        self.items = ItemTable(self.root / item_dir,
                               json.loads((self.root / item_dir / "item_table.json")
                                          .read_text(encoding="utf-8")))
        if self.items.meta.get("kind") != "item_table":
            raise ValueError("%s/items does not hold an item table" % self.root)

    def dataset(self, name):
        return Dataset(self.root / str(name), self)

    def datasets(self):
        return tuple(sorted(path.parent.name for path in self.root.glob("*/dataset.json")))

    def union_catalog(self, names):
        """Union of several datasets' candidate universes (used by shared codebooks)."""
        arrays = [np.asarray(self.dataset(name).catalog_rows, dtype=np.int64) for name in names]
        return np.unique(np.concatenate(arrays)) if arrays else np.zeros(0, dtype=np.int64)

    def item_space(self, names=None):
        """The shared item table plus the candidate universe a shared artifact is built over.

        The item table is singular; the universe is not, because each dataset has its own.
        A shared artifact (the DeYi PCA space, the Tiger codebook) therefore states its scope
        explicitly, or takes the default of everything this pipeline produced.
        """
        return ItemSpace(self.root, names if names is not None else self.datasets(), self)


class ItemSpace:
    def __init__(self, root, names, store):
        self.root = Path(root)
        self.items = store.items
        self.datasets = tuple(str(name) for name in names)
        self.catalog = store.union_catalog(self.datasets)

    def identity(self):
        return {"dataset": "+".join(self.datasets),
                "created": str(self.items.meta["created"]),
                "rows": int(len(self.items))}

    def validate(self):
        problems = []
        rows = len(self.items)
        if self.items.text.shape[0] != rows or self.items.pids.shape[0] != rows:
            problems.append("text/pids length mismatch")
        if int(self.items.pids[0]) != 0:
            problems.append("pids[0] must be the 0 sentinel")
        catalog = np.asarray(self.catalog)
        if not len(catalog):
            problems.append("catalog is empty")
        else:
            if catalog.min() <= 0 or not np.all(np.diff(catalog) > 0):
                problems.append("catalog must be ascending and exclude 0")
            if int(catalog.max()) >= rows:
                problems.append("catalog row out of range")
        if not np.isfinite(np.asarray(self.items.text[::997][:512])).all():
            problems.append("text matrix contains NaN/Inf")
        return problems


def open_store(root="output/data", item_dir="items"):
    return Store(root, item_dir)


def open_items(root="output/data", item_dir="items"):
    return open_store(root, item_dir).items


def task_index(root="output/data"):
    """The pipeline's own declaration of which dataset owns each task name."""
    return json.loads((Path(root) / "tasks.json").read_text(encoding="utf-8"))


def owner(root, task):
    index = task_index(root)
    if task not in index:
        raise KeyError("%s/tasks.json does not declare task %r" % (root, task))
    return str(index[task])


def open_dataset(root, task, dataset):
    """Open the dataset that owns ``task`` (``dataset`` names it explicitly when needed).

    ``ad`` and ``product`` exist in both datasets, so the owner is never guessed: it comes
    from the pipeline's ``tasks.json`` unless the caller names a dataset.
    """
    root = Path(root)
    name = str(dataset) if dataset else owner(root, task)
    return open_store(root).dataset(name)


def catalog_rows(root, task, dataset):
    return open_dataset(root, task, dataset).catalog_rows
