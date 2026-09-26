"""Read existing samples and interests without copying their large arrays."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


class Samples:
    """The lightweight consumer view of the canonical data; no text matrix is opened."""

    def __init__(self, root, split, cohort=None):
        folder = Path(root) / ("train" if split == "validation" else split)
        self.history = np.load(folder / "history.npy", mmap_mode="r")
        self.target = np.load(folder / "target.npy", mmap_mode="r")
        self.metadata = pq.read_table(folder / "samples.parquet")
        self.row_indices = np.arange(len(self.metadata))
        if split == "validation" and cohort is None:
            raise ValueError("Validation requires an explicit train/validation cohort")
        if cohort is not None and split in ("train", "validation"):
            selected = pq.read_table(cohort, columns=["source_row_idx"],
                                     filters=[("partition", "=", split)])["source_row_idx"].to_numpy()
            self.row_indices = np.flatnonzero(np.isin(self.metadata["source_row_idx"].to_numpy(), selected))
            self.metadata = self.metadata.take(self.row_indices)
        self.source_rows = self.metadata["source_row_idx"].to_numpy()
        self.history_lengths = self.metadata["history_length"].to_numpy()
        self.target_lengths = self.metadata["target_length"].to_numpy()
        self.pids = np.load(Path(root) / "items/pids.npy", mmap_mode="r")

    def __len__(self):
        return len(self.source_rows)

    def history_rows(self, index, view, recent):
        length = int(self.history_lengths[index])
        start = 0 if view == "full" else max(0, length - recent)
        return self.history[self.row_indices[index], start:length]

    def target_rows(self, index):
        return self.target[self.row_indices[index], :self.target_lengths[index]]


class InterestStates:
    def __init__(self, run_dir, split, source_rows):
        folder = Path(run_dir) / split
        self.h = np.load(folder / "h.npy", mmap_mode="r")
        self.mask = np.load(folder / "mask.npy", mmap_mode="r")
        exported_rows = np.load(folder / "source_row_idx.npy", mmap_mode="r")
        order = np.argsort(exported_rows)
        self.rows = order[np.searchsorted(exported_rows[order], source_rows)]
        # This is a scientific join invariant, not a content fingerprint.
        if not np.array_equal(exported_rows[self.rows], source_rows):
            raise ValueError("Interest source rows do not match the requested samples")
        self.dim = self.h.shape[-1]

    def __getitem__(self, index):
        row = self.rows[index]
        return np.asarray(self.h[row, self.mask[row]], dtype=np.float32)


class DiscreteCodes:
    """Read only the published integer interface, never Phi fitting internals or h/z."""

    def __init__(self, run_dir, split, source_rows):
        import json
        root = Path(run_dir)
        self.metadata = json.loads((root / "codebook.json").read_text())
        folder = root / split
        self.codes = np.load(folder / "codes.npy", mmap_mode="r")
        self.mask = np.load(folder / "mask.npy", mmap_mode="r")
        exported = np.load(folder / "source_row_idx.npy")
        order = np.argsort(exported)
        self.rows = order[np.searchsorted(exported[order], source_rows)]
        if not np.array_equal(exported[self.rows], source_rows):
            raise ValueError("Discrete code source rows do not match the requested samples")

    def __getitem__(self, index):
        row = self.rows[index]
        codes = self.codes[row, self.mask[row]].copy()
        codes[:, 1] += self.metadata["coarse_count"]
        return codes.reshape(-1).astype(np.int64)


def epoch_batches(lengths, batch_size, seed, epoch=0, method="bucket"):
    """Shared global batches: shuffle, locally bucket by *full* history length."""
    rng = np.random.default_rng(seed + epoch)
    order = rng.permutation(len(lengths))
    if method == "shuffle":
        return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if method != "bucket":
        raise ValueError(f"Unknown batching method: {method}")
    window = batch_size * 32
    chunks = []
    for start in range(0, len(order), window):
        indices = order[start:start + window]
        indices = indices[np.argsort(lengths[indices], kind="stable")]
        chunks.extend(indices[i:i + batch_size] for i in range(0, len(indices), batch_size))
    return [chunks[i] for i in rng.permutation(len(chunks))]
