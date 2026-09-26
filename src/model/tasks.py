"""Task-neutral reader for the prepared recommendation datasets.

One row of a task's window, per channel, chronological, with the history view applied
(``TaskSamples``).  The on-disk layout it reads is the one written by ``src.data.prepare``.
"""

from pathlib import Path

import numpy as np


class TaskSamples:
    def __init__(self, root, task, split, dataset):
        """Read one ``<task>/<split>.bin`` through the dataset entry point.

        ``root`` is the pipeline root (``output/data``). ``dataset`` names the dataset
        explicitly; by default the owner declared in ``<root>/tasks.json`` is used.
        """
        from src.data.dataset import open_dataset

        name = "train" if split == "validation" else split
        self.root = Path(root) / task / name
        self.split = open_dataset(root, task, dataset).split(task, name)
        self.channels = self.split.channels
        self.histories = {channel: self.split.history(channel) for channel in self.channels}
        self.lengths = {channel: self.split.lengths(channel) for channel in self.channels}
        self.target = self.split.target
        self.target_lengths = self.split.target_length
        self.source_rows = self.split.source_row_idx
        self.uid = self.split.uid

    def __len__(self):
        return len(self.source_rows)

    def index_for_sources(self, source_rows):
        order = np.argsort(self.source_rows)
        positions = order[np.searchsorted(self.source_rows[order], source_rows)]
        if not np.array_equal(self.source_rows[positions], source_rows):
            raise ValueError("Prepared rows and canonical data rows do not match")
        return positions

    def history_rows(self, index, mode, recent, split=None):
        """Return one chronological sequence; ``recent`` keeps only its last ``recent`` items.

        ``split`` (channel -> budget) keeps the *total* budget unchanged but draws each
        channel's own tail, so a two-channel task no longer spends the whole window on the
        channel that happens to sit last in the merge order.

        A two-channel task is a single sequence, not per-channel windows, so every task feeds
        the same recent view.
        """
        rows = []
        for name in self.channels:
            length = int(self.lengths[name][index])
            items = self.histories[name][index, :length]
            if mode == "recent" and split:
                budget = int(split.get(name, 0))
                items = items[-budget:] if budget > 0 else items[:0]
            rows.extend((name, int(item)) for item in items)
        if mode != "recent":
            return rows
        if split:
            return rows
        return rows[-recent:] if recent > 0 else []

    def target_rows(self, index):
        length = int(self.target_lengths[index])
        return self.target[index, :length]
