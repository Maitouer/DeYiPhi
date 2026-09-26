"""The shared v7 channel-local old/recent partition (physical item rows)."""
import numpy as np

FORMAT = 'channel_window_v7'
CHANNEL = {'video': 0, 'ad': 1, 'product': 2}


def recent_budgets(lengths, budget=32):
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.shape[-1] == 1:
        return np.minimum(lengths, budget)
    if lengths.shape[-1] != 2:
        raise ValueError('v7 supports one or two history channels')
    result = np.minimum(lengths, budget // 2)
    spare = budget - result.sum(axis=-1)
    for c in range(2):
        extra = np.minimum(lengths[..., c] - result[..., c], spare)
        result[..., c] += extra
        spare -= extra
    return result


class WindowSamples:
    def __init__(self, samples):
        self.samples = samples
        self.channels = list(samples.channels)
        lengths = np.stack([samples.lengths[c] for c in self.channels], axis=1)
        self.budgets = recent_budgets(lengths)
        self.old_lengths = lengths - self.budgets
        self.slot_channels = np.repeat([CHANNEL[c] for c in self.channels], 4 // len(self.channels))
        self.manifest = {'format': FORMAT, 'recent': 32, 'channels': self.channels,
                         'slots': self.slot_channels.tolist(), 'order': 'channel_local',
                         'history_item_space': 'canonical_physical_rows'}

    def recent(self, index):
        result = []
        for c, name in enumerate(self.channels):
            lo = int(self.old_lengths[index, c])
            hi = lo + int(self.budgets[index, c])
            result.extend((name, int(x)) for x in self.samples.histories[name][index, lo:hi])
        return result

    def batch(self, indices):
        indices = np.asarray(indices)
        result = []
        for c, name in enumerate(self.channels):
            lengths = self.old_lengths[indices, c]
            width = max(1, int(lengths.max()))
            rows = np.array(self.samples.histories[name][indices, :width], dtype=np.int64)
            mask = np.arange(width)[None, :] < lengths[:, None]
            rows[~mask] = 0
            distance = self.samples.lengths[name][indices, None] - np.arange(width)[None, :]
            recency = np.minimum(np.maximum(distance, 0), 512).astype(np.int64)
            behavior = np.zeros((*rows.shape, 5), dtype=np.float32)
            # Behavior is optional in the canonical input; preserve its actual feature width.
            source = getattr(self.samples.split, 'behavior', None)
            if name == 'video' and source is not None:
                behavior = np.maximum(np.nan_to_num(np.asarray(source[indices, :width], dtype=np.float32), nan=0.0), 0)
            result.append((rows, mask, recency, behavior, CHANNEL[name]))
        return result
