import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.model.tiger.evaluate import (
    DevicePrefixIndex,
    PrefixIndex,
    _ranked_topk,
    evaluation_manifest,
    generate,
    generate_reference,
    publish_shard_manifest,
)


class _RandomHead(torch.nn.Module):
    def __init__(self, width, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("scores", torch.randn(16384, width, generator=generator))

    def forward(self, values):
        return self.scores[values[:, 0].long() % len(self.scores)]


class _ZeroHead(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, values):
        return values.new_zeros((len(values), self.width))


class _ToyTiger:
    def __init__(self, width, tied=False):
        head = _ZeroHead if tied else _RandomHead
        self.output_heads = [
            head(width) if tied else head(width, 100 + depth) for depth in range(4)
        ]
        self.width = width
        # Mirror ``Tiger``'s interface: ``generate`` reads ``phi`` on that object, and the
        # anchor head widths exist (unused) on the native path.
        self.phi = False
        self.anchor_widths = (width, width)

    def encode_history(self, history_raw, history_mask):
        return history_raw[:, :1, :1].float(), history_mask[:, :1]

    def decode(self, hidden, _mask, prefix):
        value = hidden[:, 0, 0].long()
        for digit in prefix.unbind(dim=1):
            value = value * self.width + digit
        return value.float().view(-1, 1, 1)

    def step_logits(self, decoded, depth):
        """``Tiger.step_logits``: logits of hierarchy ``depth`` from the last decode position."""
        return self.output_heads[depth](decoded[:, -1])


def _codes(width):
    return np.asarray(list(itertools.product(range(width), repeat=4)), dtype=np.int64)


def _batch(rows):
    history = torch.arange(rows, dtype=torch.long).view(rows, 1, 1).expand(-1, -1, 4)
    return {"history_raw": history, "history_mask": torch.ones((rows, 1), dtype=torch.bool)}


class TigerEvaluateTests(unittest.TestCase):
    def test_device_csr_matches_host_prefix_lookup(self):
        codes = _codes(4)
        index = PrefixIndex(codes, np.arange(len(codes)), 4)
        device = DevicePrefixIndex(index, torch.device("cpu"))
        roots, root_valid = device.candidates(torch.empty((3, 0), dtype=torch.long))
        for row in range(3):
            np.testing.assert_array_equal(roots[row, root_valid[row]].numpy(), index.allowed(()))
        for depth in range(1, 4):
            prefixes = torch.as_tensor(codes[::17, :depth])
            digits, valid = device.candidates(prefixes)
            for prefix, row_digits, row_valid in zip(prefixes.tolist(), digits, valid, strict=True):
                np.testing.assert_array_equal(
                    row_digits[row_valid].numpy(), index.allowed(tuple(prefix))
                )

    def test_vectorized_beam_matches_reference_without_ties(self):
        codes = _codes(8)
        index = PrefixIndex(codes, np.arange(len(codes)), 8)
        model, batch = _ToyTiger(8), _batch(5)
        expected = generate_reference(model, batch, index, 7)
        actual, scores = generate(
            model, batch, index, 7, DevicePrefixIndex(index, torch.device("cpu"))
        )
        self.assertEqual(scores.shape, (5, 7))
        for left, right in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(np.asarray(left), right)

    def test_ties_are_explicitly_parent_then_digit_ordered(self):
        codes = _codes(4)
        index = PrefixIndex(codes, np.arange(len(codes)), 4)
        actual, _scores = generate(
            _ToyTiger(4, tied=True), _batch(1), index, 3,
            DevicePrefixIndex(index, torch.device("cpu")),
        )
        np.testing.assert_array_equal(
            actual[0], np.asarray([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]])
        )

    def test_ranked_topk_uses_input_order_for_equal_scores(self):
        scores = torch.zeros((1, 4))
        values, selected, valid = _ranked_topk(scores, torch.ones_like(scores, dtype=torch.bool), 3)
        torch.testing.assert_close(values, torch.zeros((1, 3)))
        torch.testing.assert_close(selected, torch.tensor([[0, 1, 2]]))
        self.assertTrue(valid.all())

    def test_shard_manifest_rejects_a_different_model_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.pt").write_bytes(b"model")
            # ``evaluation_manifest`` uses ``cfg.evaluation.get(...)``, which a plain namespace
            # does not provide.
            cfg = OmegaConf.create({"evaluation": {"beams": 32}, "codebook": {"width": 2048}})
            manifest = evaluation_manifest(root, cfg, "test", 3)
            path = root / "shards" / "manifest.json"
            path.parent.mkdir()
            publish_shard_manifest(path, manifest, 0, 1)
            publish_shard_manifest(path, manifest, 0, 1)
            changed = dict(manifest, beams=16)
            with self.assertRaisesRegex(ValueError, "stale Tiger evaluation shards"):
                publish_shard_manifest(path, changed, 0, 1)


if __name__ == "__main__":
    unittest.main()
