import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from src.model.tiger import runtime_config, validate_parallelism
from src.model.tiger.data import _compact_history_rows
from src.model.tiger.train import microbatches


class _Samples:
    channels = ("video", "ad")
    histories = {
        "video": np.asarray([[1, 2, 0], [3, 0, 0], [4, 5, 6]], dtype=np.int64),
        "ad": np.asarray([[7, 0], [8, 9], [0, 0]], dtype=np.int64),
    }
    lengths = {
        "video": np.asarray([2, 1, 3], dtype=np.int64),
        "ad": np.asarray([1, 2, 0], dtype=np.int64),
    }


class TigerEfficiencyTests(unittest.TestCase):
    def test_vectorized_history_matches_chronological_channel_concatenation(self):
        samples = _Samples()
        rows, lengths = _compact_history_rows(samples, [2, 0, 1], "full", 32)
        np.testing.assert_array_equal(lengths, [3, 3, 3])
        np.testing.assert_array_equal(rows, [[4, 5, 6], [1, 2, 7], [3, 8, 9]])

    def test_vectorized_recent_uses_the_combined_history_tail(self):
        rows, lengths = _compact_history_rows(_Samples(), [0, 1, 2], "recent", 2)
        np.testing.assert_array_equal(lengths, [2, 2, 2])
        np.testing.assert_array_equal(rows, [[2, 7], [8, 9], [5, 6]])

    def test_four_rank_global_batch_has_no_duplicate_work(self):
        indices = np.arange(128)
        all_rows = []
        for rank in range(4):
            batches = microbatches(indices, rank, 4, 32)
            self.assertEqual(len(batches), 1)
            rows, valid = batches[0]
            self.assertTrue(valid.all())
            all_rows.extend(rows.tolist())
        self.assertCountEqual(all_rows, indices.tolist())

    def test_cuda_topology_matches_the_requested_world_size(self):
        cfg = SimpleNamespace(runtime=SimpleNamespace(nproc=8, device="cuda"))
        validate_parallelism(cfg, 8)
        with self.assertRaises(ValueError):
            validate_parallelism(cfg, 4)

    def test_topology_derives_microbatch_without_changing_global_batch(self):
        self.assertEqual(runtime_config(4, 128).micro_batch_size, 32)
        self.assertEqual(runtime_config(8, 128).micro_batch_size, 16)
        with self.assertRaises(ValueError):
            runtime_config(3, 128)

    def test_cuda_topology_must_use_every_allocated_gpu(self):
        cfg = SimpleNamespace(runtime=SimpleNamespace(nproc=4, device="cuda"))
        with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}):
            with self.assertRaisesRegex(ValueError, "must match"):
                validate_parallelism(cfg, 4)


if __name__ == "__main__":
    unittest.main()
