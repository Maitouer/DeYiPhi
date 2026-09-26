import unittest

import numpy as np
import pyarrow.parquet as pq

from src.model.metrics import ranking_metrics, recommendation_metrics


class RankingTests(unittest.TestCase):
    def test_distinct_hits_keep_original_ranks(self):
        result = ranking_metrics([[9, 9, -1, 2, 8]], [[9, 9, 2, 5]])
        self.assertEqual(result["pass@1"][0], 1)
        self.assertEqual(result["recall@1"][0], 1 / 3)
        self.assertEqual(result["recall@5"][0], 2 / 3)
        expected = (1 + 1 / np.log2(5)) / (1 + 1 / np.log2(3) + 1 / np.log2(4))
        self.assertAlmostEqual(result["ndcg@5"][0], expected)
        self.assertTrue(all(0 <= value[0] <= 1 for value in result.values()))

    def test_zero_is_a_valid_sid_and_empty_predictions_count_zero(self):
        result = ranking_metrics([[0], [], [3]], [[0], [3], []])
        np.testing.assert_array_equal(result["recall@1"], [1, 0, 0])
        np.testing.assert_array_equal(result["ndcg@32"], [1, 0, 0])

    def test_thirty_metrics_and_k20_boundary(self):
        predictions = {space: [[-1] * 14 + [3]] for space in ("sid", "pid")}
        result = recommendation_metrics(predictions, {"sid": [[3]], "pid": [[3]]}, (1, 5, 10, 20, 32))
        self.assertEqual(len(result), 30)
        self.assertEqual(result["sid/pass@10"][0], 0)
        self.assertEqual(result["pid/pass@20"][0], 1)


if __name__ == "__main__":
    unittest.main()
