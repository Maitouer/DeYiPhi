"""Head-tail predictive statistics against the exact geometry (``03_phi.md`` §6-§8).

The estimator is the one place where phi trades exactness for cost, so the tests here check the
properties that make the trade measurable: the head is an exact lower bound of the partition, the
tail is what closes the gap, head membership is resolved exactly, and the calibration report says
how far the estimate is from a full-catalog scan.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.model.phi import estimator
from src.model.phi.items import CatalogUnits, TaskCatalog

TEMPERATURE = 0.15


class _FakeItems:
    """A tiny item table: ``p(i)`` rows, unit-normalised on read exactly like ``FrozenItems``."""

    def __init__(self, matrix):
        self.embeddings = np.asarray(matrix, dtype=np.float32)
        self.rows, self.dim = self.embeddings.shape

    def unit(self, rows, *, device=None, dtype=torch.float32):
        values = np.asarray(rows, dtype=np.int64).reshape(-1)
        out = torch.from_numpy(np.asarray(self.embeddings[values], dtype=np.float32))
        out = torch.nn.functional.normalize(out, dim=-1)
        return out if device is None else out.to(device=device, dtype=dtype)


def synthetic_catalog(items: int = 4096, dim: int = 32, seed: int = 11):
    matrix = torch.randn(items, dim, generator=torch.Generator().manual_seed(seed))
    matrix = torch.nn.functional.normalize(matrix, dim=-1).numpy()
    fake = _FakeItems(matrix)
    return fake, TaskCatalog.from_rows(fake, np.arange(items), lam=0.5)


def synthetic_queries(count: int = 24, dim: int = 32, seed: int = 12):
    values = torch.randn(count, dim, generator=torch.Generator().manual_seed(seed))
    return torch.nn.functional.normalize(values, dim=-1)


class HeadTailTest(unittest.TestCase):
    def setUp(self):
        self.items, self.catalog = synthetic_catalog()
        self.units = CatalogUnits(self.catalog, device=torch.device("cpu"))
        self.queries = synthetic_queries()
        self.device = torch.device("cpu")

    def _exact(self):
        return estimator.exact(self.queries, self.units, self.catalog, TEMPERATURE,
                               item_block=512, query_block=8, device=self.device,
                               dtype=torch.float32)

    def test_head_is_an_exact_lower_bound_of_the_partition(self):
        log_exact, _moment = self._exact()
        index, _score = estimator.head_search(self.queries, self.units, head=128, item_block=512,
                                              query_block=8, device=self.device,
                                              dtype=torch.float32)
        rows = self.units.gather(index).float()
        floor = torch.logsumexp(
            (self.queries @ rows.transpose(1, 2)) / TEMPERATURE, dim=-1)
        self.assertTrue(bool((floor <= log_exact + 1e-4).all()),
                        "a head-only partition must not exceed the exact one")

    def test_head_tail_matches_the_exact_geometry(self):
        log_exact, moment_exact = self._exact()
        statistics = estimator.estimate(
            self.queries, self.units, self.catalog, TEMPERATURE, head=256, tail=1024,
            item_block=512, query_block=8, device=self.device, dtype=torch.float32,
            generator=torch.Generator().manual_seed(3))
        report = estimator.calibration_report(statistics.log_partition, statistics.moment,
                                              log_exact, moment_exact)
        self.assertLess(report["partition_error"]["mean"], 0.05)
        # The moment direction is the quantity the codebook actually consumes, so it is the one
        # that has to be tight on average; a single state out of 24 may still sit off-axis when
        # the head happens to cover little of its mass.
        self.assertGreater(report["moment_cosine"]["mean"], 0.98)
        self.assertGreater(report["moment_cosine"]["min"], 0.95)
        self.assertLess(report["moment_relative_error"]["mean"], 0.2)
        self.assertGreater(float(statistics.head_mass.min()), 0.0)
        self.assertGreater(float(statistics.tail_ess.min()), 1.0)

    def test_a_wider_estimator_is_closer_to_the_exact_geometry(self):
        log_exact, moment_exact = self._exact()
        narrow = estimator.estimate(
            self.queries, self.units, self.catalog, TEMPERATURE, head=128, tail=512,
            item_block=512, query_block=8, device=self.device, dtype=torch.float32,
            generator=torch.Generator().manual_seed(3))
        wide = estimator.estimate(
            self.queries, self.units, self.catalog, TEMPERATURE, head=512, tail=2048,
            item_block=512, query_block=8, device=self.device, dtype=torch.float32,
            generator=torch.Generator().manual_seed(3))
        narrow_report = estimator.calibration_report(narrow.log_partition, narrow.moment,
                                                     log_exact, moment_exact)
        wide_report = estimator.calibration_report(wide.log_partition, wide.moment,
                                                   log_exact, moment_exact)
        self.assertLess(wide_report["partition_error"]["mean"],
                        narrow_report["partition_error"]["mean"])
        self.assertGreater(wide_report["moment_cosine"]["mean"],
                           narrow_report["moment_cosine"]["mean"])

    def test_head_membership_is_resolved_exactly(self):
        head = torch.tensor([[3, 1, 7], [5, 5, 9]])
        tail = torch.tensor([[1, 2, 9], [9, 8, 5]])
        keep = estimator._tail_membership(head, tail)
        # 1 is row 0's head (dropped), 2 and 9 are not; 9 and 5 are row 1's head, 8 is not.
        expected = torch.tensor([[False, True, True], [False, True, False]])
        self.assertTrue(torch.equal(keep, expected))

    def test_refine_reruns_only_the_failing_states(self):
        statistics = estimator.estimate(
            self.queries, self.units, self.catalog, TEMPERATURE, head=64, tail=256,
            item_block=512, query_block=8, device=self.device, dtype=torch.float32,
            generator=torch.Generator().manual_seed(5))
        statistics.head_mass_floor = 0.99            # force every state into the fallback
        statistics.tail_ess_floor = 1e9
        subset = torch.arange(self.queries.shape[0])
        estimator.refine(self.queries, self.units, self.catalog, TEMPERATURE, statistics, subset,
                         rounds=1, factor=2.0, item_block=512, query_block=8, device=self.device,
                         dtype=torch.float32, generator=torch.Generator().manual_seed(6))
        self.assertEqual(len(statistics.fallback), 1)
        self.assertEqual(statistics.fallback[0]["states"], int(subset.numel()))
        self.assertGreater(statistics.fallback[0]["tail"], 256)
        self.assertGreater(statistics.fallback[0]["head"], 64)

    def test_assignment_agreement_is_reported_from_two_geometries(self):
        exact = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0]])
        approximate = torch.tensor([[0.1, 0.9, 2.0], [0.0, 2.0, 1.0]])
        agreement = estimator.assignment_agreement(approximate, exact)
        self.assertEqual(agreement["top1"], 0.5)
        self.assertEqual(agreement["top2"], 1.0)


if __name__ == "__main__":
    unittest.main()
