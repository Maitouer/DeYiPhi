"""The predictive-KL medoid codebook (``02_phi.md`` §4.5, ``03_phi.md`` §9).

The identity ``D_KL(p_s || p_c) = C_s + A(c) - c^T mu_s / tau_p`` is what lets phi quantize
*predictions* on cached sufficient statistics instead of on the catalog, so it is checked here
against a brute-force KL on a tiny explicit catalog rather than trusted.
"""

from __future__ import annotations

import unittest

import torch

from src.model.phi import codebook as codebook_module

TEMPERATURE = 0.15


def explicit_statistics(items: int = 6, dim: int = 3, count: int = 12, seed: int = 5):
    """The exact ``(z, A, mu)`` triple of a handful of states over a tiny explicit catalog."""
    generator = torch.Generator().manual_seed(seed)
    catalog = torch.nn.functional.normalize(torch.randn(items, dim, generator=generator), dim=-1)
    states = torch.nn.functional.normalize(torch.randn(count, dim, generator=generator), dim=-1)
    logits = states @ catalog.T / TEMPERATURE
    log_partition = torch.logsumexp(logits, dim=-1)
    probability = torch.softmax(logits, dim=-1)
    moment = probability @ catalog
    return catalog, states, log_partition, moment, probability


class MedoidCodebookTest(unittest.TestCase):
    def test_distance_reproduces_the_exact_kl_up_to_the_state_constant(self):
        catalog, states, log_partition, moment, probability = explicit_statistics()
        # p_c for the codeword state t: the codeword is a state, not an arbitrary prototype.
        cross = moment @ states.T
        distance = log_partition[None, :] - cross / TEMPERATURE
        for source in range(states.shape[0]):
            constant = (states[source] @ moment[source]) / TEMPERATURE - log_partition[source]
            for codeword in range(states.shape[0]):
                target = probability[codeword]
                kl = float((probability[source] * (probability[source].log()
                                                    - target.log())).sum())
                self.assertAlmostEqual(float(distance[source, codeword]),
                                       kl - float(constant), places=5)

    def test_fit_keeps_codewords_on_real_states_and_lowers_the_objective(self):
        _catalog, states, log_partition, moment, _probability = explicit_statistics(count=32)
        weight = torch.full((32,), 1.0 / 32, dtype=torch.float64)
        book = codebook_module.fit(states, moment, log_partition, weight, 4, TEMPERATURE,
                                   iterations=8, seed=7)
        self.assertEqual(tuple(book.center.shape), (4, states.shape[1]))
        for index in range(book.size):
            self.assertTrue(torch.equal(book.center[index], states[book.medoid[index]]),
                            "a codeword must be a real DeYi state, not a free prototype")
            self.assertEqual(float(book.log_partition[index]), float(log_partition[book.medoid[index]]))
        objectives = [entry["objective"] for entry in book.history]
        # Assign and medoid update are both exact minimisers of the same objective, so the
        # recorded sequence is Lloyd-monotone; a wrong cross term (mu_t instead of z_t) shows up
        # here as an ascent rather than as a slightly worse codebook.
        for earlier, later in zip(objectives, objectives[1:]):
            self.assertLessEqual(later, earlier + 1e-9,
                                 "the medoid alternation must not increase the objective")
        self.assertLessEqual(objectives[-1], objectives[0] + 1e-9)

    def test_medoid_update_minimises_the_cluster_objective(self):
        """The closed-form update must agree with a brute-force scan over the cluster's states."""
        _catalog, states, log_partition, moment, _probability = explicit_statistics(count=24)
        weight = torch.full((24,), 1.0 / 24, dtype=torch.float64)
        book = codebook_module.fit(states, moment, log_partition, weight, 3, TEMPERATURE,
                                   iterations=1, seed=11)
        for cluster in range(book.size):
            members = torch.nonzero(book.label == cluster, as_tuple=False).squeeze(1)
            if members.numel() == 0:
                continue
            total = (weight[members].double()[:, None] * moment[members].double()).sum(0)
            mass = weight[members].double().sum()
            cost = (mass * log_partition[members].double()
                    - states[members].double() @ total / TEMPERATURE)
            self.assertEqual(int(members[int(cost.argmin())]), int(book.medoid[cluster]))
        self.assertEqual(int(book.label.min()), 0)
        self.assertLess(int(book.label.max()), 4)

    def test_more_codewords_never_hurt_the_fit_objective(self):
        _catalog, states, log_partition, moment, _probability = explicit_statistics(count=32)
        weight = torch.full((32,), 1.0 / 32, dtype=torch.float64)
        coarse = codebook_module.fit(states, moment, log_partition, weight, 2, TEMPERATURE,
                                     iterations=8, seed=7)
        fine = codebook_module.fit(states, moment, log_partition, weight, 4, TEMPERATURE,
                                   iterations=8, seed=7)
        self.assertLessEqual(fine.history[-1]["objective"], coarse.history[-1]["objective"] + 1e-9)

    def test_many_clusters_stay_occupied_and_the_fit_stops_churning(self):
        """A vocabulary larger than the distinct geometry must not loop on empty clusters."""
        _catalog, states, log_partition, moment, _probability = explicit_statistics(count=64)
        weight = torch.full((64,), 1.0 / 64, dtype=torch.float64)
        book = codebook_module.fit(states, moment, log_partition, weight, 16, TEMPERATURE,
                                   iterations=25, seed=5)
        self.assertLessEqual(len(book.history), 25)          # it terminates
        self.assertGreater(int(torch.unique(book.label).numel()), 1)
        # The returned codebook is the best configuration visited, never a churned one.
        scores = codebook_module.distance(moment, book.center, book.log_partition, TEMPERATURE)
        label = scores.argmin(dim=1)
        final = float((scores.gather(1, label[:, None]).squeeze(1).double()
                       * weight.double()).sum())
        self.assertLessEqual(final, min(entry["updated"] for entry in book.history) + 1e-9)
        # Empty-cluster restarts are recorded, so a churning fit is visible in the artifact.
        self.assertIn("restarts", book.history[-1])

    def test_diagnostics_report_sharing_and_dead_codes(self):
        _catalog, states, log_partition, moment, _probability = explicit_statistics(count=48)
        weight = torch.full((48,), 1.0 / 48, dtype=torch.float64)
        uid = torch.arange(48) % 6
        book = codebook_module.fit(states, moment, log_partition, weight, 8, TEMPERATURE,
                                   iterations=6, seed=3)
        report = codebook_module.diagnostics(book, states, moment, log_partition, weight, uid,
                                             TEMPERATURE)
        for key in ("predictive_kl", "usage", "support", "dead_codes", "history"):
            self.assertIn(key, report)
        self.assertGreaterEqual(report["predictive_kl"]["mean"], 0.0)
        self.assertLessEqual(report["usage"]["max_share"], 1.0)
        self.assertGreaterEqual(report["support"]["uid_min"], 0)

    def test_state_split_is_by_user(self):
        uid = torch.arange(100) % 10
        fit, held = codebook_module.held_out_split(uid, 0.2, 2026)
        self.assertEqual(int((fit | held).sum()), 100)
        self.assertEqual(int((fit & held).sum()), 0)
        self.assertGreater(int(held.sum()), 0)
        users_fit = set(uid[fit].tolist())
        users_held = set(uid[held].tolist())
        self.assertFalse(users_fit & users_held, "a held-out read must not share users")


if __name__ == "__main__":
    unittest.main()
