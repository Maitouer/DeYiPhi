"""Router distillation, full-state tokenization and the route posterior.

These are the three places where a mistake is quiet: a router that memorises, a tokenizer that
leaks the mask, or a route that ignores duplicated tokens all produce plausible-looking artifacts.
"""

from __future__ import annotations

import unittest

import torch

from src.model.phi import codebook as codebook_module
from src.model.phi import route as route_module
from src.model.phi import router as router_module
from src.model.phi import tokenize as tokenize_module

TEMPERATURE = 0.15


def separable_states(clusters: int = 4, per_cluster: int = 64, dim: int = 16, seed: int = 9):
    """States drawn around orthogonal directions: a router either learns this or it is broken."""
    generator = torch.Generator().manual_seed(seed)
    axes = torch.eye(clusters, dim)
    states = axes.repeat_interleave(per_cluster, dim=0)
    states = states + 0.05 * torch.randn(states.shape, generator=generator)
    return torch.nn.functional.normalize(states, dim=-1), torch.arange(clusters).repeat_interleave(
        per_cluster)


def statistics_for(states: torch.Tensor):
    """Sufficient statistics of a state set over itself (a self-consistent tiny catalog)."""
    logits = states @ states.T / TEMPERATURE
    log_partition = torch.logsumexp(logits, dim=-1)
    moment = torch.softmax(logits, dim=-1) @ states
    return log_partition, moment


class RouterTest(unittest.TestCase):
    def test_distillation_recovers_separable_labels(self):
        states, label = separable_states()
        log_partition, moment = statistics_for(states)
        weight = torch.full((states.shape[0],), 1.0 / states.shape[0], dtype=torch.float64)
        book = codebook_module.Codebook(medoid=torch.arange(4), center=states[[0, 64, 128, 192]],
                                        log_partition=log_partition[[0, 64, 128, 192]],
                                        label=label)
        held = torch.zeros(states.shape[0], dtype=torch.bool)
        held[::4] = True
        state = router_module.train(states, label, weight, dim=states.shape[1], size=4,
                                    hidden=32, epochs=60, batch_size=128, learning_rate=5e-3,
                                    seed=1, validation=held, moment=moment, reference=book,
                                    temperature=TEMPERATURE, device=torch.device("cpu"))
        self.assertGreater(state.validation["top1_agreement"], 0.9)
        self.assertLess(state.validation["regret"]["mean"], 1e-6)
        self.assertIn("state_dict", state.__dict__)
        module = state.build()
        output = router_module.logits(module, states)
        self.assertGreater(float((output.argmax(dim=-1) == label).float().mean()), 0.9)

    def test_tokenize_leaves_masked_slots_unset(self):
        states, label = separable_states()
        module = router_module.MLP(states.shape[1], 32, 4, 2)
        mask = torch.ones((8, 3), dtype=torch.bool)
        mask[0, 1:] = False
        mask[3, 0] = False

        class _Arm:
            def payload(self, split):
                return {"states": states[:8].unsqueeze(1).repeat(1, 3, 1), "mask": mask,
                        "source_row_idx": torch.arange(8)}

        result = tokenize_module.run(_Arm(), module, "train", vocabulary=4, batch=4,
                                     device=torch.device("cpu"))
        self.assertTrue(bool((result["codes"][~mask] == -1).all()))
        self.assertTrue(bool((result["codes"][mask] >= 0).all()))
        self.assertEqual(float(result["confidence"][~mask].max()), 0.0)


class RouteTest(unittest.TestCase):
    def test_duplicated_tokens_aggregate_their_weight(self):
        alpha = torch.tensor([[0.5, 0.3, 0.2]])
        codes = torch.tensor([[2, 0, 2]])
        log_alpha, present = route_module.aggregate_alpha(codes, alpha, 4)
        self.assertAlmostEqual(float(log_alpha.exp()[0, 2]), 0.7, places=6)
        self.assertAlmostEqual(float(log_alpha.exp()[0, 0]), 0.3, places=6)
        self.assertFalse(bool(present[0, 1]))
        self.assertFalse(bool(present[0, 3]))
        self.assertAlmostEqual(float(log_alpha.exp().sum()), 1.0, places=6)

    def test_masked_slots_contribute_nothing(self):
        alpha = torch.tensor([[0.5, 0.5]])
        codes = torch.tensor([[1, -1]])
        log_alpha, present = route_module.aggregate_alpha(codes, alpha, 3)
        self.assertAlmostEqual(float(log_alpha.exp()[0, 1]), 0.5, places=6)
        self.assertTrue(torch.isinf(log_alpha[0, 0]))      # an absent token carries no mass
        self.assertEqual(float(log_alpha.exp()[0, 0]), 0.0)
        self.assertFalse(bool(present[0, 2]))

    def test_posterior_prefers_the_token_that_explains_the_target(self):
        target = torch.tensor([[[1.0, 0.0]]])
        center = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        normaliser = torch.zeros(2, dtype=torch.float64)
        log_alpha = torch.log(torch.tensor([[0.5, 0.5]], dtype=torch.float64))
        present = torch.ones((1, 2), dtype=torch.bool)
        mask = torch.ones((1, 1), dtype=torch.bool)
        probability = route_module.posterior(target, center, normaliser, log_alpha, present,
                                             TEMPERATURE, mask)
        self.assertEqual(int(probability.argmax(dim=-1)), 0)
        self.assertGreater(float(probability[0, 0, 0]), 0.99)

    def test_a_row_without_any_state_has_no_candidate_token(self):
        """The pivot that keeps ``posterior`` finite must never become an assigned route."""
        codes = torch.full((1, 3), -1)
        alpha = torch.zeros((1, 3))
        log_alpha, present = route_module.aggregate_alpha(codes, alpha, 2)
        self.assertFalse(bool(present.any()))
        # The caller's own guard: no candidate token -> no route, whatever softmax returned.
        target = torch.tensor([[[1.0, 0.0]]])
        center = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        normaliser = torch.zeros(2, dtype=torch.float64)
        probability = route_module.posterior(target, center, normaliser, log_alpha, present,
                                             TEMPERATURE, torch.ones((1, 1), dtype=torch.bool))
        self.assertTrue(bool(torch.isfinite(probability).all()))
        self.assertTrue(bool((~present.any(dim=-1)).all()))

    def test_posterior_entropy_is_finite_when_tokens_are_absent(self):
        """Absent tokens are exactly zero, and ``0 * log 0`` must not become NaN."""
        target = torch.tensor([[[1.0, 0.0]]])
        center = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        normaliser = torch.zeros(2, dtype=torch.float32)
        log_alpha = torch.log(torch.tensor([[1.0, 0.5]], dtype=torch.float32))
        present = torch.tensor([[True, False]])
        probability = route_module.posterior(target, center, normaliser, log_alpha, present,
                                             TEMPERATURE, torch.ones((1, 1), dtype=torch.bool))
        self.assertEqual(float(probability[0, 0, 1]), 0.0)
        entropy = -torch.special.xlogy(probability, probability).sum(-1)
        self.assertTrue(bool(torch.isfinite(entropy).all()))
        self.assertAlmostEqual(float(entropy), 0.0, places=6)

    def test_unavailable_tokens_and_targets_are_excluded(self):
        target = torch.tensor([[[1.0, 0.0]]])
        center = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        normaliser = torch.zeros(2, dtype=torch.float64)
        log_alpha = torch.log(torch.tensor([[1.0, 0.5]], dtype=torch.float64))
        present = torch.tensor([[True, False]])
        mask = torch.tensor([[False]])
        probability = route_module.posterior(target, center, normaliser, log_alpha, present,
                                             TEMPERATURE, mask)
        self.assertTrue(bool(torch.isfinite(probability).all()))
        self.assertAlmostEqual(float(probability.sum()), 0.0, places=6)
        # With only the first token available, the mass has to sit there.
        present = torch.tensor([[True, False]])
        mask = torch.tensor([[True]])
        probability = route_module.posterior(target, center, normaliser, log_alpha, present,
                                             TEMPERATURE, mask)
        self.assertAlmostEqual(float(probability[0, 0, 0]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
