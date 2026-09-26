"""Reservoir weighting, artifact layout, reuse guards and config overrides.

The reservoir is where user balance is enforced, and the manifest is what stops a stage from
reading an artifact of a different teacher, so both are contract tests rather than sanity checks.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import torch

from src.model.phi import artifacts as phi_artifacts
from src.model.phi import config as phi_config
from src.model.phi.reservoir import code_weights, sample

DEYI_YAML = """
seed: 2026
task: product
recent:
  short_video: 32
  product: 16
  ad: 16
max_history:
  short_video: 512
  product: 100
  ad: 200
log_root: log/model/deyi
data:
  root: output/data
  target: 10
tasks:
  product:
    dataset: single_channel
    target: product
pca:
  dim: 1024
  samples: 0
deyi:
  output_root: output/model/deyi
  item_space: output/model/deyi/item_space
  state_dim: 1024
  num_interests: 4
  history_layers: 2
  interest_layers: 2
  heads: 16
  ffn: 2048
  dropout: 0.1
  temporal_buckets: 64
  behavior_dim: 5
  route_dim: 128
  route_temperature: 0.5
  item_temperature: 0.15
  lambda_user: 0.1
  lambda_div: 0.01
  negatives: 2048
  user_negatives: 8
  epochs: 20
  batch_size: 512
  micro_batch_size: 256
  learning_rate: 0.00002
  weight_decay: 0.01
  validation_fraction: 0.05
  min_epochs: 3
  early_stop_patience: 1
  load_workers: 1
"""

PHI_YAML = """
deyi_config: {deyi}
output_dir: output/model/phi
vocabulary: 256
reservoir:
  size: 4096
  seed: 2026
estimator:
  head: 256
  tail: 1024
  proposal_lambda: 0.5
  item_block: 131072
  query_block: 8192
  resident: auto
  calibration: 256
  head_mass_floor: 0.5
  tail_ess_floor: 64.0
  fallback_rounds: 2
  fallback_factor: 2.0
  fallback_max_fraction: 0.05
  moment_block: 1024
  device: auto
codebook:
  iterations: 25
  seed: 2026
  exact_normalizer: true
  held_out_fraction: 0.2
router:
  hidden: 256
  layers: 2
  epochs: 80
  batch_size: 4096
  learning_rate: 0.001
  weight_decay: 0.0
  validation_fraction: 0.1
  seed: 2026
tokenize:
  batch_size: 8192
route:
  batch_size: 2048
"""


class ReservoirTest(unittest.TestCase):
    def test_weights_balance_users_and_zero_masked_slots(self):
        mask = torch.tensor([[True, True, True, True], [True, False, False, False],
                             [True, True, True, True]])
        uid = torch.tensor([7, 7, 9])                     # user 7 owns two rows
        weight = code_weights(uid, mask)
        self.assertAlmostEqual(float(weight[0].sum()), 0.5, places=6)
        self.assertAlmostEqual(float(weight[1].sum()), 0.5, places=6)
        self.assertAlmostEqual(float(weight[2].sum()), 1.0, places=6)
        self.assertEqual(float(weight[1, 1:].sum()), 0.0)
        # A four-interest row splits its user's unit mass evenly.
        self.assertAlmostEqual(float(weight[0, 0]), 0.125, places=6)

    def test_sampling_is_deterministic_and_respects_the_weights(self):
        weight = torch.zeros((4, 4), dtype=torch.float64)
        weight[0, 0] = 1.0
        weight[3, 2] = 1.0
        first = sample(weight, 2, torch.Generator().manual_seed(11))
        again = sample(weight, 2, torch.Generator().manual_seed(11))
        self.assertTrue(torch.equal(first, again))
        self.assertEqual(sorted(first.tolist()), [0, 14])
        # Zero-weight slots can never be drawn, whatever the requested size.
        self.assertEqual(sample(weight, 2, torch.Generator().manual_seed(3)).numel(), 2)

    def test_sampling_cannot_exceed_the_available_mass(self):
        weight = torch.zeros((2, 2), dtype=torch.float64)
        weight[0, 0] = 1.0
        drawn = sample(weight, 16, torch.Generator().manual_seed(1))
        self.assertEqual(drawn.numel(), 1)


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.environ["PHI_OUTPUT_DIR"] = self.tmp.name
        self.cfg = self._config()

    def _config(self):
        base = Path(self.tmp.name) / "config"
        base.mkdir(parents=True, exist_ok=True)
        deyi = base / "deyi.yaml"
        deyi.write_text(DEYI_YAML, encoding="utf-8")
        phi = base / "phi.yaml"
        phi.write_text(PHI_YAML.format(deyi=deyi), encoding="utf-8")
        return phi_config.load(str(phi))

    def test_arm_root_carries_dataset_task_and_teacher(self):
        root = phi_artifacts.arm_root(self.cfg)
        self.assertEqual(root, Path(self.tmp.name) / "single_channel" / "product" / "k4_r16")
        self.assertEqual(phi_artifacts.arm_name(self.cfg), "k4_r16")
        self.assertEqual(phi_artifacts.token_name(3), "<|g_3|>")
        self.assertEqual(phi_artifacts.TOKEN_NONE, "<|g_none|>")
        self.assertEqual(len(phi_artifacts.token_names(4)), 4)

    def test_reuse_requires_both_the_identity_and_the_file(self):
        identity = {"a": 1}
        self.assertIsNone(phi_artifacts.reuse(self.cfg, "estimate", identity))
        phi_artifacts.save_pt({"ok": True}, phi_artifacts.stats(self.cfg))
        phi_artifacts.record(self.cfg, "estimate", identity, {"summary": 1})
        self.assertEqual(phi_artifacts.reuse(self.cfg, "estimate", identity,
                                            phi_artifacts.stats(self.cfg)), {"summary": 1})
        self.assertIsNone(phi_artifacts.reuse(self.cfg, "estimate", {"a": 2},
                                              phi_artifacts.stats(self.cfg)))
        phi_artifacts.stats(self.cfg).unlink()
        self.assertIsNone(phi_artifacts.reuse(self.cfg, "estimate", identity,
                                              phi_artifacts.stats(self.cfg)))

    def test_manifest_remembers_the_arm(self):
        phi_artifacts.record(self.cfg, "reservoir", {"size": 1}, {})
        manifest = phi_artifacts.read_manifest(self.cfg)
        self.assertEqual(manifest["arm"]["dataset"], "single_channel")
        self.assertEqual(manifest["arm"]["num_interests"], 4)
        self.assertEqual(manifest["arm"]["recent"], 16)

    def test_env_overrides_win_over_the_file(self):
        os.environ["PHI_VOCABULARY"] = "64"
        os.environ["PHI_RESERVOIR_SIZE"] = "2048"
        os.environ["PHI_DEYI_K"] = "8"
        os.environ["PHI_DEYI_RECENT"] = "32"
        self.addCleanup(os.environ.pop, "PHI_VOCABULARY", None)
        self.addCleanup(os.environ.pop, "PHI_RESERVOIR_SIZE", None)
        self.addCleanup(os.environ.pop, "PHI_DEYI_K", None)
        self.addCleanup(os.environ.pop, "PHI_DEYI_RECENT", None)
        cfg = self._config()
        self.assertEqual(int(cfg.phi.vocabulary), 64)
        self.assertEqual(int(cfg.phi.reservoir.size), 2048)
        self.assertEqual(phi_artifacts.arm_name(cfg), "k8_r32")
        self.assertEqual(int(cfg.deyi.num_interests), 8)
        # ``PHI_DEYI_RECENT`` replaces this task's entry in the recent table; the scalar the phi
        # paths key on is the mirror of that entry, not a second place to write the setting.
        self.assertEqual(int(cfg.recent[cfg.task]), 32)
        self.assertEqual(int(cfg.deyi.recent), 32)

    def test_validation_rejects_inconsistent_settings(self):
        cfg = self._config()
        cfg.phi.vocabulary = 99999
        with self.assertRaises(ValueError):
            phi_config.validate(cfg)
        cfg = self._config()
        cfg.phi.estimator.proposal_lambda = 1.5
        with self.assertRaises(ValueError):
            phi_config.validate(cfg)
        cfg = self._config()
        cfg.phi.router.layers = 1
        with self.assertRaises(ValueError):
            phi_config.validate(cfg)


if __name__ == "__main__":
    unittest.main()
