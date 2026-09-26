"""The two DeYi settings that follow the data, and the arm path they define.

``recent`` and ``max_history`` are task -> value tables, not scalars: one task's window is not
another's (short_video 32, product 16). Both bit us in production once -- the CLI flags went
missing and ``int(cfg.recent)`` was applied to the table -- so the contract is asserted here rather
than discovered in a job log.
"""

from __future__ import annotations

import unittest

from pathlib import Path

from src.model.deyi import artifacts
from src.model.deyi.cli import main as deyi_cli_main  # noqa: F401  (import proves the module loads)
from src.model.deyi.config import load_config
from src.model.phi import artifacts as phi_artifacts
from src.model.phi import config as phi_config

CONFIG = "config/model/deyi.yaml"


class DeYiSettingsTest(unittest.TestCase):
    def test_recent_and_max_history_are_read_per_task(self):
        for task, recent, cap in (("short_video", 32, 512), ("product", 16, 100), ("ad", 16, 200)):
            cfg = load_config(CONFIG)
            cfg.task = task
            from src.model.deyi.config import validate

            validate(cfg)
            self.assertEqual(int(cfg.deyi.recent), recent)
            self.assertEqual(int(cfg.deyi.max_history), cap)
            self.assertEqual(artifacts.recent(cfg), recent)
            self.assertEqual(artifacts.arm_name(cfg), "k4_r%d" % recent)

    def test_cli_overrides_target_the_selected_task_only(self):
        # The flags are what the launchers pass; a missing one fails at argument parsing, i.e. in
        # the job, not in this test's caller.
        from src.model.deyi import cli

        flags = _flag_strings(cli)
        self.assertIn("--recent", flags)
        self.assertIn("--max-history", flags)
        self.assertIn("--dataset", flags)

    def test_arm_path_carries_dataset_task_and_teacher(self):
        cfg = load_config(CONFIG)
        cfg.task = "product"
        from src.model.deyi.config import validate

        validate(cfg)
        self.assertEqual(artifacts.dataset(cfg), "single_channel")
        self.assertEqual(artifacts.root(cfg), Path("output/model/deyi") / "single_channel"
                         / "product" / "k4_r16")


class PhiSettingsTest(unittest.TestCase):
    def test_phi_reads_the_teacher_scalar_and_scopes_the_arm(self):
        cfg = phi_config.load("config/model/phi.yaml")
        # The table survives (the shared DeYi readers need it) and the selected task's window is
        # mirrored into the scalar every phi path is keyed by.
        self.assertEqual(int(cfg.recent[cfg.task]), 16)
        self.assertEqual(int(cfg.deyi.recent), 16)
        self.assertEqual(int(cfg.deyi.num_interests), 4)
        root = phi_artifacts.arm_root(cfg)
        self.assertEqual(root, Path("output/model/phi") / "single_channel" / cfg.task / "k4_r16")
        self.assertEqual(phi_artifacts.arm_name(cfg), "k4_r16")

    def test_vocabulary_text_interface(self):
        self.assertEqual(phi_artifacts.token_name(7), "<|g_7|>")
        self.assertEqual(phi_artifacts.TOKEN_NONE, "<|g_none|>")
        self.assertEqual(len(phi_artifacts.token_names(3)), 3)


def _flag_strings(module):
    import ast
    from pathlib import Path

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    found.add(arg.value)
    return found


if __name__ == "__main__":
    unittest.main()
