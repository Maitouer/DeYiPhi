"""Every model entry point must follow the data / fit-pca logging style.

The job creates its own ``log/model/<stage>/<run-id>/`` directory (console.log, errors.log,
progress.*) through ``scripts/model/common_logging.sh`` and points Slurm's own output at
``/dev/null``. A new sbatch that forgets the directives silently drops ``slurm-<jobid>.out`` into
the project root, which is exactly what this test prevents.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").rglob("*.sbatch"))


def test_there_are_entry_points_to_check():
    assert SCRIPTS, "no sbatch entry points found"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_sbatch_uses_the_common_logging_style(path):
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --output=/dev/null" in text, "Slurm stdout must go to /dev/null"
    assert "#SBATCH --error=/dev/null" in text, "Slurm stderr must go to /dev/null"
    # Either the shared helper or the older inline equivalent (deyi/prepare.sbatch) -- both create
    # ``log/model/<stage>/<run-id>/`` and redirect their own stdout/stderr there.
    assert "common_logging.sh" in text or "log/" in text, \
        "the job must create its own log directory under log/"
    # Scratch must never fall back to the shared project tree: either the shared helper (which
    # pins TMPDIR with a write probe) or an explicit TMPDIR in the script itself.
    assert "common_logging.sh" in text or "TMPDIR" in text, \
        "the job must pin TMPDIR or use the shared helper"
