"""Where every phi stage writes, and the guards that stop a stale artifact from being read.

Layout (``docs/v1_algorithm/03_phi.md`` §3.2)::

    output/model/phi/
    └── <dataset>/<task>/k<K>_r<R>/          one frozen DeYi arm owns one phi vocabulary
        ├── reservoir.pt                     Stage B: the user-balanced state sample
        ├── stats.pt                         Stage B: (A_hat, mu_hat) + stability per state
        ├── calibration.json                 Stage B: the exact-scan quality report
        ├── codebook.pt                      Stage C: medoids, exact normalizers, labels
        ├── router.pt                        Stage D: the distilled tokenizer
        ├── tokens/{train,test}.pt           Stage D: g_rk for every row and slot
        ├── routes/{train,test}.pt           Stage E: g*(r, y) per target
        ├── audit.json                       the reported diagnostics
        └── manifest.json                    per-stage identity + payload

The arm name carries the dataset because ``product`` exists in more than one dataset, and it
carries ``r<R>`` because two arms that differ only in the old/recent cut are different
teachers (same rule as ``src/model/deyi/artifacts.py``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

FORMAT = "phi_predictive_vocabulary_v1"
# Bump this when a stage's *math* changes. Artifact identities are input-only (no hashing, per
# AGENTS.md), so without a revision a fixed bug would silently reuse the artifact it produced --
# the failure mode where a correct fix appears to do nothing.
CODE_REVISION = 4

# The vocabulary's text interface: one token per shared predictive concept. The retired
# two-level ``<|g_a_*|>``/``<|g_b_*|>`` names belonged to the item-region tree and are gone.
TOKEN_FORMAT = "<|g_%d|>"
TOKEN_NONE = "<|g_none|>"

RESERVOIR = "reservoir.pt"
STATS = "stats.pt"
CALIBRATION = "calibration.pt"
CODEBOOK = "codebook.pt"
ROUTER = "router.pt"
AUDIT = "audit.json"
MANIFEST = "manifest.json"


def token_name(index: int) -> str:
    return TOKEN_FORMAT % int(index)


def token_names(size: int) -> list[str]:
    return [token_name(index) for index in range(int(size))]


def root(cfg) -> Path:
    return Path(str(cfg.phi.output_dir))


def arm_name(cfg) -> str:
    return "k%d_r%d" % (int(cfg.deyi.num_interests), int(cfg.deyi.recent))


def arm_dir(output_dir, dataset, task, num_interests, recent) -> Path:
    """``<output_dir>/<dataset>/<task>/k<K>_r<R>``: one frozen teacher's vocabulary.

    Consumers that hold a (dataset, task, K, recent) quadruple rather than a phi config use this
    instead of rebuilding the layout, so the producer and the consumer can never disagree about
    where an arm lives.
    """
    return (Path(str(output_dir)) / str(dataset) / str(task)
            / ("k%d_r%d" % (int(num_interests), int(recent))))


def arm_root(cfg) -> Path:
    """One (DeYi arm) -> one vocabulary: ``<output_dir>/<dataset>/<task>/k<K>_r<R>``."""
    from src.model.deyi import artifacts as deyi_artifacts

    return arm_dir(root(cfg), deyi_artifacts.dataset(cfg), cfg.task, cfg.deyi.num_interests,
                   cfg.deyi.recent)


def reservoir(cfg) -> Path:
    return arm_root(cfg) / RESERVOIR


def stats(cfg) -> Path:
    return arm_root(cfg) / STATS


def calibration(cfg) -> Path:
    return arm_root(cfg) / CALIBRATION


def codebook(cfg) -> Path:
    return arm_root(cfg) / CODEBOOK


def router(cfg) -> Path:
    return arm_root(cfg) / ROUTER


def tokens(cfg, split: str) -> Path:
    _check_split(split)
    return arm_root(cfg) / "tokens" / ("%s.pt" % split)


def routes(cfg, split: str) -> Path:
    _check_split(split)
    return arm_root(cfg) / "routes" / ("%s.pt" % split)


def audit(cfg) -> Path:
    return arm_root(cfg) / AUDIT


def manifest(cfg) -> Path:
    return arm_root(cfg) / MANIFEST


def _check_split(split: str) -> None:
    if split not in ("train", "test"):
        raise ValueError("unsupported split: %s" % split)


def save_pt(payload, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    try:
        torch.save(payload, pending)
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def save_json(payload, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    try:
        pending.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                           encoding="utf-8")
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def read_json(path: Path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_manifest(cfg):
    return read_json(manifest(cfg)) or {"format": FORMAT, "stages": {}}


def record(cfg, stage: str, identity: dict, payload: dict | None = None) -> dict:
    """Append one finished stage to the manifest (atomic rewrite, no partial files)."""
    from src.model.deyi import artifacts as deyi_artifacts

    manifest_data = read_manifest(cfg)
    manifest_data["format"] = FORMAT
    manifest_data.setdefault("arm", {}).update({
        "dataset": deyi_artifacts.dataset(cfg), "task": str(cfg.task),
        "num_interests": int(cfg.deyi.num_interests), "recent": int(cfg.deyi.recent),
        "arm": arm_name(cfg)})
    manifest_data.setdefault("stages", {})[stage] = {"identity": identity, "payload": payload or {}}
    save_json(manifest_data, manifest(cfg))
    return manifest_data["stages"][stage]


def reuse(cfg, stage: str, identity: dict, path: Path | None = None):
    """The recorded payload of ``stage`` when it still describes the current inputs.

    A stage is reusable when its identity matches *and* its artifact exists: an identity-only
    match after a deleted file would otherwise silently skip the work.
    """
    saved = (read_manifest(cfg).get("stages") or {}).get(stage)
    if saved is None:
        return None
    if saved.get("identity") != identity:
        return None
    if path is not None and not Path(path).is_file():
        return None
    return dict(saved.get("payload") or {})


def require(path: Path, stage: str) -> Path:
    """A stage input, with the missing-file message naming the stage that produces it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            "the `%s` stage has not run for this arm: %s is missing "
            "(python -m src.model.phi.cli %s --task %s)" % (stage, path, stage, stage))
    return path


def file_marker(path: Path) -> dict:
    """Size + mtime of a frozen input: enough to notice a re-run, without hashing."""
    path = Path(path)
    stat = path.stat()
    return {"path": str(path), "bytes": int(stat.st_size), "mtime": int(stat.st_mtime)}
