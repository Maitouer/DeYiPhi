"""Item-space manifest, reuse rule and self-checks.

Reuse is decided by a plain declarative marker copied from ``items/item_table.json`` plus the
PCA settings that produced the space — no hashing and no byte-level defence, per
``AGENTS.md``. Rebuilding the item table changes ``created``, and changing ``dim``/``samples``
changes the marker, so either automatically triggers a re-fit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from src.model.deyi import artifacts

FORMAT = artifacts.ITEM_SPACE_FORMAT


def canonical_tag(cfg, space):
    """The declarative identity of everything that determines this space."""
    return {
        "created": str(space.items.meta["created"]),
        "scope": "+".join(space.datasets),
        "rows": int(len(space.items)),
        "dim": int(cfg.pca.dim),
        "normalize_input": True,
        "samples": int(cfg.pca.get("samples", 0)),
    }


def read_manifest(cfg):
    return artifacts.read_manifest(artifacts.item_space_manifest(cfg))


def completed(cfg, tag):
    """Return the manifest when it matches ``tag`` and every artifact file exists."""
    manifest = read_manifest(cfg)
    if not manifest or manifest.get("format") != FORMAT or manifest.get("canonical") != tag:
        return None
    for path in (artifacts.pca(cfg), artifacts.embeddings(cfg), artifacts.item_space_stats(cfg)):
        if not path.is_file():
            return None
    return manifest


def write_manifest(cfg, payload):
    path = artifacts.item_space_manifest(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def require_item_space(cfg):
    """Load the manifest and confirm it still describes the current item table."""
    from src.data.dataset import open_store

    space = open_store(str(cfg.data.root)).item_space()
    tag = canonical_tag(cfg, space)
    manifest = completed(cfg, tag)
    if manifest is None:
        raise ValueError(
            "the DeYi item space is missing or was built from a different item table / PCA "
            "settings; run scripts/model/deyi/prepare.sbatch (%s)" % artifacts.item_space(cfg)
        )
    return manifest


def selfcheck(cfg, space, mean, components, embeddings_path, stats, progress, checks_path,
              sample_rows=200, seed=2026):
    """Structural and numerical checks of the freshly written item space."""
    results = []

    def check(name, ok, detail=""):
        results.append({"check": name, "ok": bool(ok), "detail": detail})
        progress.line("selfcheck", "%s %s %s" % ("PASS" if ok else "FAIL", name, detail),
                      force=True)

    embeddings = np.load(embeddings_path, mmap_mode="r")
    rows = len(space.items)
    dim = int(cfg.pca.dim)
    check("embeddings shape/dtype",
          tuple(embeddings.shape) == (rows, dim) and embeddings.dtype == np.float16,
          "%s %s" % (tuple(embeddings.shape), embeddings.dtype))

    weight = torch.from_numpy(np.asarray(components)).float()
    error = float((weight @ weight.T - torch.eye(dim)).abs().max())
    check("components orthonormal", error < 1e-3, "max|WW^T-I|=%.2e" % error)

    # Sample rows that are actually in the catalog: the parity check stays meaningful even
    # when the item table holds rows that no task references.
    catalog = np.asarray(space.catalog)
    generator = np.random.default_rng(seed)
    take = min(sample_rows, len(catalog))
    if take:
        picked = np.sort(generator.choice(catalog, size=take, replace=False)).astype(np.int64)
        raw = np.asarray(space.items.text[picked], dtype=np.float32)
        norm = np.linalg.norm(raw, axis=1, keepdims=True)
        normalized = raw / np.where(norm > 0, norm, 1.0)
        expected = (normalized - np.asarray(mean)) @ np.asarray(components).T
        got = np.asarray(embeddings[picked], dtype=np.float32)
        worst = float(np.abs(expected - got).max())
        check("sampled re-projection matches", worst < 5e-2, "max|d|=%.3e" % worst)
        check("catalog vectors have non-zero norm", bool((norm > 0).all()))
    else:
        check("sampled re-projection matches", True, "catalog empty")
        check("catalog vectors have non-zero norm", True, "catalog empty")

    check("no NaN/Inf", bool(np.isfinite(np.asarray(embeddings[::997][:512])).all()))
    check("row 0 is never referenced",
          bool(len(catalog) and catalog.min() > 0 and int(space.items.pids[0]) == 0),
          "catalog_min=%d" % (int(catalog.min()) if len(catalog) else -1))
    check("no zero-norm source vector", int(stats.get("zero_norm_rows", 0)) == 0,
          "zero_norm=%d" % int(stats.get("zero_norm_rows", 0)))

    payload = {"passed": sum(1 for item in results if item["ok"]),
               "failed": sum(1 for item in results if not item["ok"]), "checks": results}
    checks_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
