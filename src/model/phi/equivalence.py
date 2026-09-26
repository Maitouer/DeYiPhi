"""Held-out predictive equivalence of a shared vocabulary (``05_analysis_design.md`` §19-22).

Recommendation accuracy cannot show that a codebook groups states by *predictive* behaviour, so
this stage measures the mechanism directly on states the codebook never saw: the **test split**,
whose users are disjoint from the training reservoir by construction.

For each sampled held-out state ``z`` with representative ``c = c_{q(z)}`` it reports

* §21 held-out predictive distortion ``D_KL(p_z || p_c)`` and
* §22 future-NLL increase ``-log p(y|c) + log p(y|z)`` over the state's own future targets,

both **exact** over the task catalog: ``A(z)`` (log-partition) and ``mu_z = E_{i~p_z}[e(i)]`` are
computed by streaming the catalog once in blocks, so no sampled-softmax approximation enters the
comparison. Run it once per vocabulary (predictive / euclidean / ...) and compare the JSON files.

    python -m src.model.phi.equivalence --task product --k 4 --recent 16 --states 1024
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import artifacts as phi_artifacts
from . import codebook as codebook_module
from . import config as phi_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=phi_config.DEFAULT)
    parser.add_argument("--task", default="product")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--recent", type=int, default=14)
    parser.add_argument("--states", type=int, default=1024, help="held-out states to score")
    parser.add_argument("--block", type=int, default=32768, help="catalog rows per block")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    # Which distortion defines the partition this stage scores. Normally read from the codebook
    # artifact, but the artifact only records it from the metric-aware revision onwards, so an
    # explicit flag keeps older vocabularies comparable instead of silently defaulting to
    # ``predictive`` (which is exactly how the first euclidean run was scored under the wrong
    # criterion).
    parser.add_argument("--metric", choices=("predictive", "euclidean", "cosine"))
    parser.add_argument("--out", help="output json (defaults next to the vocabulary)")
    return parser.parse_args(argv)


def _catalog_statistics(items, catalog, states, temperature, block, device, progress=None):
    """Exact ``A(z)`` and ``mu_z`` for every state, in two streaming passes over the catalog."""
    count = int(states.shape[0])
    log_partition = torch.full((count,), -float("inf"), dtype=torch.float32, device=device)
    total = float(temperature)
    for start in range(0, int(catalog.size), int(block)):
        rows = catalog[start:start + int(block)]
        unit = items.unit(np.asarray(rows, dtype=np.int64), device=device, dtype=torch.float32)
        logits = (unit @ states.T) / total                     # (B, N)
        log_partition = torch.logaddexp(log_partition, torch.logsumexp(logits, dim=0))
        if progress is not None and start % (int(block) * 8) == 0:
            progress(start + len(rows), int(catalog.size))
    moment = torch.zeros_like(states)
    for start in range(0, int(catalog.size), int(block)):
        rows = catalog[start:start + int(block)]
        unit = items.unit(np.asarray(rows, dtype=np.int64), device=device, dtype=torch.float32)
        logits = (unit @ states.T) / total
        # Absolute probabilities, *not* a per-block softmax: ``exp(s/tau - A)`` already sums to one
        # over the whole catalog, whereas renormalising inside each block silently reweights sparse
        # blocks and breaks the consistency between ``A(z)`` and ``mu_z`` (it made the held-out KL
        # come out negative).
        weight = torch.exp(logits - log_partition[None, :])
        moment += weight.T @ unit
    return log_partition, moment


def run(cfg, *, held_out_states: int, block: int, seed: int, device: str, out=None,
        metric: str | None = None) -> dict:
    from . import items as items_module
    from . import states as states_module

    started = time.time()
    arm = states_module.TeacherArm(cfg)
    items = items_module.FrozenItems(cfg)
    from src.data.dataset import open_store

    held_out_dataset = open_store(str(cfg.data.root)).dataset(str(arm.dataset))
    catalog = np.asarray(held_out_dataset.task_catalog_rows(cfg.task), dtype=np.int64)
    codebook = torch.load(str(phi_artifacts.codebook(cfg)), map_location="cpu", weights_only=False)
    metric = str(metric or codebook.get("metric", "predictive"))
    temperature = float(arm.item_temperature())

    payload = arm.payload("test")                    # held-out: test users never enter the reservoir
    states, mask = payload["states"].float(), payload["mask"].bool()
    split_rows = states_module.SplitRows(cfg, arm, "test")
    positions = split_rows.positions(payload["source_row_idx"].numpy())
    target_rows, target_mask = split_rows.target_rows(positions, int(cfg.data.target))

    rows = torch.nonzero(mask, as_tuple=False)
    order = torch.randperm(int(rows.shape[0]), generator=torch.Generator().manual_seed(int(seed)))
    picked = rows[order[:min(int(held_out_states), int(rows.shape[0]))]]
    state = states[picked[:, 0], picked[:, 1]].to(device)
    owner = picked[:, 0].to(torch.long)

    log_partition, moment = _catalog_statistics(
        items, catalog, state, temperature, int(block), device,
        progress=lambda done, total: print("[equivalence] catalog %d/%d" % (done, total), flush=True))

    center = codebook["center"].float().to(device)
    normaliser = codebook["log_partition"].float().to(device)
    scores = codebook_module.scores_for(state, moment, center, normaliser, temperature, metric)
    label = scores.argmin(dim=1)
    representative = center[label]
    representative_A = normaliser[label]

    keep = torch.from_numpy(target_mask).to(device)[owner]           # (N, T)
    targets = torch.from_numpy(target_rows.astype("int64")).to(device)[owner]
    target_unit = items.unit(targets.reshape(-1).clamp_min(0).cpu().numpy(), device=device,
                             dtype=torch.float32).reshape(keep.shape[0], keep.shape[1], items.dim)

    with torch.no_grad():
        # §21: D_KL(p_z || p_c) = A(c) - A(z) + (z - c)^T mu_z / tau_p
        kl = representative_A - log_partition + ((state - representative) * moment).sum(-1) / temperature
        # §22: NLL(z,y) = A(z) - z^T e(y)/tau_p, so the increase is NLL(c) - NLL(z).
        # Explicit per-target dot product: ``state @ target_unit.transpose(1, 2)`` would be read as
        # a batched matmul and return an extra singleton axis.
        dot_z = (state.unsqueeze(1) * target_unit).sum(-1)
        dot_c = (representative.unsqueeze(1) * target_unit).sum(-1)
        nll_z = log_partition[:, None] - dot_z / temperature
        nll_c = representative_A[:, None] - dot_c / temperature
        delta = (nll_c - nll_z) * keep
        per_state = delta.sum(1) / keep.sum(1).clamp_min(1)
        mean_delta = float(((delta.sum(1) / keep.sum(1).clamp_min(1))[keep.any(1)]).mean())
        standard_error = float(per_state[keep.any(1)].std(unbiased=True)
                               / math.sqrt(int(keep.any(1).sum())))

    report = {
        "format": phi_artifacts.FORMAT + "_equivalence",
        "arm": {"dataset": str(arm.dataset), "task": cfg.task,
                "num_interests": int(cfg.deyi.num_interests), "recent": int(cfg.deyi.recent)},
        "metric": metric,
        "held_out": {"split": "test", "states_scored": int(state.shape[0]),
                     "rows_scored": int(torch.unique(owner).numel()),
                     "targets_scored": int(keep.sum()),
                     "catalog_rows": int(catalog.size)},
        "predictive_kl": {"mean": float(kl.mean()), "std": float(kl.std(unbiased=True)),
                          "standard_error": float(kl.std(unbiased=True) / math.sqrt(kl.numel())),
                          "median": float(kl.median())},
        "future_nll_increase": {"mean": mean_delta, "standard_error": standard_error,
                                "per_state_mean": float(per_state[keep.any(1)].mean())},
        "seconds": round(time.time() - started, 1),
    }
    path = Path(out) if out else (phi_artifacts.arm_root(cfg) / "equivalence.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print("[equivalence] wrote %s" % path, flush=True)
    return report


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = phi_config.load(args.config)
    cfg.task = args.task
    cfg.deyi.num_interests = int(args.k)
    cfg.recent[cfg.task] = int(args.recent)
    cfg.deyi.recent = int(args.recent)
    run(cfg, held_out_states=int(args.states), block=int(args.block), seed=int(args.seed),
        device=args.device, out=args.out, metric=args.metric)


if __name__ == "__main__":
    main()
