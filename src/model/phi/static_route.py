"""Item-only (static) route targets: Analysis III's middle control (``05_analysis_design.md`` §17.2).

The contextual route of ``03_phi.md`` §15 is a *user-dependent* posterior

    beta_rg(y) ∝ alpha_bar_rg * exp(c_g^T e(y)/tau_p - A_g),

where ``alpha_bar`` aggregates the teacher's recent readout onto the tokens. Dropping that term
leaves a route that depends only on the target item and the frozen vocabulary:

    g_static(y) = argmax_g [ c_g^T e(y) / tau_p - A_g ].

This module writes that partition to ``routes_static/<split>.pt`` in exactly the schema the
contextual file uses, so the downstream arm swaps one tensor and changes nothing else. Comparing
it with the contextual route and with "no route at all" separates *user-dependent predictive
explanation* from *static item category*.

    python -m src.model.phi.static_route --task product --k 4 --recent 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from . import artifacts as phi_artifacts
from . import config as phi_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=phi_config.DEFAULT)
    parser.add_argument("--task", default="product")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--recent", type=int, default=16)
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def build(cfg, split: str, *, batch: int = 4096, device: str = "cpu") -> dict:
    from . import items as items_module
    from . import states as states_module

    arm = states_module.TeacherArm(cfg)
    items = items_module.FrozenItems(cfg)
    codebook = torch.load(str(phi_artifacts.codebook(cfg)), map_location="cpu",
                          weights_only=False)
    center = codebook["center"].float().to(device)
    normaliser = codebook["log_partition"].float().to(device)
    vocabulary = int(center.shape[0])
    temperature = arm.item_temperature()

    payload = arm.payload(split)
    rows = int(payload["mask"].shape[0])
    split_rows = states_module.SplitRows(cfg, arm, split)
    positions = split_rows.positions(payload["source_row_idx"].numpy())
    target_rows, target_mask = split_rows.target_rows(positions, int(cfg.data.target))
    width = int(target_rows.shape[1])

    route = torch.full((rows, width), -1, dtype=torch.int16)
    confidence = torch.zeros((rows, width), dtype=torch.float16)
    entropy_sum, entropy_count = 0.0, 0
    for start in range(0, rows, int(batch)):
        stop = min(start + int(batch), rows)
        keep = torch.from_numpy(target_mask[start:stop])
        flat = torch.from_numpy(target_rows[start:stop].reshape(-1))
        # ``flat < 0`` marks a padding slot; clamp keeps the gather legal and the mask drops it.
        gathered = items.unit(flat.clamp_min(0).numpy(), device=device, dtype=torch.float32)
        gathered = gathered.reshape(stop - start, width, items.dim).to(device)
        logits = (torch.einsum("btd,gd->btg", gathered, center) / float(temperature)
                  - normaliser[None, None, :])
        logits = logits.masked_fill(~keep[:, :, None].to(device), float("-inf"))
        empty = ~torch.isfinite(logits).any(dim=-1, keepdim=True)
        probability = torch.softmax(torch.where(empty, torch.zeros_like(logits), logits), dim=-1)
        probability = probability * keep[:, :, None].to(device).to(probability.dtype)
        best = probability.argmax(dim=-1)
        assign = keep.to(device)
        route[start:stop] = torch.where(assign, best.to(torch.int16),
                                        torch.full_like(best, -1, dtype=torch.int16)).cpu()
        top = probability.max(dim=-1).values
        confidence[start:stop] = torch.where(assign, top.to(torch.float16),
                                            torch.zeros_like(top, dtype=torch.float16)).cpu()
        entry = -torch.special.xlogy(probability, probability).sum(-1)
        entropy_sum += float(entry[assign].sum())
        entropy_count += int(assign.sum())

    return {"format": phi_artifacts.FORMAT + "_routes_static", "split": split, "rows": rows,
            "targets": width, "vocabulary": vocabulary, "mode": "static",
            "source_row_idx": payload["source_row_idx"].clone(),
            "target_rows": torch.from_numpy(target_rows.astype("int64")),
            "target_mask": torch.from_numpy(target_mask),
            "route": route, "confidence": confidence,
            "route_entropy": (entropy_sum / entropy_count) if entropy_count else 0.0,
            "targets_with_route": int((route >= 0).sum())}


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = phi_config.load(args.config)
    cfg.task = args.task
    cfg.deyi.num_interests = int(args.k)
    cfg.recent[cfg.task] = int(args.recent)
    cfg.deyi.recent = int(args.recent)
    for split in [name.strip() for name in args.splits.split(",") if name.strip()]:
        payload = build(cfg, split, batch=int(args.batch), device=args.device)
        path = phi_artifacts.arm_root(cfg) / "routes_static" / ("%s.pt" % split)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(path))
        print("[static-route] %s rows=%d targets=%d with_route=%d entropy=%.3f -> %s"
              % (split, payload["rows"], payload["targets"], payload["targets_with_route"],
                 payload["route_entropy"], path), flush=True)


if __name__ == "__main__":
    main()
