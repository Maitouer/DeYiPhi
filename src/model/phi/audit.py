"""The diagnostics a phi result has to report (``02_phi.md`` §12-§14, ``03_phi.md`` §23).

Three of the four sections answer "is the approximation safe?" -- the head-tail estimator, the
medoid constraint and the router each own an error term, and each is reported next to the exact
reference when one exists. The fourth answers "does the route mean anything?": a token that every
user always picks, or a route posterior that is always one-hot, would show up here long before it
showed up in a downstream metric.

Usage statistics are diagnostics, never objectives: the documents explicitly refuse a
usage-balance or entropy term, so a long tail is reported rather than suppressed.
"""

from __future__ import annotations

import numpy as np
import torch

from . import artifacts as phi_artifacts


def report(cfg, arm, items, catalog, codebook, stats, router_state, split: str = "train",
           progress=None) -> dict:
    report_data = {
        "format": phi_artifacts.FORMAT,
        "arm": phi_artifacts.read_manifest(cfg).get("arm", {}),
        "inputs": {"item_space": items.manifest.get("canonical"), "temperature": arm.item_temperature(),
                   "catalog": int(catalog.size), "num_interests": int(arm.num_interests),
                   "recent": int(arm.recent)},
        "estimator": stats.status(),
        "codebook": codebook.diagnostics_report if hasattr(codebook, "diagnostics_report") else {},
        "router": dict(router_state.validation),
    }
    calibration_path = phi_artifacts.calibration(cfg)
    if calibration_path.is_file():
        calibration = torch.load(str(calibration_path), map_location="cpu", weights_only=False)
        report_data["calibration"] = {"report": calibration.get("report", {}),
                                      "states": len(calibration.get("rows", []))}
    report_data["route"] = route_report(cfg, arm, split, codebook, progress=progress)
    report_data["notes"] = {
        "usage": "usage statistics are diagnostics; the pipeline has no balance or entropy term",
        "downstream": ("Recall / NDCG / valid-generation / latency are properties of the "
                       "generative run and are reported there, not here"),
        "route": "a route is context-dependent supervision, never a permanent item region",
    }
    return report_data


def route_report(cfg, arm, split: str, codebook, *, progress=None) -> dict:
    """Route diagnostics: how concentrated, how diverse, and how many routes an item carries."""
    from . import states as states_module

    path = phi_artifacts.routes(cfg, split)
    if not path.is_file():
        return {"available": False, "reason": "no route artifact for split %s" % split}
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    route = payload["route"].long()
    keep = payload["target_mask"].bool()
    confidence = payload["confidence"].float()
    if not bool(keep.any()):
        return {"available": False, "reason": "no targets carry a route"}
    chosen = route[keep]
    top = confidence[keep]
    vocabulary = int(payload["vocabulary"])
    share = torch.bincount(chosen.clamp_min(0), minlength=vocabulary).double()
    share = share / share.sum().clamp_min(1e-30)
    entropy = -(share[share > 0] * share[share > 0].log2()).sum()
    count = torch.bincount(chosen.clamp_min(0), minlength=vocabulary)
    routes = {
        "available": True, "split": split, "targets": int(keep.sum()),
        "used_tokens": int(count.gt(0).sum()), "unused_tokens": int(count.eq(0).sum()),
        "usage_entropy_bits": float(entropy),
        "max_token_share": float(share.max()),
        "confidence": {"mean": float(top.mean()), "median": float(top.median()),
                       "p10": float(top.quantile(0.1))},
        "posterior_entropy": float(payload.get("route_entropy", 0.0)),
        "candidates_per_target": int(codebook.size),
    }
    # How many distinct routes does one *item* carry? Large values are the signature of a
    # context-dependent route; a single route would mean phi had degenerated into a static region.
    items_rows = payload["target_rows"].numpy()
    flat_items = items_rows[keep.numpy()]
    unique_items, inverse = np.unique(flat_items, return_inverse=True)
    per_item = np.bincount(inverse, minlength=len(unique_items))
    routes["per_item"] = {"items": int(len(unique_items)),
                          "targets_per_item_mean": float(per_item.mean()),
                          "targets_per_item_max": int(per_item.max())}
    distinct = np.zeros(len(unique_items), dtype=np.int64)
    order = np.lexsort((chosen.numpy(), inverse))
    pairs = np.unique(np.stack([inverse[order], chosen.numpy()[order]], axis=1), axis=0)
    np.add.at(distinct, pairs[:, 0], 1)
    routes["per_item"]["distinct_routes_mean"] = float(distinct.mean())
    routes["per_item"]["distinct_routes_max"] = int(distinct.max())

    split_rows = states_module.SplitRows(cfg, arm, split)
    positions = split_rows.positions(payload["source_row_idx"].numpy())
    uid = split_rows.uids(positions)
    user_ids = np.repeat(uid[:, None], keep.shape[1], axis=1)[keep.numpy()]
    unique_users, user_inverse = np.unique(user_ids, return_inverse=True)
    if len(unique_users):
        user_pairs = np.unique(np.stack([user_inverse, chosen.numpy()], axis=1), axis=0)
        per_user = np.bincount(user_pairs[:, 0], minlength=len(unique_users))
        routes["per_user"] = {"users": int(len(unique_users)),
                              "distinct_routes_mean": float(per_user.mean()),
                              "distinct_routes_max": int(per_user.max())}
    return routes


def summarize(report_data: dict) -> dict:
    """The compact form printed at the end of the audit stage."""
    estimator = report_data.get("estimator", {})
    codebook = report_data.get("codebook", {})
    route = report_data.get("route", {})
    return {
        "states": estimator.get("states"),
        "dead_codes": codebook.get("dead_codes"),
        "predictive_kl": (codebook.get("predictive_kl") or {}).get("mean"),
        "validation_kl": ((codebook.get("split_kl") or {}).get("held_out_uid") or {}).get("mean"),
        "router_top1": (report_data.get("router") or {}).get("top1_agreement"),
        "router_regret": ((report_data.get("router") or {}).get("regret") or {}).get("mean"),
        "route_confidence": (route.get("confidence") or {}).get("mean"),
        "routes_per_item": (route.get("per_item") or {}).get("distinct_routes_mean"),
    }
