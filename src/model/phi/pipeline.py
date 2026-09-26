"""Stage orchestration for the phi chain.

The stages follow ``03_phi.md`` §2 and stay separately runnable on purpose: the expensive parts
(the head-tail estimate, the exact normalizers) are the parts a rerun must be able to avoid, and
each stage's identity records exactly which inputs produced it.
"""

from __future__ import annotations

import json
import os
from time import monotonic

import numpy as np
import torch

from . import artifacts as phi_artifacts
from . import runtime
from . import states as states_module

STAGES = ("plan", "reservoir", "estimate", "codebook", "router", "tokenize", "route", "audit")


def _base_identity(cfg, arm) -> dict:
    return {"arm": arm.identity(), "code": phi_artifacts.CODE_REVISION}


def _compute_identity(device, dtype) -> dict:
    # The search runs in the compute dtype, so a CPU run and a bf16 GPU run are different
    # artifacts by construction rather than by accident.
    return {"device": str(device.type), "dtype": str(dtype).replace("torch.", "")}


def plan(cfg, progress, split="train"):
    """Resolve every input and print the work the configured stages would do."""
    from .items import FrozenItems, TaskCatalog

    arm = states_module.TeacherArm(cfg)
    items = FrozenItems(cfg)
    catalog = TaskCatalog(cfg, items, lam=float(cfg.phi.estimator.proposal_lambda))
    info = {
        "dataset": arm.dataset, "task": arm.task, "num_interests": arm.num_interests,
        "recent": arm.recent, "state_dim": arm.dim,
        "item_space": items.path, "item_rows": items.rows, "dim": items.dim,
        "catalog": catalog.size, "checkpoint": str(arm.checkpoint_path()),
        "item_temperature": arm.item_temperature(), "route_temperature": arm.route_temperature(),
        "vocabulary": int(cfg.phi.vocabulary), "reservoir": int(cfg.phi.reservoir.size),
        "head": int(cfg.phi.estimator.head), "tail": int(cfg.phi.estimator.tail),
        "output": str(phi_artifacts.arm_root(cfg)),
        "states": {name: arm.rows(name) for name in ("train", "test")
                   if arm.states_path(name).is_file()},
    }
    progress.banner([
        "=" * 96,
        "phi | job=%s stage=plan task=%s split=%s" % (os.environ.get("SLURM_JOB_ID"), cfg.task,
                                                      split),
        "  teacher    : %s  k=%d  r=%d  tau_p=%.4f" % (arm.states_path("train"), arm.num_interests,
                                                       arm.recent, info["item_temperature"]),
        "  item space : %s  rows=%d dim=%d" % (items.path, items.rows, items.dim),
        "  catalog    : %d items (the task's own candidate universe)" % catalog.size,
        "  vocabulary : L=%d over a reservoir of %d states (M=%d head, N=%d tail)"
        % (info["vocabulary"], info["reservoir"], info["head"], info["tail"]),
        "  output     : %s" % info["output"],
        "  stages     : " + " -> ".join(STAGES),
        "=" * 96,
    ])
    for line, value in info.items():
        progress.line("plan", "%s=%s" % (line, value))
    return info


def build_reservoir(cfg, progress):
    """Stage B (part 1): the user-balanced reservoir of frozen DeYi states."""
    from .reservoir import build

    arm = states_module.TeacherArm(cfg)
    identity = {**_base_identity(cfg, arm),
                "size": int(cfg.phi.reservoir.size), "seed": int(cfg.phi.reservoir.seed)}
    saved = phi_artifacts.reuse(cfg, "reservoir", identity, phi_artifacts.reservoir(cfg))
    if saved is not None:
        progress.line("reservoir", "reused %s" % phi_artifacts.reservoir(cfg), force=True)
        return saved
    started = monotonic()
    payload = build(cfg, arm, size=int(cfg.phi.reservoir.size), seed=int(cfg.phi.reservoir.seed),
                    progress=progress)
    phi_artifacts.save_pt(payload, phi_artifacts.reservoir(cfg))
    summary = {"states": int(payload["states"].shape[0]), "users": int(payload["uid"].unique().numel()),
               "rows": int(payload["row"].unique().numel()),
               "mass": float(payload["weight"].double().sum()),
               "slots_mean": float(payload["count"].float().mean()),
               "seconds": round(monotonic() - started, 1)}
    phi_artifacts.record(cfg, "reservoir", identity, summary)
    progress.line("reservoir", json.dumps(summary), force=True)
    return summary


def estimate_stats(cfg, progress):
    """Stage B (part 2): head-tail statistics for the reservoir, plus the calibration set."""
    from . import estimator as estimator_module
    from .items import CatalogUnits, FrozenItems, TaskCatalog

    arm = states_module.TeacherArm(cfg)
    items = FrozenItems(cfg)
    estimator_cfg = cfg.phi.estimator
    device = runtime.device(estimator_cfg.device)
    dtype = runtime.compute_dtype(device)
    identity = {**_base_identity(cfg, arm),
                "reservoir": phi_artifacts.file_marker(phi_artifacts.reservoir(cfg)),
                "head": int(estimator_cfg.head), "tail": int(estimator_cfg.tail),
                "lambda": float(estimator_cfg.proposal_lambda),
                "item_block": int(estimator_cfg.item_block),
                "query_block": int(estimator_cfg.query_block),
                # The stability controls and the calibration set are part of the result, not
                # decoration: changing any of them must invalidate the cached statistics.
                "head_mass_floor": float(estimator_cfg.head_mass_floor),
                "tail_ess_floor": float(estimator_cfg.tail_ess_floor),
                "fallback_rounds": int(estimator_cfg.fallback_rounds),
                "fallback_factor": float(estimator_cfg.fallback_factor),
                "calibration": int(estimator_cfg.calibration),
                "resident": str(estimator_cfg.resident),
                "compute": _compute_identity(device, dtype)}
    saved = phi_artifacts.reuse(cfg, "estimate", identity, phi_artifacts.stats(cfg))
    if saved is not None:
        progress.line("estimate", "reused %s" % phi_artifacts.stats(cfg), force=True)
        return saved
    started = monotonic()
    reservoir = torch.load(str(phi_artifacts.require(phi_artifacts.reservoir(cfg), "reservoir")),
                           map_location="cpu",
                           weights_only=False)
    catalog = TaskCatalog(cfg, items, lam=float(estimator_cfg.proposal_lambda))
    units = CatalogUnits(catalog, device=device, budget_bytes=runtime.free_device_bytes(device),
                         resident=str(estimator_cfg.resident))
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.phi.reservoir.seed) + 1)
    progress.line("estimate", "device=%s dtype=%s resident=%s catalog=%d"
                  % (device, dtype, units.resident is not None, catalog.size), force=True)
    statistics = estimator_module.estimate(
        reservoir["states"], units, catalog, arm.item_temperature(),
        head=int(estimator_cfg.head), tail=int(estimator_cfg.tail),
        item_block=int(estimator_cfg.item_block), query_block=int(estimator_cfg.query_block),
        device=device, dtype=dtype, generator=generator, progress=progress,
        moment_block=int(estimator_cfg.moment_block))
    statistics.head_mass_floor = float(estimator_cfg.head_mass_floor)
    statistics.tail_ess_floor = float(estimator_cfg.tail_ess_floor)
    failing = torch.nonzero((statistics.head_mass < statistics.head_mass_floor)
                            | (statistics.tail_ess < statistics.tail_ess_floor),
                            as_tuple=False).squeeze(1)
    share = float(failing.numel()) / max(int(statistics.head_mass.numel()), 1)
    if share > float(estimator_cfg.fallback_max_fraction):
        # The fallback is for the few hard states, not for a floor that the whole reservoir cannot
        # meet: refining everything would triple the cost and the peak memory to fix nothing.
        progress.line("estimate", "fallback skipped: %.1f%% of states are below the floors "
                      "(>%.1f%%); treat this as a floor/calibration question, not as hard states"
                      % (100 * share, 100 * float(estimator_cfg.fallback_max_fraction)), force=True)
        failing = torch.zeros(0, dtype=torch.long)
    estimator_module.refine(reservoir["states"], units, catalog, arm.item_temperature(), statistics,
                            failing, rounds=int(estimator_cfg.fallback_rounds),
                            factor=float(estimator_cfg.fallback_factor),
                            item_block=int(estimator_cfg.item_block),
                            query_block=int(estimator_cfg.query_block), device=device, dtype=dtype,
                            generator=generator, progress=progress)
    payload = {"format": phi_artifacts.FORMAT + "_stats", "log_partition": statistics.log_partition,
               "moment": statistics.moment, "head_mass": statistics.head_mass,
               "tail_ess": statistics.tail_ess, "head": statistics.head, "tail": statistics.tail,
               "fallback": list(statistics.fallback)}
    phi_artifacts.save_pt(payload, phi_artifacts.stats(cfg))
    summary = dict(statistics.status())
    summary["seconds"] = round(monotonic() - started, 1)
    summary["resident"] = units.resident is not None
    calibration = _calibrate(cfg, arm, items, catalog, units, device, dtype, progress)
    if calibration is not None:
        # Tensors in, tensors out: the calibration keeps the exact ``(A, mu)`` of a few hundred
        # states so the codebook stage can measure its own assignment agreement against them.
        phi_artifacts.save_pt(calibration, phi_artifacts.calibration(cfg))
        summary["calibration"] = calibration.get("report")
    phi_artifacts.record(cfg, "estimate", identity, summary)
    progress.line("estimate", json.dumps({k: v for k, v in summary.items() if k != "fallback"}),
                  force=True)
    return summary


def _calibrate(cfg, arm, items, catalog, units, device, dtype, progress):
    """Exact full-catalog scan on a few hundred states; the estimator's own error bar."""
    count = int(cfg.phi.estimator.calibration)
    if count <= 0:
        return None
    from . import estimator as estimator_module

    reservoir = torch.load(str(phi_artifacts.require(phi_artifacts.reservoir(cfg), "reservoir")),
                           map_location="cpu",
                           weights_only=False)
    statistics = torch.load(str(phi_artifacts.require(phi_artifacts.stats(cfg), "estimate")),
                            map_location="cpu", weights_only=False)
    selected = torch.linspace(0, int(reservoir["states"].shape[0]) - 1,
                              steps=min(count, int(reservoir["states"].shape[0]))).round().long()
    query = reservoir["states"][selected]
    progress.line("estimate", "calibration: exact full-catalog scan over %d states"
                  % int(selected.numel()), force=True)
    log_exact, moment_exact = estimator_module.exact(
        query, units, catalog, arm.item_temperature(),
        item_block=int(cfg.phi.estimator.item_block),
        query_block=int(cfg.phi.estimator.query_block), device=device, dtype=dtype,
        progress=progress)
    report = estimator_module.calibration_report(
        statistics["log_partition"][selected], statistics["moment"][selected],
        log_exact, moment_exact)
    return {"rows": selected.tolist(), "log_partition": log_exact, "moment": moment_exact,
            "report": report}


def fit_codebook(cfg, progress):
    """Stage C: predictive-KL medoids, then the exact normalizers of the L final tokens."""
    from . import codebook as codebook_module
    from .items import CatalogUnits, FrozenItems, TaskCatalog

    arm = states_module.TeacherArm(cfg)
    items = FrozenItems(cfg)
    identity = {**_base_identity(cfg, arm),
                "stats": phi_artifacts.file_marker(phi_artifacts.stats(cfg)),
                "vocabulary": int(cfg.phi.vocabulary),
                "iterations": int(cfg.phi.codebook.iterations),
                "seed": int(cfg.phi.codebook.seed),
                # The distortion criterion is part of the artifact identity: without it a
                # euclidean run would silently reuse the predictive codebook.
                "metric": str(cfg.phi.codebook.get("metric", "predictive"))}
    saved = phi_artifacts.reuse(cfg, "codebook", identity, phi_artifacts.codebook(cfg))
    if saved is not None:
        progress.line("codebook", "reused %s" % phi_artifacts.codebook(cfg), force=True)
        return saved
    started = monotonic()
    reservoir = torch.load(str(phi_artifacts.require(phi_artifacts.reservoir(cfg), "reservoir")),
                           map_location="cpu", weights_only=False)
    statistics = torch.load(str(phi_artifacts.require(phi_artifacts.stats(cfg), "estimate")),
                            map_location="cpu", weights_only=False)
    book = codebook_module.fit(reservoir["states"].float(), statistics["moment"],
                               statistics["log_partition"], reservoir["weight"],
                               int(cfg.phi.vocabulary), arm.item_temperature(),
                               iterations=int(cfg.phi.codebook.iterations),
                               seed=int(cfg.phi.codebook.seed), progress=progress,
                               metric=str(cfg.phi.codebook.get("metric", "predictive")))
    device = runtime.device(cfg.phi.estimator.device)
    dtype = runtime.compute_dtype(device)
    catalog = TaskCatalog(cfg, items, lam=float(cfg.phi.estimator.proposal_lambda))
    units = CatalogUnits(catalog, device=device, budget_bytes=runtime.free_device_bytes(device),
                         resident=str(cfg.phi.estimator.resident))
    if bool(cfg.phi.codebook.exact_normalizer):
        # §13: only the L medoids get an exact normalizer, which is O(L|C|d) rather than O(|S||C|d).
        progress.line("codebook", "exact normalizers over %d medoids" % book.size, force=True)
        exact_log, exact_moment = _exact_normalizers(
            book, units, catalog, arm.item_temperature(), cfg.phi.estimator,
            device, dtype, progress)
        book.log_partition = exact_log.float()
        book.moment = exact_moment.float()
        book.exact = True
    report = codebook_module.diagnostics(
        book, reservoir["states"].float(), statistics["moment"], statistics["log_partition"],
        reservoir["weight"], reservoir["uid"], arm.item_temperature(),
        split_users=codebook_module.held_out_split(
            reservoir["uid"], float(cfg.phi.codebook.held_out_fraction),
            int(cfg.phi.codebook.seed)))
    report["exact_normalizer"] = bool(book.exact)
    report["assignment_agreement"] = _assignment_agreement(cfg, book, statistics,
                                                          arm.item_temperature())
    payload = {"format": phi_artifacts.FORMAT + "_codebook", "center": book.center,
               "log_partition": book.log_partition, "medoid": book.medoid, "label": book.label,
               "moment": book.moment if book.moment is not None else torch.zeros(0),
               "exact": bool(book.exact), "history": book.history,
               "tokens": phi_artifacts.token_names(book.size),
               "token_none": phi_artifacts.TOKEN_NONE, "report": report}
    # ``metric`` is written into the artifact: Analysis II fits several partitions of the same
    # states side by side, and a reader must be able to tell which distortion produced this one.
    payload["metric"] = str(getattr(book, "metric", cfg.phi.codebook.get("metric", "predictive")))
    phi_artifacts.save_pt(payload, phi_artifacts.codebook(cfg))
    summary = {"size": book.size, "dead_codes": report["dead_codes"], "exact": bool(book.exact),
               "predictive_kl_mean": report["predictive_kl"]["mean"],
               "seconds": round(monotonic() - started, 1)}
    phi_artifacts.record(cfg, "codebook", identity, summary)
    progress.line("codebook", json.dumps(summary), force=True)
    return summary


def _exact_normalizers(book, units, catalog, temperature, estimator_cfg, device, dtype, progress):
    from . import estimator as estimator_module

    return estimator_module.exact(book.center, units, catalog, temperature,
                                  item_block=int(estimator_cfg.item_block),
                                  query_block=int(estimator_cfg.query_block), device=device,
                                  dtype=dtype, progress=progress)


def _assignment_agreement(cfg, book, statistics, temperature):
    """``03_phi.md`` §8: how often the estimated geometry assigns the same codeword as the exact one.

    Measured on the calibration states, where a full-catalog scan gave the exact ``(A, mu)``. The
    frozen codebook is shared by both sides, so the comparison isolates the estimator's error
    rather than mixing it with a different codebook.
    """
    from . import estimator as estimator_module

    path = phi_artifacts.calibration(cfg)
    if not path.is_file():
        return {"available": False, "reason": "no calibration artifact"}
    calibration = torch.load(str(path), map_location="cpu", weights_only=False)
    rows = torch.as_tensor(calibration["rows"], dtype=torch.long)
    from .codebook import distance

    approximate = distance(statistics["moment"][rows], book.center, book.log_partition, temperature)
    exact = distance(torch.as_tensor(calibration["moment"]), book.center, book.log_partition,
                     temperature)
    agreement = estimator_module.assignment_agreement(approximate, exact)
    agreement["states"] = int(rows.numel())
    return agreement


def build_router(cfg, progress):
    """Stage D (part 1): distil the reservoir's predictive-KL labels into the fast router."""
    from . import codebook as codebook_module
    from . import router as router_module

    arm = states_module.TeacherArm(cfg)
    identity = {**_base_identity(cfg, arm),
                "codebook": phi_artifacts.file_marker(phi_artifacts.codebook(cfg)),
                "hidden": int(cfg.phi.router.hidden), "layers": int(cfg.phi.router.layers),
                "epochs": int(cfg.phi.router.epochs), "seed": int(cfg.phi.router.seed)}
    saved = phi_artifacts.reuse(cfg, "router", identity, phi_artifacts.router(cfg))
    if saved is not None:
        progress.line("router", "reused %s" % phi_artifacts.router(cfg), force=True)
        return saved
    started = monotonic()
    reservoir = torch.load(str(phi_artifacts.require(phi_artifacts.reservoir(cfg), "reservoir")),
                           map_location="cpu",
                           weights_only=False)
    statistics = torch.load(str(phi_artifacts.require(phi_artifacts.stats(cfg), "estimate")),
                            map_location="cpu", weights_only=False)
    payload = torch.load(str(phi_artifacts.require(phi_artifacts.codebook(cfg), "codebook")),
                         map_location="cpu", weights_only=False)
    book = codebook_module.Codebook(medoid=payload["medoid"], center=payload["center"],
                                    log_partition=payload["log_partition"],
                                    label=payload["label"], moment=payload.get("moment"))
    fit_users, held_out = codebook_module.held_out_split(
        reservoir["uid"], float(cfg.phi.router.validation_fraction), int(cfg.phi.router.seed))
    device = runtime.device(cfg.phi.estimator.device)
    state = router_module.train(
        reservoir["states"].float(), book.label, reservoir["weight"], dim=arm.dim, size=book.size,
        hidden=int(cfg.phi.router.hidden), layers=int(cfg.phi.router.layers),
        epochs=int(cfg.phi.router.epochs), batch_size=int(cfg.phi.router.batch_size),
        learning_rate=float(cfg.phi.router.learning_rate),
        weight_decay=float(cfg.phi.router.weight_decay), seed=int(cfg.phi.router.seed),
        validation=held_out, moment=statistics["moment"], reference=book,
        temperature=arm.item_temperature(), device=device, progress=progress)
    phi_artifacts.save_pt({"format": phi_artifacts.FORMAT + "_router",
                           "state_dict": state.state_dict, "config": state.config,
                           "validation": state.validation}, phi_artifacts.router(cfg))
    summary = {"epochs": int(state.validation.get("epochs", 0)),
               "top1_agreement": state.validation.get("top1_agreement"),
               "regret_mean": (state.validation.get("regret") or {}).get("mean"),
               "seconds": round(monotonic() - started, 1)}
    phi_artifacts.record(cfg, "router", identity, summary)
    progress.line("router", json.dumps(summary), force=True)
    return summary


def tokenize(cfg, progress, split="train"):
    """Stage D (part 2): every state of one split, without touching the catalog."""
    from . import router as router_module
    from . import tokenize as tokenize_module

    arm = states_module.TeacherArm(cfg)
    identity = {**_base_identity(cfg, arm),
                "router": phi_artifacts.file_marker(phi_artifacts.router(cfg)),
                "states": phi_artifacts.file_marker(arm.states_path(split)),
                "split": split}
    saved = phi_artifacts.reuse(cfg, "tokenize/%s" % split, identity,
                               phi_artifacts.tokens(cfg, split))
    if saved is not None:
        progress.line("tokenize", "reused %s" % phi_artifacts.tokens(cfg, split), force=True)
        return saved
    started = monotonic()
    payload = torch.load(str(phi_artifacts.require(phi_artifacts.router(cfg), "router")),
                         map_location="cpu", weights_only=False)
    module = router_module.Router(payload["state_dict"], payload["config"],
                                  payload.get("validation", {})).build()
    device = runtime.device(cfg.phi.estimator.device)
    module = module.to(device)
    result = tokenize_module.run(arm, module, split, vocabulary=int(payload["config"]["size"]),
                                 batch=int(cfg.phi.tokenize.batch_size), device=device,
                                 progress=progress)
    phi_artifacts.save_pt(result, phi_artifacts.tokens(cfg, split))
    summary = {"split": split, "rows": result["rows"], "slots": result["slots"],
               "valid_slots": result["valid_slots"], "vocabulary": result["vocabulary"],
               "seconds": round(monotonic() - started, 1)}
    phi_artifacts.record(cfg, "tokenize/%s" % split, identity, summary)
    progress.line("tokenize", json.dumps(summary), force=True)
    return summary


def route_labels(cfg, progress, split="train"):
    """Stage E: one predictive route per supervised target."""
    from . import codebook as codebook_module
    from . import route as route_module
    from .items import FrozenItems

    arm = states_module.TeacherArm(cfg)
    identity = {**_base_identity(cfg, arm),
                "tokens": phi_artifacts.file_marker(phi_artifacts.tokens(cfg, split)),
                "codebook": phi_artifacts.file_marker(phi_artifacts.codebook(cfg)),
                "split": split}
    saved = phi_artifacts.reuse(cfg, "route/%s" % split, identity, phi_artifacts.routes(cfg, split))
    if saved is not None:
        progress.line("route", "reused %s" % phi_artifacts.routes(cfg, split), force=True)
        return saved
    started = monotonic()
    payload = torch.load(str(phi_artifacts.require(phi_artifacts.codebook(cfg), "codebook")),
                         map_location="cpu", weights_only=False)
    book = codebook_module.Codebook(medoid=payload["medoid"], center=payload["center"],
                                    log_partition=payload["log_partition"],
                                    label=payload["label"])
    items = FrozenItems(cfg)
    result = route_module.run(cfg, arm, items, book, split,
                              batch=int(cfg.phi.route.batch_size),
                              device=runtime.device(cfg.phi.estimator.device), progress=progress)
    phi_artifacts.save_pt(result, phi_artifacts.routes(cfg, split))
    summary = {"split": split, "rows": result["rows"], "targets": result["targets"],
               "targets_with_route": result["targets_with_route"],
               "route_entropy": result["route_entropy"],
               "seconds": round(monotonic() - started, 1)}
    phi_artifacts.record(cfg, "route/%s" % split, identity, summary)
    progress.line("route", json.dumps(summary), force=True)
    return summary


def audit_stage(cfg, progress, split="train"):
    """Assemble every diagnostic the two documents require."""
    from . import audit as audit_module
    from . import codebook as codebook_module
    from .items import FrozenItems, TaskCatalog

    arm = states_module.TeacherArm(cfg)
    items = FrozenItems(cfg)
    catalog = TaskCatalog(cfg, items, lam=float(cfg.phi.estimator.proposal_lambda))
    codebook_payload = torch.load(str(phi_artifacts.require(phi_artifacts.codebook(cfg),
                                                            "codebook")),
                                  map_location="cpu", weights_only=False)
    book = codebook_module.Codebook(medoid=codebook_payload["medoid"],
                                    center=codebook_payload["center"],
                                    log_partition=codebook_payload["log_partition"],
                                    label=codebook_payload["label"])
    book.diagnostics_report = codebook_payload.get("report", {})
    router_payload = torch.load(str(phi_artifacts.require(phi_artifacts.router(cfg), "router")),
                                map_location="cpu", weights_only=False)
    from .router import Router

    router_state = Router(router_payload["state_dict"], router_payload["config"],
                          router_payload.get("validation", {}))
    statistics = torch.load(str(phi_artifacts.require(phi_artifacts.stats(cfg), "estimate")),
                            map_location="cpu", weights_only=False)
    from .estimator import Estimate

    stats_state = Estimate(log_partition=statistics["log_partition"],
                           moment=statistics["moment"], head_mass=statistics["head_mass"],
                           tail_ess=statistics["tail_ess"], head=statistics["head"],
                           tail=statistics["tail"], fallback=statistics.get("fallback", []))
    report_data = audit_module.report(cfg, arm, items, catalog, book, stats_state, router_state,
                                      split=split, progress=progress)
    report_data["summary"] = audit_module.summarize(report_data)
    phi_artifacts.save_json(report_data, phi_artifacts.audit(cfg))
    progress.line("audit", json.dumps(report_data["summary"]), force=True)
    return report_data


def run(cfg, stage: str, split: str = "train"):
    progress = runtime.reporter(stage)
    started = monotonic()
    progress.stage_begin(1, stage, "task=%s split=%s" % (cfg.task, split))
    if stage == "plan":
        result = plan(cfg, progress, split)
    elif stage == "reservoir":
        result = build_reservoir(cfg, progress)
    elif stage == "estimate":
        result = estimate_stats(cfg, progress)
    elif stage == "codebook":
        result = fit_codebook(cfg, progress)
    elif stage == "router":
        result = build_router(cfg, progress)
    elif stage == "tokenize":
        result = tokenize(cfg, progress, split)
    elif stage == "route":
        result = route_labels(cfg, progress, split)
    elif stage == "audit":
        result = audit_stage(cfg, progress, split)
    else:
        raise ValueError("unknown phi stage: %s" % stage)
    progress.stage_end("seconds=%.1f" % (monotonic() - started))
    progress.finish("phi %s done (%.1fs)" % (stage, monotonic() - started))
    return result
