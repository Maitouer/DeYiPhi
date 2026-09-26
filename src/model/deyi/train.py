"""Train DeYi: old history -> K predictive interest memories (01_deyi_encoder.md).

``recent`` is a first-class parameter (CLI/script), not a config constant: it decides where
old history stops and recent history starts, so it changes both the objective and the model.
The arm directory carries it (``<task>/k<K>_r<R>``) and every run writes its hyper-parameter
snapshot next to the checkpoint.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

torch.set_float32_matmul_precision("high")

from src.model.deyi import artifacts
from src.model.deyi.data import (
    CandidateSampler,
    build_tables,
    candidate_batch,
    catalog_frequency,
    collate_from_tables,
    load_samples,
    projected_catalog,
)
from src.model.deyi.loss import deyi_objective, user_row_counts
from src.model.deyi.model import DeYi, model_config
from src.model.deyi.runtime import append_jsonl, cosine_scheduler, device, progress, save_pt, seed_all

TRAIN_KEYS = ("old_content", "old_mask", "old_behavior", "old_recency", "recent_content",
              "recent_mask", "target_mask", "old_rows", "recent_rows", "target_rows",
              "source_row_idx", "uid")


def _autocast(target):
    return torch.autocast("cuda", dtype=torch.bfloat16) if target.type == "cuda" else torch.autocast("cpu", enabled=False)


def _move(batch, target):
    return {name: batch[name].to(target, non_blocking=True) for name in TRAIN_KEYS
            if name in batch}


def _slice(batch, start, stop, target):
    return {name: value[start:stop].to(target, non_blocking=True) for name, value in batch.items()}


def _device_content(rows, mask, table):
    """Gather ``e(i)`` for one padded row block from the device-resident item table."""
    content = table[(rows - 1).clamp_min(0)]
    return content.masked_fill(~mask.unsqueeze(-1), 0)


def _validation_key(source_row_idx: int, seed: int) -> int:
    """Splitmix64 mixing; a linear key degenerates into a contiguous prefix."""
    value = (int(source_row_idx) * 0x9E3779B97F4A7C15 + int(seed)) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (value ^ (value >> 31)) % 1_000_000


def split_validation(samples, fraction, seed):
    if fraction <= 0.0:
        return samples, []
    threshold = int(fraction * 1_000_000)
    train_samples, validation_samples = [], []
    for sample in samples:
        key = _validation_key(sample.source_row_idx, seed)
        (validation_samples if key < threshold else train_samples).append(sample)
    if not train_samples or not validation_samples:
        raise ValueError("validation split produced an empty partition")
    return train_samples, validation_samples


def user_negative_index(uids, count, generator):
    """``(B, count)`` row indices of *other* users, -1 where none is available."""
    uids = np.asarray(uids)
    rows = len(uids)
    index = np.full((rows, max(0, int(count))), -1, dtype=np.int64)
    if not count or rows < 2:
        return torch.from_numpy(index)
    for row in range(rows):
        pool = np.flatnonzero(uids != uids[row])
        if not len(pool):
            continue
        take = min(int(count), len(pool))
        index[row, :take] = generator.choice(pool, size=take, replace=False)
    return torch.from_numpy(index)


# Rows of the candidate table copied to the device per step of the build. The table is
# ``|C| x d`` and the biggest task's catalog is ~13M items, so an unchunked build asks for the
# whole fp32 copy plus the whole destination at once, which does not fit on a single card just to
# normalise. The normalisation is row-wise, so chunking is exact -- the result is bit-identical to
# the one-shot version, including its ``to(source dtype)`` rounding.
DEVICE_CANDIDATE_CHUNK = 1 << 20


def device_catalog_table(catalog, chunk=DEVICE_CANDIDATE_CHUNK):
    """The task catalog on the device, L2-normalised row-wise, in the source dtype.

    ``p(i)`` is Euclidean and the encoder consumes ``e(i)``, so normalising once here keeps every
    device-side gather cheap. Only the chunk being normalised is ever materialised in fp32.
    """
    started = time.perf_counter()
    target = device()
    rows, width = int(catalog.catalog_rows.numel()), int(catalog.embeddings.shape[1])
    source = catalog.embeddings.dtype
    table = torch.empty((rows, width), dtype=source, device=target)
    step = max(1, int(chunk))
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        block = catalog.embeddings.index_select(0, catalog.catalog_rows[start:stop]).to(target)
        table[start:stop] = F.normalize(block.float(), dim=-1).to(source)
    print("[deyi/train] device item table %.2f GiB (chunk=%d) in %.1f s"
          % (table.numel() * table.element_size() / float(1 << 30), step,
             time.perf_counter() - started), flush=True)
    return table


def _forward(model, batch, catalog_table):
    if catalog_table is not None:
        # Fill the content back into the batch: the caller's objective needs the same tensors
        # the encoder saw, and the device path is the only one that materialises them.
        batch["old_content"] = _device_content(batch["old_rows"], batch["old_mask"], catalog_table)
        batch["recent_content"] = _device_content(batch["recent_rows"], batch["recent_mask"],
                                                  catalog_table)
    encoded = model(batch["old_content"], batch["old_behavior"], batch["old_recency"],
                    batch["old_mask"], batch["recent_content"], batch["recent_mask"])
    encoded["states"] = torch.nan_to_num(encoded["states"], nan=0.0, posinf=1.0, neginf=-1.0)
    encoded["alpha"] = torch.nan_to_num(encoded["alpha"], nan=0.0)
    return encoded


def _evaluate(model, samples, tables, catalog, frequency, section, target, epoch, step,
              sampler, catalog_table, config):
    if not samples:
        return None
    model.eval()
    totals = {name: 0.0 for name in ("prediction", "user", "diversity", "mean_rank",
                                     "recall_at_1", "recall_at_10", "loss")}
    valid_rows = skipped = 0
    batch_size = int(section.batch_size)
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            stop = start + batch_size
            batch = collate_from_tables(tables, range(start, min(stop, len(samples))))
            gpu = _move(batch, target)
            micro = int(section.micro_batch_size)
            for offset in range(0, len(gpu["old_mask"]), micro):
                end = min(offset + micro, len(gpu["old_mask"]))
                current = _slice(gpu, offset, end, target)
                micro_valid = int(current["target_mask"].sum())
                if micro_valid <= 0:
                    continue
                row_uids = np.asarray([sample.uid for sample in
                                       samples[start + offset:start + end]])
                encoded = _forward(model, current, catalog_table)
                candidates = candidate_batch(
                    current, catalog, frequency, int(section.negatives),
                    seed=int(config.seed) + 100_003 * epoch + 9_176 * (step + start + offset),
                    sampler=sampler, embeddings=catalog_table)
                result = deyi_objective(
                    model, states=encoded["states"], state_mask=encoded["state_mask"],
                    alpha=encoded["alpha"], target_mask=current["target_mask"],
                    target_index=candidates["target_index"].to(target),
                    candidate_content=candidates["content"].to(target),
                    candidate_index=candidates["candidate_index"].to(target),
                    candidate_mask=candidates["candidate_mask"].to(target),
                    positive_mask=candidates["positive_mask"].to(target),
                    negative_log_count=candidates["negative_log_count"].to(target),
                    row_weight=user_row_counts(row_uids).to(target),
                    recent_content=current["recent_content"], recent_mask=current["recent_mask"],
                    other_index=None,
                    item_temperature=float(section.item_temperature),
                    lambda_user=0.0, lambda_div=float(section.lambda_div))
                if not torch.isfinite(result["loss"]):
                    skipped += 1
                    continue
                weight = float(micro_valid)
                for name in totals:
                    totals[name] += weight * float(result[name])
                valid_rows += micro_valid
    model.train()
    if valid_rows <= 0:
        raise ValueError("validation split has no valid target positions")
    if skipped:
        print(f"[deyi/validation] epoch={epoch} skipped {skipped} non-finite micro-batches",
              flush=True)
    return {name: value / valid_rows for name, value in totals.items()}


def train(cfg, resume: bool = False, epochs_per_job: int | None = None) -> dict:
    if not resume and (artifacts.checkpoint(cfg).exists() or any(
            artifacts.states(cfg, split).exists() for split in ("train", "test"))):
        raise FileExistsError(
            "fresh DeYi training needs an unused arm directory: %s" % artifacts.root(cfg))
    seed_all(int(cfg.seed))
    section = cfg.deyi
    # ``recent`` is a task -> window table in the config; the arm's own window is what the sampler
    # needs, and ``artifacts.recent`` is the single reader of that table.
    recent = int(artifacts.recent(cfg))
    total_epochs = int(section.epochs)
    # No chunking: one job runs the full budget. ``--resume`` still continues from the last
    # persisted epoch, and a checkpoint is written after every epoch so a kill costs one epoch.
    chunk_epochs = total_epochs
    checkpoint_path = artifacts.checkpoint(cfg)
    resume_state = None
    if resume and checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved.get("complete"):
            print("[deyi/train] checkpoint already complete; nothing to do", flush=True)
            return {"checkpoint": str(checkpoint_path.resolve()), "steps": int(saved.get("steps", 0)),
                    "epochs_ran": int(saved.get("epochs_ran", 0)), "complete": True, "resumed": True}
        resume_state = saved

    samples = load_samples(cfg, "train", recent=recent)
    print(f"[deyi/train] dataset={artifacts.dataset(cfg)} task={cfg.task} "
          f"k={int(section.num_interests)} recent={recent} rows={len(samples)}", flush=True)
    # Candidate universe = the task's legal catalog (dataset.task_catalog_rows), so negative
    # sampling and the sampled denominator live in the same space the task is evaluated on.
    catalog = projected_catalog(cfg)
    catalog_table = None
    if device().type == "cuda" and bool(section.get("device_candidates", True)):
        catalog_table = device_catalog_table(catalog, int(section.get("device_candidate_chunk",
                                                                      DEVICE_CANDIDATE_CHUNK)))

    train_samples, validation_samples = split_validation(
        samples, float(section.validation_fraction), int(section.validation_seed))
    train_tables = build_tables(train_samples, catalog)
    validation_tables = build_tables(validation_samples, catalog) if validation_samples else None
    frequency = catalog_frequency(train_samples, catalog)
    sampler = CandidateSampler(int(catalog.catalog_rows.numel()), frequency)

    target = device()
    model = DeYi(model_config(cfg)).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(section.learning_rate),
                                  betas=(0.9, 0.95), weight_decay=float(section.weight_decay))
    batch_size = int(section.batch_size)
    micro = int(section.micro_batch_size)
    batches = math.ceil(len(train_samples) / batch_size)
    scheduler = cosine_scheduler(optimizer, batches * total_epochs, float(section.warmup_ratio))
    artifacts.root(cfg).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, artifacts.run_config(cfg))
    log_path = artifacts.train_log(cfg)
    if resume_state is None:
        log_path.write_text("", encoding="utf-8")
    order_generator = torch.Generator().manual_seed(int(cfg.seed))
    negative_generator = np.random.default_rng(int(cfg.seed))
    step, best_validation, best_epoch, stale, best_state, start_epoch = 0, float("inf"), -1, 0, None, 0
    if resume_state is not None:
        model.load_state_dict(resume_state["resume_model_state"])
        optimizer.load_state_dict(resume_state["resume_optimizer_state"])
        start_epoch = int(resume_state.get("epochs_ran", 0))
        step = int(resume_state.get("steps", 0))
        best_epoch = int(resume_state.get("best_epoch", -1))
        stale = int(resume_state.get("resume_stale_epochs", 0))
        best_state = resume_state.get("model_state")
        best_validation = float(resume_state.get("best_validation_prediction") or float("inf"))
        if resume_state.get("resume_order_generator") is not None:
            order_generator.set_state(resume_state["resume_order_generator"])
        for _ in range(step):
            scheduler.step()
    min_epochs = int(section.min_epochs)
    patience = int(section.early_stop_patience)
    min_delta = float(section.early_stop_min_delta)
    skipped_non_finite = 0
    stopped_early = False
    last_epoch = start_epoch - 1
    model.train()

    def persist(complete, epochs_ran, last):
        save_pt({
            "format": model.CHECKPOINT_FORMAT,
            "model_state": best_state if best_state is not None else
            {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "complete": bool(complete),
            "resume_model_state": {name: value.detach().cpu()
                                   for name, value in model.state_dict().items()},
            "resume_optimizer_state": optimizer.state_dict(),
            "resume_stale_epochs": stale,
            "resume_order_generator": order_generator.get_state(),
            "steps": step, "epochs_ran": int(epochs_ran),
            "best_epoch": best_epoch if best_epoch >= 0 else max(last, 0),
            "best_validation_prediction": best_validation if best_state is not None else None,
            "total_epochs": total_epochs, "chunk_epochs": chunk_epochs,
            "design": model.DESIGN, "objective": "pred_plus_user_plus_logvolume",
            "task": str(cfg.task), "recent": recent,
            "num_interests": model.num_interests, "state_dim": model.dim,
            "item_temperature": float(section.item_temperature),
            "route_temperature": float(section.route_temperature),
            "lambda_user": float(section.lambda_user), "lambda_div": float(section.lambda_div),
            "negative_samples": int(section.negatives),
            "user_negatives": int(section.user_negatives),
            "pca_dim": int(cfg.pca.dim),
        }, checkpoint_path)

    for epoch in range(start_epoch, min(start_epoch + chunk_epochs, total_epochs)):
        last_epoch = epoch
        order = torch.randperm(len(train_samples), generator=order_generator).tolist()
        started = time.time()
        for batch_index, start in enumerate(range(0, len(order), batch_size), 1):
            indices = order[start:start + batch_size]
            batch = collate_from_tables(train_tables, indices)
            valid_total = int(batch["target_mask"].sum())
            if valid_total <= 0:
                continue
            gpu = _move(batch, target)
            # ``uid`` lives on the samples, not on the tensor tables; read it from the same
            # index order the tables were built with.
            batch_uids = np.asarray([train_samples[index].uid for index in indices])
            optimizer.zero_grad(set_to_none=True)
            metrics = {name: 0.0 for name in ("loss", "prediction", "user", "diversity",
                                              "mean_rank", "recall_at_1", "recall_at_10")}
            for offset in range(0, len(batch["old_mask"]), micro):
                end = min(offset + micro, len(batch["old_mask"]))
                current = _slice(gpu, offset, end, target)
                micro_valid = int(current["target_mask"].sum())
                if micro_valid <= 0:
                    continue
                # Sample the cross-user negatives INSIDE the micro-batch: ``other_index`` must
                # index this micro-batch's ``states``, not the whole optimizer batch.
                other_index = user_negative_index(
                    batch_uids[offset:end], int(section.user_negatives),
                    negative_generator).to(target)
                encoded = _forward(model, current, catalog_table)
                candidates = candidate_batch(
                    current, catalog, frequency, int(section.negatives),
                    seed=int(cfg.seed) + 100_003 * epoch + 9_176 * step,
                    sampler=sampler, embeddings=catalog_table)
                result = deyi_objective(
                    model, states=encoded["states"], state_mask=encoded["state_mask"],
                    alpha=encoded["alpha"], target_mask=current["target_mask"],
                    target_index=candidates["target_index"].to(target),
                    candidate_content=candidates["content"].to(target),
                    candidate_index=candidates["candidate_index"].to(target),
                    candidate_mask=candidates["candidate_mask"].to(target),
                    positive_mask=candidates["positive_mask"].to(target),
                    negative_log_count=candidates["negative_log_count"].to(target),
                    row_weight=user_row_counts(batch_uids[offset:end]).to(target),
                    recent_content=current["recent_content"], recent_mask=current["recent_mask"],
                    other_index=other_index,
                    item_temperature=float(section.item_temperature),
                    lambda_user=float(section.lambda_user), lambda_div=float(section.lambda_div))
                if not torch.isfinite(result["loss"]):
                    skipped_non_finite += 1
                    continue
                fraction = float(micro_valid) / valid_total
                (fraction * result["loss"]).backward()
                for name in metrics:
                    metrics[name] += fraction * float(result[name].detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(section.gradient_clip))
            optimizer.step()
            scheduler.step()
            append_jsonl(log_path, {"epoch": epoch, "step": step, "rows": len(indices),
                                    "learning_rate": scheduler.get_last_lr()[0],
                                    "design": model.DESIGN, "recent": recent,
                                    "item_temperature": float(section.item_temperature),
                                    "lambda_user": float(section.lambda_user),
                                    "lambda_div": float(section.lambda_div), **metrics})
            progress("deyi-epoch%d" % epoch, batch_index, batches, started,
                     "loss=%.4f recall@10=%.4f" % (metrics["loss"], metrics["recall_at_10"]))
            step += 1
        validation = _evaluate(model, validation_samples, validation_tables, catalog, frequency,
                               section, target, epoch, step, sampler, catalog_table, cfg)
        if validation is not None:
            append_jsonl(log_path, {"kind": "validation", "epoch": epoch, "step": step,
                                    "validation_prediction": validation["prediction"],
                                    "validation_recall_at_10": validation["recall_at_10"],
                                    "validation_mean_rank": validation["mean_rank"]})
            if validation["prediction"] + min_delta < best_validation:
                best_validation = validation["prediction"]
                best_epoch = epoch
                stale = 0
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
            else:
                stale += 1
            persist(False, epoch + 1, epoch)
            if epoch + 1 >= min_epochs and stale >= patience:
                stopped_early = True
                break
    complete = bool(stopped_early or last_epoch + 1 >= total_epochs)
    persist(complete, last_epoch + 1, last_epoch)
    return {"checkpoint": str(checkpoint_path.resolve()), "arm": artifacts.arm_name(cfg),
            "steps": step, "train_rows": len(train_samples),
            "validation_rows": len(validation_samples), "epochs_ran": last_epoch + 1,
            "best_epoch": best_epoch if best_epoch >= 0 else max(last_epoch, 0),
            "complete": complete, "recent": recent,
            "skipped_non_finite_micro_batches": skipped_non_finite}
