"""Exact-coverage, row-balanced training with a device-independent global batch."""

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.model.data import epoch_batches
from . import load_config, parser, saved_config, validate_parallelism
from .data import TigerDataset
from .model import row_losses, row_losses_chain
from .model import Tiger


def microbatches(indices, rank, world, micro):
    """Pad only an incomplete final global batch so each rank takes one step."""
    if not len(indices):
        return []
    size = math.ceil(len(indices) / (world * micro)) * world * micro
    padded = np.resize(indices, size)
    valid = np.arange(size) < len(indices)
    return [(padded[start:start + world * micro][rank::world],
             valid[start:start + world * micro][rank::world]) for start in range(0, size, world * micro)]


def lr_multiplier(step, total, warmup, minimum):
    if step < warmup:
        return (step + 1) / warmup
    fraction = min(1, max(0, (step - warmup) / max(1, total - warmup)))
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * fraction)) / 2


def _mem(tag, device):
    # Memory tracing is diagnostic only; production runs stay quiet unless asked.
    if os.environ.get("TIGER_PROBE") and device.type == "cuda":
        print("[probe] %s allocated=%.2f GiB reserved=%.2f GiB" % (
            tag, torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30),
            flush=True)


def train(cfg, steps=None, resume=False):
    # The saved run config wins over submit-time env vars unless we apply the override
    # here; without this the micro batch silently stays at the configured default.
    cfg.runtime.micro_batch_size = int(os.environ.get("TIGER_MICRO_BATCH",
                                                  cfg.runtime.micro_batch_size))
    cfg.runtime.eval_batch_size = int(os.environ.get("TIGER_EVAL_BATCH",
                                                 cfg.runtime.eval_batch_size))
    print("[tiger/train] micro_batch_size=%d eval_batch_size=%d" %
          (cfg.runtime.micro_batch_size, cfg.runtime.eval_batch_size), flush=True)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(cfg.runtime.device)
    validate_parallelism(cfg, world)
    if world > 1 and int(cfg.train.global_batch_size) != world * int(cfg.runtime.micro_batch_size):
        raise ValueError(
            "Tiger global_batch_size must equal world_size * micro_batch_size; "
            "otherwise full batches contain zero-weight duplicate work"
        )
    torch.set_num_threads(cfg.runtime.cpu_threads)
    if device.type == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        device = torch.device("cuda", torch.cuda.current_device())
        torch.backends.cuda.matmul.allow_tf32 = True
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    torch.manual_seed(cfg.seed)
    data = TigerDataset(cfg, "train")
    raw = Tiger(cfg).to(device)
    initial_interest = raw.interest_embeddings.weight.detach().clone() if raw.phi_v8 else None
    radii = dict(raw.radius_report or {})
    if getattr(data, "deyi_states", None) is not None:
        # Only-initialisation alignment: record where the incoming states started so a later
        # run can tell whether the prefix and the SID tokens share one radius scale.
        sample = data.deyi_states[:4096].float()
        radii["source_radius"] = float(sample.norm(dim=-1).mean())
        radii["source_over_target"] = radii["source_radius"] / float(radii.get("target_radius", 1.0))
        print("[radius] %s" % radii, flush=True)
    if cfg.runtime.gradient_checkpointing:
        options = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
        raw.encoder.gradient_checkpointing_enable(**options)
        raw.decoder.gradient_checkpointing_enable(**options)
    # prefix sampling makes the decoder graph conditional, so a rank can legitimately leave a
    # head's parameters without gradient in a given step; without this DDP aborts with
    # "Expected to have finished reduction in the prior iteration".
    model = DistributedDataParallel(raw, device_ids=[device.index] if device.type == "cuda" else None,
                                    broadcast_buffers=False,
                                    find_unused_parameters=True) if world > 1 else raw
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate,
                                  weight_decay=cfg.train.weight_decay, fused=device.type == "cuda")
    plan = [batch for epoch in range(cfg.train.epochs)
            for batch in epoch_batches(data.lengths, cfg.train.global_batch_size, cfg.seed, epoch,
                                       cfg.train.get("batching", "bucket"))]
    total = len(plan)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(
        step, total, cfg.train.warmup_steps, cfg.train.min_lr_ratio))
    root = Path(cfg.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "checkpoint.pt"
    if (checkpoint.exists() or (root / "train_summary.json").exists()) and not resume:
        raise FileExistsError(f"Use --resume for {root}")
    completed = 0
    if resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        raw.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        completed = state["step"]
    if rank == 0:
        OmegaConf.save(cfg, root / "config.yaml")
    stop = total if steps is None else min(total, steps)
    started = monotonic()

    def publish_summary(peak_by_rank):
        summary = dict(step=completed, total_steps=total, training_complete=completed == total,
                       train_rows=len(data), epochs=cfg.train.epochs, global_batch_size=cfg.train.global_batch_size,
                       positives_per_epoch=int(data.samples.target_lengths.sum()),
                       seconds=monotonic() - started, smoke_rows=cfg.get("smoke_rows"),
                       peak_gpu_memory_gib=peak_by_rank[0] if peak_by_rank else None,
                       peak_gpu_memory_gib_by_rank=peak_by_rank,
                       min_peak_gpu_memory_gib=min(peak_by_rank) if peak_by_rank else None)
        if raw.phi_v8:
            summary['interest_embedding'] = {
                'trainable': bool(raw.interest_embeddings.weight.requires_grad),
                'parameterization': 'independent_token_rows',
                'tokens': raw.interest_embeddings.num_embeddings,
                'initialization_delta_l2': float((raw.interest_embeddings.weight.detach() - initial_interest).float().norm()),
                'prototype_used_in_forward': False}
        (root / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    model.train()
    stream = (root / "train.jsonl").open("a") if rank == 0 else None
    for indices in plan[completed:stop]:
        # Same step/rank stream after a restart; no nondeterministic worker sampler.
        torch.manual_seed(cfg.seed + 10_000_019 * completed + rank)
        optimizer.zero_grad(set_to_none=True)
        micros = microbatches(indices, rank, world, cfg.runtime.micro_batch_size)
        summed = torch.zeros((), device=device)
        for index, (rows, valid) in enumerate(micros):
            batch = data.batch(rows, device)
            if os.environ.get("TIGER_PROBE"):
                print("[probe] rows=%d shapes=%s" % (
                    len(rows), {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)}),
                    flush=True)
            _mem("after batch", device)
            sync = model.no_sync() if world > 1 and index < len(micros) - 1 else nullcontext()
            with sync:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    _mem("before forward", device)
                    outputs = model(**batch)
                    _mem("after forward", device)
                    if isinstance(outputs, list):
                        # phi: the chain is [g1, g2, c0..c3] and its depths have different
                        # vocabularies, so each depth is scored with its own head.
                        # Read the mode from the config, not from ``model``: under DDP the module is
                        # wrapped and ``model.phi_pred`` does not exist on the wrapper.
                        phi_chain = str(cfg.get("phi", {}).get("variant", "")) == "phi"
                        anchor_key = "target_route" if phi_chain else "target_anchor"
                        chain_targets = torch.cat((batch[anchor_key], batch["target_raw"]), dim=1)
                        losses = row_losses_chain(outputs, chain_targets, batch["target_owner"],
                                                  len(rows))
                    else:
                        losses = row_losses(outputs, batch["target_raw"], batch["target_owner"],
                                            len(rows))
                    _mem("after loss", device)
                    loss = losses.sum() if valid.all() else (
                        losses * torch.as_tensor(valid, device=device)
                    ).sum()
                (loss * world / len(indices)).backward()
                summed += loss.detach()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        completed += 1
        should_log = completed == 1 or completed % cfg.train.log_every == 0 or completed == stop
        should_save = completed % cfg.train.save_every == 0 or completed == stop
        if world > 1 and should_log:
            dist.all_reduce(summed)
        peak_by_rank = None
        if should_save and device.type == "cuda":
            local_peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 2 ** 30,
                                      dtype=torch.float32, device=device)
            if world > 1:
                gathered = [torch.empty_like(local_peak) for _ in range(world)]
                dist.all_gather(gathered, local_peak)
                peak_by_rank = [float(value.item()) for value in gathered]
            else:
                peak_by_rank = [float(local_peak.item())]
        if rank == 0 and should_log:
            record = dict(step=completed, total_steps=total, loss=float(summed / len(indices)),
                          gradient_norm=float(norm), seconds=monotonic() - started,
                          micro_batch_size=cfg.runtime.micro_batch_size, world_size=world)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(f"[tiger/{cfg.experiment}] {json.dumps(record)}", flush=True)
        if rank == 0 and should_save:
            temporary = checkpoint.with_suffix(".pending.pt")
            torch.save(dict(model=raw.state_dict(), optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), step=completed, total_steps=total), temporary)
            temporary.replace(checkpoint)
            if completed < total:
                publish_summary(peak_by_rank)
        if world > 1 and should_save:
            dist.barrier()
    # Also finish publication when resuming a completed checkpoint after an
    # interruption between its save and the final inference-model export.
    final_peaks = None
    if completed == total and device.type == "cuda":
        local_peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 2 ** 30,
                                  dtype=torch.float32, device=device)
        if world > 1:
            gathered = [torch.empty_like(local_peak) for _ in range(world)]
            dist.all_gather(gathered, local_peak)
            final_peaks = [float(value.item()) for value in gathered]
        else:
            final_peaks = [float(local_peak.item())]
    if rank == 0 and completed == total:
        temporary = root / "model.pending.pt"
        torch.save({"model": raw.state_dict()}, temporary)
        temporary.replace(root / "model.pt")
        publish_summary(final_peaks)
    if stream is not None:
        stream.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return completed


def main():
    args = parser().parse_args()
    cfg = saved_config(args) if args.resume else load_config(
        args.config, args.task, args.mode, args.run_dir, args.deyi_k,
        args.nproc, args.phi,
        deyi_root=getattr(args, "deyi_root", None),
        phi_deyi_k=getattr(args, "phi_deyi_k", None),
        phi_deyi_arm=getattr(args, "phi_deyi_arm", None),
        phi_v7=getattr(args, "phi_v7", False),
        deyi_v7=getattr(args, "deyi_v7", False),
        phi_v8=getattr(args, "phi_v8", False), deyi_v8=getattr(args, "deyi_v8", False),
        phi_v9=getattr(args, "phi_v9", False), deyi_v9=getattr(args, "deyi_v9", False),
        phi_v9_fast=getattr(args, "phi_v9_fast", False),
        phi_v10_pg=getattr(args, "phi_v10_pg", False),
        phi_pg_root=getattr(args, "phi_pg_root", None),
        phi_no_user_prefix=getattr(args, "phi_no_user_prefix", False),
        phi_v100=getattr(args, "phi_v100", False),
        deyi_v100=getattr(args, "deyi_v100", False)
    )
    if "LOCAL_RANK" not in os.environ and cfg.runtime.nproc > 1:
        subprocess.run([cfg.runtime.python, "-m", "torch.distributed.run", "--standalone",
                        f"--nproc_per_node={cfg.runtime.nproc}", "--module",
                        "src.model.tiger.train", *sys.argv[1:]], check=True)
    else:
        train(cfg, args.steps, args.resume)


if __name__ == "__main__":
    main()
