"""Catalog-constrained beam search and all 30 shared SID/PID metrics."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from src.model.metrics import recommendation_metrics
from . import parser, saved_config, validate_parallelism
from .data import TigerDataset
from .model import Tiger


class PrefixIndex:
    """Sorted integer keys + compact prefix CSR, not a millions-entry Python trie."""
    def __init__(self, codes, pids, width, physical=None):
        self.width = width
        self.weights = width ** np.arange(3, -1, -1, dtype=np.int64)
        keys = codes.astype(np.int64) @ self.weights
        order = np.argsort(keys, kind="stable")
        self.keys, self.pids = keys[order], pids[order]
        # Physical canonical rows of each SID, so callers can attach per-item statistics
        # (e.g. a training-popularity prior) without a second lookup structure.
        self.physical = None if physical is None else np.asarray(physical, dtype=np.int64)[order]
        digits = codes[order]
        self.root = np.unique(digits[:, 0])
        self.levels = []
        for depth in range(1, 4):
            prefix = self.keys // self.weights[depth - 1]
            starts = np.r_[0, np.flatnonzero(np.diff(prefix)) + 1]
            edges = np.r_[0, np.flatnonzero((np.diff(prefix) != 0) | (np.diff(digits[:, depth]) != 0)) + 1]
            offsets = np.searchsorted(edges, np.r_[starts, len(keys)])
            self.levels.append((prefix[starts], offsets, digits[edges, depth]))

    def allowed(self, prefix):
        if not prefix:
            return self.root
        key = 0
        for digit in prefix:
            key = key * self.width + digit
        keys, offsets, values = self.levels[len(prefix) - 1]
        position = np.searchsorted(keys, key)
        return values[offsets[position]:offsets[position + 1]]

    def lookup(self, codes):
        keys = np.asarray(codes, dtype=np.int64) @ self.weights
        positions = np.searchsorted(self.keys, keys)
        if self.physical is None:
            return keys, self.pids[positions]
        return keys, self.pids[positions], self.physical[positions]


class DevicePrefixIndex:
    """GPU-resident view of only the CSR edges needed during constrained beam search."""

    def __init__(self, index, device):
        self.width = int(index.width)
        self.root = torch.as_tensor(index.root, dtype=torch.int16, device=device)
        self.levels = []
        for keys, offsets, values in index.levels:
            self.levels.append((
                torch.as_tensor(keys, dtype=torch.long, device=device),
                torch.as_tensor(offsets, dtype=torch.long, device=device),
                torch.as_tensor(values, dtype=torch.int16, device=device),
            ))

    def candidates(self, prefixes):
        """Return padded allowed digits and an exact validity mask for active prefixes."""
        if prefixes.ndim != 2:
            raise ValueError("prefixes must have shape [parents, depth]")
        parents, depth = prefixes.shape
        if depth == 0:
            return (
                self.root.expand(parents, -1),
                torch.ones((parents, self.root.numel()), dtype=torch.bool, device=prefixes.device),
            )
        if depth > len(self.levels):
            raise ValueError(f"Tiger SID depth exceeds {len(self.levels) + 1}")

        prefix_keys = torch.zeros(parents, dtype=torch.long, device=prefixes.device)
        for digit in prefixes.unbind(dim=1):
            prefix_keys = prefix_keys * self.width + digit.long()
        keys, offsets, values = self.levels[depth - 1]
        positions = torch.searchsorted(keys, prefix_keys)
        safe_positions = positions.clamp_max(keys.numel() - 1)
        known = (positions < keys.numel()) & (keys[safe_positions] == prefix_keys)
        starts, ends = offsets[safe_positions], offsets[safe_positions + 1]
        degrees = (ends - starts).masked_fill(~known, 0)
        slots = torch.arange(self.width, device=prefixes.device)
        valid = slots[None] < degrees[:, None]
        edges = (starts[:, None] + slots).clamp_max(values.numel() - 1)
        return values[edges], valid


def user_g_candidates(codes, mask, width, empty_id):
    counts = torch.zeros((len(codes),width),dtype=torch.long,device=codes.device)
    counts.scatter_add_(1,codes.long(),mask.long())
    allowed = counts > 0
    empty = ~mask.bool().any(1)
    allowed[empty,empty_id] = True
    return allowed


def _ranked_topk(scores, valid, count):
    """Top-k with explicit tie order: input position ascending after score descending."""
    if scores.ndim != 2 or scores.shape != valid.shape:
        raise ValueError("scores and valid must be equally shaped matrices")
    if count < 1 or count > scores.shape[1]:
        raise ValueError("invalid top-k count")
    masked = scores.masked_fill(~valid, float("-inf"))
    # Stable sorting makes beam results portable across CPU/CUDA and BF16 ties.
    selected = torch.argsort(masked, dim=1, descending=True, stable=True)[:, :count]
    return masked.gather(1, selected), selected, valid.gather(1, selected)


@torch.inference_mode()
def generate_reference(model, batch, index, beams):
    """Original host-driven beam implementation retained for equivalence tests."""
    device = batch["history_raw"].device
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        # The batch may carry training-only tensors (e.g. the phi target anchors): the
        # encoder only ever sees its own keyword set.
        encoder = ("history_raw", "history_mask", "continuous_states", "continuous_mask",
                   "history_anchor", "interest_anchor", "interest_anchor_mask",
                   "interest_codes", "interest_mask", "interest_channels")
        hidden, mask = model.encode_history(**{k: v for k, v in batch.items() if k in encoder})
        prefixes = [[()] for _ in range(len(hidden))]
        scores = [torch.zeros(1, device=device) for _ in prefixes]
        for depth in range(4):
            owners = torch.tensor([i for i, rows in enumerate(prefixes) for _ in rows], device=device)
            tokens = torch.tensor([p for rows in prefixes for p in rows], device=device, dtype=torch.long)
            # Four decoder positions only. Recomputing these small prefixes avoids
            # version-specific cache containers and exactly retains causal scoring.
            decoded = model.decode(hidden.index_select(0, owners), mask.index_select(0, owners), tokens)
            logp = model.output_heads[depth](decoded[:, -1]).float().log_softmax(-1)
            cursor = 0
            for row in range(len(prefixes)):
                candidates, values = [], []
                for parent, prefix in enumerate(prefixes[row]):
                    allowed = torch.as_tensor(index.allowed(prefix).astype(np.int64), device=device)
                    value, selected = logp[cursor].index_select(0, allowed).topk(min(beams, len(allowed)))
                    values.append(value + scores[row][parent])
                    candidates.extend([(*prefix, digit) for digit in allowed[selected].tolist()])
                    cursor += 1
                scores[row], selected = torch.cat(values).topk(min(beams, len(candidates)))
                prefixes[row] = [candidates[i] for i in selected.tolist()]
    return prefixes


@torch.inference_mode()
def generate(model, batch, index, beams, device_index=None):
    """Four constrained decode steps with fixed shapes: no per-step device/host synchronisation.

    Every beam slot is expanded at every depth (inactive slots carry ``-inf``), so the search
    never calls ``nonzero()``/``item()`` inside the loop. Returns the beam codes and their
    cumulative log-probabilities, which lets the caller apply a popularity rerank.
    """
    device = batch["history_raw"].device
    device_index = device_index or DevicePrefixIndex(index, device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        # The batch may carry training-only tensors (e.g. the phi target anchors): the
        # encoder only ever sees its own keyword set.
        encoder = ("history_raw", "history_mask", "continuous_states", "continuous_mask",
                   "history_anchor", "interest_anchor", "interest_anchor_mask",
                   "interest_codes", "interest_mask", "interest_channels")
        hidden, mask = model.encode_history(**{k: v for k, v in batch.items() if k in encoder})
        rows = len(hidden)
        hidden = hidden.repeat_interleave(beams, dim=0)
        mask = mask.repeat_interleave(beams, dim=0)
        if ((model.phi or getattr(model, "phi_pred", False))
                and getattr(model, "route_mode", "contextual") != "none"):
            # phi: the region first (unconstrained), then the usual constrained codebook
            # levels. Only the four code digits reach the metrics; the region is ignored there,
            # because the route only has to pick the region the target's SID lives in.
            g_levels=getattr(model,'g_levels',1)
            chain = torch.zeros((rows, beams, g_levels+4), dtype=torch.long, device=device)
            scores = torch.full((rows, beams), float("-inf"), dtype=torch.float32, device=device)
            scores[:, 0] = 0.0
            prior = None
            for depth in range(g_levels+4):
                # ``decode_chain`` expects flat (rows*beams, n) anchor/code prefixes.
                anchors_prev = chain[:, :, :min(depth, g_levels)].reshape(rows * beams, min(depth, g_levels))
                codes_prev = chain[:, :, g_levels:depth].reshape(rows * beams, max(depth - g_levels, 0))
                decoded = model.decode_chain(hidden, mask, anchors_prev, codes_prev)
                logp = model.step_logits(decoded, depth).float().log_softmax(-1)
                if depth == 0:
                    # pi(g|prompt): the row's own region distribution at the first chain position.
                    # All beams of a row share the encoder state, so one column is the whole row.
                    prior = logp.reshape(rows, beams, -1)[:, 0, :].clone()
                if depth < g_levels:
                    width=model.anchor_widths[depth] if g_levels>1 else model.anchor_width
                    digits = torch.arange(width, device=device).unsqueeze(0).expand(
                        rows * beams, -1)
                    allowed = torch.ones_like(digits, dtype=torch.bool)
                    if g_levels==2 and depth==1:
                        allowed=model.fine_to_coarse[digits]==chain[:,:,0].reshape(rows*beams,1)
                    if getattr(model,'user_g_only',False):
                        allowed = user_g_candidates(batch['interest_codes'],batch['interest_mask'],
                            model.anchor_width,model.empty_g_id).repeat_interleave(beams,0)
                else:
                    prefixes = chain[:, :, g_levels:depth].reshape(rows * beams, depth - g_levels)
                    digits, allowed = device_index.candidates(prefixes)
                candidates = digits.shape[1]
                # Padded/inactive候选 use -1; clamp the gather and let ``allowed`` mask them
                local = logp.gather(1, digits.long().clamp_min(0)) + scores.reshape(rows * beams, 1)
                local = local.masked_fill(~allowed, float("-inf")).reshape(rows, beams * candidates)
                scores, selected, _ = _ranked_topk(local, torch.ones_like(local, dtype=torch.bool),
                                                    beams)
                parent = torch.div(selected, candidates, rounding_mode="floor")
                digit = digits.reshape(rows, beams * candidates).gather(1, selected)
                chain = chain.gather(1, parent.unsqueeze(-1).expand(-1, -1, g_levels+4))
                chain[:, :, depth] = digit
            codes = chain[:, :, g_levels:]
            return codes.cpu().numpy(), scores.cpu().numpy()
        codes = torch.zeros((rows, beams, 4), dtype=torch.long, device=device)
        scores = torch.full((rows, beams), float("-inf"), dtype=torch.float32, device=device)
        scores[:, 0] = 0.0

        for depth in range(4):
            prefixes = codes[:, :, :depth].reshape(rows * beams, depth)
            decoded = model.decode(hidden, mask, prefixes)
            logp = model.step_logits(decoded, depth).float().log_softmax(-1)
            digits, allowed = device_index.candidates(prefixes)
            candidates = digits.shape[1]
            local = logp.gather(1, digits.long()) + scores.reshape(rows * beams, 1)
            local = local.masked_fill(~allowed, float("-inf")).reshape(rows, beams * candidates)
            # Stable tie order: score descending, then parent, then catalog digit ascending.
            scores, selected, _ = _ranked_topk(local, torch.ones_like(local, dtype=torch.bool), beams)
            parent = torch.div(selected, candidates, rounding_mode="floor")
            # ``selected`` indexes the flattened (parent, digit) grid, so the digit tensor must
            # be flattened the same way; gathering with a parent-local index silently mixes
            # digits across parents and produces SIDs that are not in the catalog.
            digit = digits.reshape(rows, beams * candidates).gather(1, selected)
            codes = codes.gather(1, parent.unsqueeze(-1).expand(-1, -1, 4))
            codes[:, :, depth] = digit

    return codes.cpu().numpy(), scores.cpu().numpy()


def evaluation_manifest(root, cfg, split, rows):
    model = root / "model.pt"
    state = model.stat()
    return {
        "format": "tiger-eval-fixed-shape-csr-v2",
        "split": str(split),
        "rows": int(rows),
        "beams": int(cfg.evaluation.beams),
        "popularity_alpha": float(cfg.evaluation.get("popularity_alpha", 0.0)),
        "codebook_width": int(cfg.codebook.width),
        "model_bytes": int(state.st_size),
        "model_mtime_ns": int(state.st_mtime_ns),
        "tie_break": "score_desc_then_parent_then_catalog_digit",
        "phi_variant": str(cfg.get('phi', {}).get('variant', '')),
        "native_dedup": str(cfg.get('phi', {}).get('variant', '')) == 'v10_pg',
        "user_g_only": bool(cfg.get('phi', {}).get('user_g_only', False)),
        "hierarchical_g": bool(cfg.get('phi', {}).get('hierarchical', False)),
    }


def window_items(data):
    """Recent item count per row: v100 has no channel window, so the row length is it."""
    window = getattr(data, 'window', None)
    return np.asarray(window.budgets.sum(1) if window is not None else data.lengths)


def training_popularity(cfg, rows):
    """Per physical row: frequency in the *training* targets (never the evaluated split)."""
    train = TigerDataset(cfg, "train")
    targets = np.asarray(train.samples.target, dtype=np.int64)
    lengths = np.asarray(train.samples.target_lengths, dtype=np.int64)
    mask = np.arange(targets.shape[1])[None] < lengths[:, None]
    return np.bincount(targets[mask], minlength=rows)


def publish_shard_manifest(path, manifest, rank, world):
    """Publish a small identity record before distributed ranks begin resuming shards."""
    error = [None]
    if rank == 0:
        if path.exists():
            current = json.loads(path.read_text())
            if current != manifest:
                error[0] = f"stale Tiger evaluation shards at {path.parent}; use a new run directory"
        else:
            temporary = path.with_suffix(".pending.json")
            temporary.write_text(json.dumps(manifest, indent=2) + "\n")
            temporary.replace(path)
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise ValueError(error[0])


def distinct_native_codes(codes, scores):
    """Drop inactive beams and retain the highest original beam score per native SID."""
    seen, kept = set(), []
    for i in np.argsort(-np.asarray(scores), kind='stable'):
        if not np.isfinite(scores[i]):
            continue
        key = tuple(int(x) for x in codes[i])
        if key not in seen:
            seen.add(key)
            kept.append(int(i))
    if not kept:
        raise ValueError('Beam produced no finite native item')
    return codes[kept]


def evaluate(cfg):
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(cfg.runtime.device)
    validate_parallelism(cfg, world)
    torch.set_num_threads(cfg.runtime.cpu_threads)
    if device.type == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        device = torch.device("cuda", torch.cuda.current_device())
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    root = Path(cfg.run_dir)
    summary = json.loads((root / "train_summary.json").read_text())
    if not summary["training_complete"]:
        raise ValueError("Finish the full training budget before evaluating")
    state = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
    split = cfg.evaluation.get("split", "test")
    data = TigerDataset(cfg, split)
    model = Tiger(cfg).to(device).eval()
    model.load_state_dict(state["model"])
    del state
    from src.data.dataset import catalog_rows

    catalog = np.asarray(catalog_rows(str(cfg.data_dir), str(cfg.task), cfg.dataset))
    index = PrefixIndex(data.codes[catalog], data.pids[catalog], cfg.codebook.width, catalog)
    device_index = DevicePrefixIndex(index, device)
    print("[tiger/eval] beams=%d rows=%d split=%s"
          % (int(cfg.evaluation.beams), len(data), split), flush=True)
    folder = root / ("eval" if split == "test" else f"eval_{split}")
    shards = folder / "shards-fixed-shape-csr-v2"
    shards.mkdir(parents=True, exist_ok=True)
    publish_shard_manifest(
        shards / "manifest.json", evaluation_manifest(root, cfg, split, len(data)), rank, world
    )
    size = cfg.evaluation.shard_rows
    for shard, start in enumerate(range(0, len(data), size)):
        path = shards / f"{start:08d}.parquet"
        if shard % world != rank or path.exists():
            continue
        records = []
        for offset in range(start, min(start + size, len(data)), cfg.runtime.eval_batch_size):
            rows = list(range(offset, min(offset + cfg.runtime.eval_batch_size, start + size, len(data))))
            predicted_codes, predicted_scores = generate(
                model, data.batch(rows, device, training=False), index, cfg.evaluation.beams,
                device_index
            )
            for position, row in enumerate(rows):
                native_codes = (distinct_native_codes(predicted_codes[position], predicted_scores[position])
                                if model.phi_pg else predicted_codes[position])
                sid, pid, _ = index.lookup(native_codes)
                if not model.phi_pg and (getattr(model, 'phi_v7', False) or getattr(model, 'phi_v8', False) or data.deyi_v7 or data.deyi_v8) and len(np.unique(sid)) != int(cfg.evaluation.beams):
                    raise ValueError('v7 beam must contain 32 distinct legal native SIDs')
                target = data.samples.target_rows(row)
                records.append(dict(source_row_idx=int(data.samples.source_rows[row]),
                                    predicted_sid=sid.tolist(), predicted_pid=pid.tolist(),
                                    target_sid=(data.codes[target].astype(np.int64) @ index.weights).tolist(),
                                    target_pid=data.pids[target].tolist()))
        table = pa.Table.from_pylist(records)
        temporary = path.with_suffix(".pending.parquet")
        pq.write_table(table, temporary)
        temporary.replace(path)
        print(f"[tiger/eval] rank={rank} rows={start + len(records)}/{len(data)}", flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        files = [shards / f"{start:08d}.parquet" for start in range(0, len(data), size)]
        table = pa.concat_tables([pq.read_table(path) for path in files])
        spaces = ("sid", "pid")
        scores = recommendation_metrics({s: table[f"predicted_{s}"].to_pylist() for s in spaces},
                                        {s: table[f"target_{s}"].to_pylist() for s in spaces},
                                        cfg.evaluation.eval_ks)
        pq.write_table(table, folder / "predictions.parquet")
        pq.write_table(pa.table({"source_row_idx": table["source_row_idx"], **scores}), folder / "per_sample.parquet")
        result = dict(experiment=cfg.experiment, model="tiger", complete=len(table) == len(data),
                      evaluation_split=split, cohort=cfg.get("cohort"),
                      rows=len(data), generated_rows=len(table), smoke_rows=cfg.get("smoke_rows"),
                      prompt_style="tiger_sid", beam_backend="vectorized_csr_v1",
                      tie_break="score_desc_then_parent_then_catalog_digit",
                      metrics={key: float(value.mean()) for key, value in scores.items()})
        if model.phi_pg:
            unique_counts = np.array([len(x) for x in table['predicted_sid'].to_pylist()])
            result.update(generation_contract='G_plus_native_4_digit_relaxed_G_then_native_dedup',
                interest_prefix_tokens=4 if model.pg_user_prefix else 0, item_interest_tokens=1, raw_beam_width=int(cfg.evaluation.beams),
                mean_unique_items=float(unique_counts.mean()), min_unique_items=int(unique_counts.min()),
                duplicate_or_inactive_fraction=float(1-unique_counts.mean()/int(cfg.evaluation.beams)),
                mean_history_items=float(window_items(data).mean()),
                mean_encoder_tokens=float((6*window_items(data)+(data.v7_mask.sum(1) if model.pg_user_prefix else 0)).mean()))
            if model.user_g_only:
                result.update(generation_contract='user_G_only_then_native_SID_dedup',
                              item_assignment='user_candidates',empty_g_id=model.empty_g_id)
            if model.hierarchical:
                result.update(generation_contract='G1_G2_ancestry_then_relaxed_native_SID_dedup',
                    interest_prefix_tokens=8,item_interest_tokens=2,
                    g1_width=model.anchor_widths[0],g2_width=model.anchor_widths[1],
                    mean_encoder_tokens=float((7*window_items(data)+2*data.v7_mask.sum(1)).mean()))
        elif getattr(model, 'phi_v7', False) or getattr(model, 'phi_v8', False) or data.deyi_v7 or data.deyi_v8:
            result['generation_contract'] = 'native_4_digit_unique_sid'
            result['interest_prefix_tokens'] = 4
            result['mean_history_items'] = float(window_items(data).mean())
            result['mean_encoder_tokens'] = float((5 * data.window.budgets.sum(1) + (data.deyi_mask.numpy() if (data.deyi_v7 or data.deyi_v8) else data.v7_mask).sum(1)).mean())
        (folder / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    if world > 1:
        dist.destroy_process_group()


def main():
    args = parser().parse_args()
    cfg = saved_config(args)
    if "LOCAL_RANK" not in os.environ and cfg.runtime.nproc > 1:
        subprocess.run([
            cfg.runtime.python, "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={cfg.runtime.nproc}", "--module",
            "src.model.tiger.evaluate", *sys.argv[1:],
        ], check=True)
    else:
        evaluate(cfg)


if __name__ == "__main__":
    main()
