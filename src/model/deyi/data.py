"""Task-neutral DeYi batches over the recommendation datasets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.model.deyi import artifacts
from src.model.tasks import TaskSamples

BEHAVIOR_DIM = 5


@dataclass(frozen=True)
class ProjectedCatalog:
    embeddings: torch.Tensor
    pids: torch.Tensor
    catalog_rows: torch.Tensor
    row_to_catalog: torch.Tensor

    def catalog_index(self, rows):
        physical = torch.as_tensor(rows, dtype=torch.long)
        return self.row_to_catalog.index_select(0, physical)


@dataclass(frozen=True)
class CandidateSampler:
    catalog_size: int
    frequency: torch.Tensor

    def __post_init__(self):
        raw = self.frequency.float().clamp_min(0)
        if raw.numel() != self.catalog_size:
            raise ValueError("frequency does not match task target catalog")
        if float(raw.sum()) <= 0:
            raw = torch.ones_like(raw)
        proposal = 0.5 * raw / raw.sum() + 0.5 / self.catalog_size
        object.__setattr__(self, "proposal", proposal)
        cumulative = torch.cumsum(proposal, 0)
        cumulative[-1] = 1.0
        object.__setattr__(self, "cumulative", cumulative)


@dataclass(frozen=True)
class Sample:
    source_row_idx: int
    uid: int
    history_pids: tuple[int, ...]
    old_pids: tuple[int, ...]
    recent_pids: tuple[int, ...]
    target_pids: tuple[int, ...]
    old_behavior: tuple[tuple[float, ...], ...]
    old_recency: tuple[int, ...]
    target_limit: int


def projected_catalog(cfg, target_rows=None) -> ProjectedCatalog:
    from src.model.deyi import item_space as item_space_module

    manifest = item_space_module.require_item_space(cfg)
    from src.data.dataset import open_dataset

    dataset = open_dataset(str(cfg.data.root), str(cfg.task), artifacts.dataset(cfg))
    pids = dataset.items.pids
    if int(manifest["rows"]) != int(pids.size):
        raise ValueError("DeYi item space manifest disagrees with the item table")
    # The item space keeps no pids copy of its own; the item table is authoritative and the
    # manifest marker plus the row-count check above enforce alignment.
    embeddings = np.load(artifacts.embeddings(cfg), mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[0] != pids.size:
        raise ValueError("projected catalog must be indexed by physical item row")
    if target_rows is None:
        # The frozen legal candidate set of THIS task: document §15 samples the denominator
        # over the task's own catalog, not over a dataset-wide or target-only subset.
        catalog_rows = np.asarray(dataset.task_catalog_rows(cfg.task)).astype(np.int64)
    else:
        catalog_rows = np.unique(np.asarray(target_rows, dtype=np.int64))
        catalog_rows = catalog_rows[catalog_rows > 0]
    row_to_catalog = torch.full((pids.size,), -1, dtype=torch.long)
    rows = torch.from_numpy(np.asarray(catalog_rows, dtype=np.int64))
    row_to_catalog[rows] = torch.arange(rows.numel(), dtype=torch.long)
    return ProjectedCatalog(
        embeddings=torch.from_numpy(embeddings),
        pids=torch.from_numpy(np.asarray(pids[catalog_rows], dtype=np.int64)),
        catalog_rows=rows,
        row_to_catalog=row_to_catalog,
    )


def _behavior_table(cfg, split: str):
    """Behavior channels live inside <dataset>/<task>/<split>.bin."""
    from src.data.dataset import open_dataset

    name = "train" if split == "validation" else split
    return open_dataset(str(cfg.data.root), str(cfg.task),
                        artifacts.dataset(cfg)).split(cfg.task, name).behavior


def behavior_path(cfg, split: str) -> str:
    """Where the behavior channels live (for log messages)."""
    name = "train" if split == "validation" else split
    return "%s/%s/%s/%s.bin[behavior]" % (cfg.data.root, artifacts.dataset(cfg), cfg.task, name)


def behavior_available(cfg, split: str) -> bool:
    """Whether this task split actually carries behavior channels."""
    return _behavior_table(cfg, split) is not None


def _build_samples(cfg, split: str, start: int = 0, stop: int | None = None, *,
                   history_only: bool = False, recent: int | None = None) -> list[Sample]:
    data_root = Path(str(cfg.data.root))
    task = str(cfg.task)
    canonical = TaskSamples(data_root, task, split, dataset=artifacts.dataset(cfg))
    behavior = _behavior_table(cfg, split)
    # ``recent`` is a first-class tunable: the CLI/script value wins, the config value is the
    # default. It decides where old history stops and recent history starts, so it is part of
    # the arm identity (see artifacts.arm_name).
    recent_limit = int(recent if recent is not None else artifacts.recent(cfg))
    target_limit = int(cfg.data.target)
    max_history = int(cfg.max_history[cfg.task])
    if behavior is None:
        # Task/data gap, not a silent default: record it in the run log because the
        # behavior projection stays at its bias and the negative-sampling exclusion
        # set degenerates to "every old item is non-negative feedback".
        print(
            f"[deyi/data] {task}/{split}: no behavior table at {behavior_path(cfg, split)}; "
            "the behavior channel is constant zero for this task",
            flush=True,
        )
    samples: list[Sample] = []
    end = len(canonical) if stop is None else min(int(stop), len(canonical))
    for index in range(int(start), end):
        old_rows, recent_rows = [], []
        old_behaviors, old_recency, all_history = [], [], []
        merged_rows, merged_behaviors = [], []
        # A task's channels are read as one chronological sequence (the dataset declares the
        # order), so the recent window is the last `recent` items of that sequence. The
        # encoder never conditions on channel identity: the event is item + behavior + recency.
        for channel in canonical.channels:
            length = int(canonical.lengths[channel][index])
            rows = np.asarray(canonical.histories[channel][index, :length], dtype=np.int64)
            channel_behavior = None
            if behavior is not None:
                channel_behavior = np.asarray(behavior[index, :length], dtype=np.float32)
            merged_rows.extend(int(row) for row in rows)
            for position in range(len(rows)):
                if channel_behavior is None:
                    merged_behaviors.append((0.0,) * BEHAVIOR_DIM)
                else:
                    values = np.nan_to_num(channel_behavior[position], nan=0.0)
                    values = np.where(values < 0, 0.0, values)
                    merged_behaviors.append(tuple(float(value) for value in values))
        merged_length = len(merged_rows)
        if merged_length > max_history:
            raise ValueError(
                f"{task}/{split} row {index} merged history has {merged_length} items, above "
                f"deyi.max_history={max_history}; raise the configured limit first because "
                "the recency embedding is sized by it"
            )
        all_history.extend(merged_rows)
        cut = max(0, merged_length - recent_limit)
        for position in range(cut):
            old_rows.append(merged_rows[position])
            old_recency.append(merged_length - position)
            old_behaviors.append(merged_behaviors[position])
        for position in range(cut, merged_length):
            recent_rows.append(merged_rows[position])
        target = () if history_only else tuple(
            int(row) for row in canonical.target_rows(index)[:target_limit]
        )
        if not history_only and not target:
            raise ValueError(f"{task}/{split} row {index} has no target item")
        uid_value = int(canonical.uid[index]) if getattr(canonical, "uid", None) is not None else None
        samples.append(
            Sample(
                source_row_idx=int(canonical.source_rows[index]),
                uid=index if uid_value is None else int(uid_value),
                history_pids=tuple(all_history),
                old_pids=tuple(old_rows),
                recent_pids=tuple(recent_rows),
                target_pids=target,
                old_behavior=tuple(old_behaviors),
                old_recency=tuple(old_recency),
                target_limit=target_limit,
            )
        )
    return samples


def _ragged_indices(counts: torch.Tensor):
    owners = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    starts = torch.cumsum(counts, 0) - counts
    return owners, torch.arange(int(counts.sum())) - starts[owners]


def _flat(groups, dtype=torch.long):
    return torch.tensor([item for group in groups for item in group], dtype=dtype)


def _gather(catalog: ProjectedCatalog, rows, width: int, dtype):
    output = torch.zeros((len(rows), width, catalog.embeddings.shape[1]), dtype=dtype)
    flat_values = [row for values in rows for row in values]
    if not flat_values:
        return output
    physical = torch.tensor(flat_values, dtype=torch.long)
    # Read the projected table in ascending row order: the memmap access pattern becomes
    # near-sequential instead of random (the dominant cost of encode/first train pass).
    # The inverse permutation restores the per-user ordering, so values are unchanged.
    order = torch.argsort(physical, stable=True)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), dtype=order.dtype)
    values = catalog.embeddings.index_select(0, physical[order]).to(dtype)[inverse]
    # ``embeddings.npy`` stores the Euclidean PCA coordinate p(i). The consumer is responsible
    # for turning it into the unit semantic vector e(i) the encoder consumes.
    values = torch.nn.functional.normalize(values.float(), dim=-1).to(dtype)
    counts = torch.tensor([len(row) for row in rows])
    owners, within = _ragged_indices(counts)
    output[owners, within] = values
    return output


def collate(samples: list[Sample], catalog: ProjectedCatalog, *, dtype=torch.float32,
            gather_content: bool = True):
    """Assemble one optimizer batch.

    ``gather_content=False`` skips the three content gathers: the caller then materialises
    old/recent/target embeddings on the GPU from its device-resident item table, which removes
    ~0.5 GB of random reads out of the 28 GB host-side mmap per batch (that gather, not the
    negative sampling, is what starved the GPUs once several arms shared a node).
    """
    batch = len(samples)
    old_width = max(1, max(len(sample.old_pids) for sample in samples))
    recent_width = max(1, max(len(sample.recent_pids) for sample in samples))
    target_width = max(1, max(len(sample.target_pids) for sample in samples))
    old_rows = [tuple(sample.old_pids) for sample in samples]
    recent_rows = [tuple(sample.recent_pids) for sample in samples]
    target_rows = [tuple(sample.target_pids) for sample in samples]
    output = {
        "old_rows": torch.zeros((batch, old_width), dtype=torch.long),
        "old_mask": torch.zeros((batch, old_width), dtype=torch.bool),
        "old_behavior": torch.zeros((batch, old_width, BEHAVIOR_DIM)),
        "old_recency": torch.zeros((batch, old_width), dtype=torch.long),
        "recent_rows": torch.zeros((batch, recent_width), dtype=torch.long),
        "recent_mask": torch.zeros((batch, recent_width), dtype=torch.bool),
        "target_rows": torch.zeros((batch, target_width), dtype=torch.long),
        "target_mask": torch.zeros((batch, target_width), dtype=torch.bool),
        "source_row_idx": torch.tensor([sample.source_row_idx for sample in samples]),
    }
    if gather_content:
        output["old_content"] = _gather(catalog, old_rows, old_width, dtype)
        output["recent_content"] = _gather(catalog, recent_rows, recent_width, dtype)
        output["target_content"] = _gather(catalog, target_rows, target_width, dtype)
    old_counts = torch.tensor([len(sample.old_pids) for sample in samples])
    recent_counts = torch.tensor([len(sample.recent_pids) for sample in samples])
    target_counts = torch.tensor([len(sample.target_pids) for sample in samples])
    old_owners, old_within = _ragged_indices(old_counts)
    recent_owners, recent_within = _ragged_indices(recent_counts)
    target_owners, target_within = _ragged_indices(target_counts)
    if old_owners.numel():
        old_catalog = catalog.catalog_index([row for sample in samples for row in sample.old_pids])
        output["old_rows"][old_owners, old_within] = old_catalog.clamp_min(-1) + 1
        output["old_mask"][old_owners, old_within] = True
        output["old_behavior"][old_owners, old_within] = _flat(
            [sample.old_behavior for sample in samples], dtype=torch.float32
        )
        output["old_recency"][old_owners, old_within] = _flat(
            [sample.old_recency for sample in samples]
        )
    if recent_owners.numel():
        recent_catalog = catalog.catalog_index([row for sample in samples for row in sample.recent_pids])
        output["recent_rows"][recent_owners, recent_within] = recent_catalog.clamp_min(-1) + 1
        output["recent_mask"][recent_owners, recent_within] = True
    if target_owners.numel():
        target_catalog = catalog.catalog_index([row for sample in samples for row in sample.target_pids])
        if bool((target_catalog < 0).any()):
            raise ValueError("a target item is absent from the task target catalog")
        output["target_rows"][target_owners, target_within] = target_catalog + 1
        output["target_mask"][target_owners, target_within] = True
    return output


@dataclass(frozen=True)
class SplitTables:
    """Padded per-split arrays, built once instead of rebuilding tensors for every batch.

    ``collate`` used to walk 512 samples in Python for every optimizer batch (262k history rows and
    1.3M behaviour floats per batch), which cost ~1.6 s per batch once several arms shared a node.
    Materialising the padded arrays once per split turns each batch into a single index_select.
    """

    old_rows: np.ndarray
    old_mask: np.ndarray
    old_behavior: np.ndarray
    old_recency: np.ndarray
    recent_rows: np.ndarray
    recent_mask: np.ndarray
    target_rows: np.ndarray
    target_mask: np.ndarray
    source_row_idx: np.ndarray


def build_tables(samples: list[Sample], catalog: ProjectedCatalog) -> SplitTables:
    """One pass over the split: every sample's padded history becomes rows of an array."""
    count = len(samples)
    old_width = max(1, max(len(sample.old_pids) for sample in samples))
    recent_width = max(1, max(len(sample.recent_pids) for sample in samples))
    target_width = max(1, max(len(sample.target_pids) for sample in samples))
    mapping = catalog.row_to_catalog
    tables = SplitTables(
        old_rows=np.zeros((count, old_width), dtype=np.int64),
        old_mask=np.zeros((count, old_width), dtype=bool),
        old_behavior=np.zeros((count, old_width, BEHAVIOR_DIM), dtype=np.float32),
        old_recency=np.zeros((count, old_width), dtype=np.int64),
        recent_rows=np.zeros((count, recent_width), dtype=np.int64),
        recent_mask=np.zeros((count, recent_width), dtype=bool),
        target_rows=np.zeros((count, target_width), dtype=np.int64),
        target_mask=np.zeros((count, target_width), dtype=bool),
        source_row_idx=np.zeros((count,), dtype=np.int64),
    )
    for index, sample in enumerate(samples):
        tables.source_row_idx[index] = int(sample.source_row_idx)
        if sample.old_pids:
            width = len(sample.old_pids)
            rows = mapping[torch.tensor(sample.old_pids, dtype=torch.long)]
            tables.old_rows[index, :width] = rows.numpy().astype(np.int64) + 1
            tables.old_mask[index, :width] = True
            tables.old_behavior[index, :width] = np.asarray(sample.old_behavior, dtype=np.float32)
            tables.old_recency[index, :width] = np.asarray(sample.old_recency, dtype=np.int64)
        if sample.recent_pids:
            width = len(sample.recent_pids)
            rows = mapping[torch.tensor(sample.recent_pids, dtype=torch.long)]
            tables.recent_rows[index, :width] = rows.numpy().astype(np.int64) + 1
            tables.recent_mask[index, :width] = True
        if sample.target_pids:
            width = len(sample.target_pids)
            rows = mapping[torch.tensor(sample.target_pids, dtype=torch.long)]
            if bool((rows < 0).any()):
                raise ValueError("a target item is absent from the task target catalog")
            tables.target_rows[index, :width] = rows.numpy().astype(np.int64) + 1
            tables.target_mask[index, :width] = True
    return tables


def collate_from_tables(tables: SplitTables, indices) -> dict[str, torch.Tensor]:
    """Slice one optimizer batch out of the pre-materialised arrays."""
    index = np.asarray(indices, dtype=np.int64)
    return {
        "old_rows": torch.from_numpy(tables.old_rows[index]),
        "old_mask": torch.from_numpy(tables.old_mask[index]),
        "old_behavior": torch.from_numpy(tables.old_behavior[index]),
        "old_recency": torch.from_numpy(tables.old_recency[index]),
        "recent_rows": torch.from_numpy(tables.recent_rows[index]),
        "recent_mask": torch.from_numpy(tables.recent_mask[index]),
        "target_rows": torch.from_numpy(tables.target_rows[index]),
        "target_mask": torch.from_numpy(tables.target_mask[index]),
        "source_row_idx": torch.from_numpy(tables.source_row_idx[index]),
    }


def catalog_frequency(samples: list[Sample], catalog: ProjectedCatalog) -> torch.Tensor:
    counts = Counter()
    for sample in samples:
        counts.update(sample.target_pids)
    frequency = torch.zeros(catalog.catalog_rows.numel(), dtype=torch.float32)
    for physical, count in counts.items():
        index = int(catalog.row_to_catalog[int(physical)])
        if index >= 0:
            frequency[index] = float(count)
    return frequency


def _sample_seed(base_seed: int, source_row: int) -> int:
    return int(base_seed + 1000003 * source_row) % (2**63 - 1)


def candidate_batch(batch, catalog, frequency, negatives, generator=None, *, seed=None,
                    sampler=None, embeddings=None):
    """Vectorised sampled-softmax occurrences within the task target catalog.

    Same contract as :func:`candidate_batch_reference`, but every step is a batched tensor op on the
    batch's own device: unique positives per row (``unique_consecutive``), the forbidden set
    (targets + recent + interacted history), the draws (one ``rand`` + one ``searchsorted`` for the
    whole row block instead of one call per row) and the membership test (padded ``searchsorted``).
    That removes the per-row Python loop, which was the single largest slice of a training step.

    ``embeddings`` selects the table used to materialise the unique candidate rows: the CPU
    catalogue by default, or a device-resident catalogue (indexed by catalog row) in the GPU path.
    """
    if negatives < 1:
        raise ValueError("negatives must be positive")
    sampler = sampler or CandidateSampler(int(catalog.catalog_rows.numel()), frequency)
    target_rows = batch["target_rows"]
    device = target_rows.device
    rows = int(target_rows.shape[0])
    catalog_size = int(catalog.catalog_rows.numel())
    sentinel = catalog_size                       # padding / rejected slot, never a real candidate

    proposal = sampler.proposal.to(device)
    cumulative = sampler.cumulative.to(device)
    proposal_padded = torch.cat([proposal, proposal.new_zeros(1)])

    target_mask = batch["target_mask"].bool()
    recent_mask = batch["recent_mask"].bool()
    old_mask = batch["old_mask"].bool()
    # Same convention as the reference implementation: old items whose last behaviour bit is *off*
    # belong to the exclusion set.
    old_interacted = old_mask & ~batch["old_behavior"][:, :, -1].bool()

    # --- positives: the row's unique task targets, sorted, padded with the sentinel ---
    padded_targets = torch.where(target_mask, target_rows - 1,
                                 torch.full_like(target_rows, sentinel))
    sorted_targets, _ = padded_targets.sort(dim=1)
    positives = torch.unique_consecutive(sorted_targets, dim=1).contiguous()
    positive_count = (positives < sentinel).sum(dim=1)

    # --- forbidden set per row: positives + recent history + interacted old history ---
    recent = torch.where(recent_mask, batch["recent_rows"] - 1, torch.full_like(batch["recent_rows"], sentinel))
    old = torch.where(old_interacted, batch["old_rows"] - 1, torch.full_like(batch["old_rows"], sentinel))
    combined, _ = torch.cat([positives, recent, old], dim=1).sort(dim=1)
    forbidden = torch.unique_consecutive(combined, dim=1).contiguous()
    allowed_total = (1.0 - proposal_padded[forbidden].sum(dim=1)).clamp_min(1e-12)

    # --- draws: proposal sampling with rejection of the forbidden set ---
    draw_generator = generator or torch.Generator(device=device)
    if generator is None:
        draw_generator.manual_seed(int(seed if seed is not None else 2026))
    rounds = []
    filled = torch.zeros(rows, dtype=torch.long, device=device)
    owners = torch.arange(rows, device=device)[:, None]
    for _ in range(8):
        width = negatives + 64
        uniforms = torch.rand((rows, width), generator=draw_generator, device=device)
        draws = torch.searchsorted(cumulative, uniforms).clamp_max(catalog_size - 1)
        # Left insertion point, exactly like the reference: if the draw is present, the entry at
        # ``positions`` equals it (``right=True`` would point one past it and miss the match).
        positions = torch.searchsorted(forbidden, draws).clamp_max(forbidden.shape[1] - 1)
        accepted = ~(forbidden.gather(1, positions) == draws)
        # Keep the accepted draws in *sampling order* while packing them to the front: sorting by
        # item id here would bias every row towards the smallest sampled items.
        order = accepted.int().cumsum(dim=1) - 1
        compacted = torch.full_like(draws, sentinel)
        compacted[owners.expand_as(draws)[accepted], order[accepted]] = draws[accepted]
        rounds.append(compacted)
        filled = filled + accepted.sum(dim=1)
        if bool((filled >= negatives).all()):
            break
    pooled = torch.cat(rounds, dim=1)
    valid = pooled < sentinel
    order = valid.int().cumsum(dim=1) - 1
    negative_rows = torch.full((rows, negatives), sentinel, dtype=torch.long, device=device)
    keep = valid & (order < negatives)
    columns = order.clamp_max(negatives - 1)
    negative_rows[owners.expand_as(columns)[keep], columns[keep]] = pooled[keep]
    if bool((negative_rows >= sentinel).any()):
        raise ValueError("too few draws survived the task target exclusion set")
    negative_log = torch.log(
        float(negatives) * proposal_padded[negative_rows].clamp_min(1e-30) / allowed_total[:, None]
    )

    # --- assemble the per-row candidate lists: positives first, then the negatives ---
    candidate_rows = torch.cat([positives, negative_rows], dim=1)
    candidate_mask = candidate_rows < sentinel
    positive_mask = torch.arange(candidate_rows.shape[1], device=device)[None, :] < positive_count[:, None]
    negative_log_count = torch.zeros(candidate_rows.shape, dtype=torch.float32, device=device)
    negative_log_count[:, positives.shape[1]:] = negative_log
    negative_log_count = negative_log_count.masked_fill(~candidate_mask, 0.0)

    global_rows = torch.unique(candidate_rows[candidate_mask], sorted=True)
    candidate_index = torch.searchsorted(global_rows, candidate_rows.clamp_max(sentinel))
    candidate_index = candidate_index.clamp_max(global_rows.numel() - 1)
    # Where each original target sits inside the row's sorted unique positive block (that is the
    # mapping the loss gathers by).
    target_index = torch.searchsorted(positives, padded_targets, right=True) - 1
    target_index = target_index.clamp_min(0).masked_fill(~target_mask, 0)

    table = embeddings if embeddings is not None else catalog.embeddings
    if embeddings is not None:
        content = table.index_select(0, global_rows)                 # table indexed by catalog row
    else:
        content = table.index_select(0, catalog.catalog_rows.index_select(0, global_rows))
    content = content.contiguous()
    if content.device.type == "cpu" and torch.cuda.is_available():
        content = content.pin_memory()
    return {
        "content": content,
        # ``pids`` is only metadata (the loss never reads it), so keep the lookup on the host: the
        # catalogue tables live there while ``global_rows`` follows the batch's device.
        "pids": catalog.pids.index_select(0, global_rows.to(catalog.pids.device)),
        "rows": global_rows,
        "candidate_index": candidate_index,
        "candidate_mask": candidate_mask,
        "positive_mask": positive_mask,
        "negative_log_count": negative_log_count,
        "target_index": target_index,
    }


def candidate_batch_reference(batch, catalog, frequency, negatives, generator=None, *, seed=None,
                              sampler=None):
    """Original per-row implementation, kept so the vectorised path can be A/B checked against it."""
    """Build sampled-softmax occurrences within the task target catalog (vectorised).

    Everything below is plain tensor work on the batch's own device, so the same code runs on CPU
    (tests, the standalone benchmark) and on CUDA (training, where it keeps the GPU the bottleneck
    instead of the candidate sampling). Semantics are the reference ones: negatives are drawn from
    the mixed proposal ``0.5 * frequency + 0.5 * uniform``, items in the row's forbidden set
    (task targets, recent history, interacted old history) are rejected, and every negative carries
    the sampled-softmax correction ``log(K * proposal / accepted_mass)``.

    ``embeddings`` selects the table used to materialise the unique candidate rows: the CPU
    catalogue by default, or a device-resident catalogue in the GPU path.
    """
    if negatives < 1:
        raise ValueError("negatives must be positive")
    batch_size = int(batch["target_mask"].shape[0])
    catalog_size = int(catalog.catalog_rows.numel())
    sampler = sampler or CandidateSampler(catalog_size, frequency)
    proposal, cumulative = sampler.proposal, sampler.cumulative
    base_seed = int(seed if seed is not None else (generator.initial_seed() if generator is not None else 2026))
    target_mask = batch["target_mask"].bool()
    forbidden_lists, positives_lists, per_user_target, kept_lists, per_user_rows = [], [], [], [], []
    for user in range(batch_size):
        targets = (batch["target_rows"][user][target_mask[user]] - 1).long()
        positives = torch.unique(targets, sorted=True)
        recent = batch["recent_rows"][user][batch["recent_mask"][user]].long() - 1
        old_valid = batch["old_mask"][user] & ~batch["old_behavior"][user, :, -1].bool()
        old_positive = batch["old_rows"][user][old_valid].long() - 1
        forbidden_lists.append(torch.cat((positives, recent[recent >= 0], old_positive[old_positive >= 0])).unique())
        positives_lists.append(positives)
        per_user_target.append({int(row): index for index, row in enumerate(positives.tolist())})
    forbidden_counts = torch.tensor([int(values.numel()) for values in forbidden_lists])
    forbidden_flat = torch.cat(forbidden_lists)
    forbidden_mass = proposal.index_select(0, forbidden_flat)
    per_user_forbidden = torch.zeros(batch_size).index_add_(
        0, torch.repeat_interleave(torch.arange(batch_size), forbidden_counts), forbidden_mass
    )
    allowed_total = 1.0 - per_user_forbidden
    for user in range(batch_size):
        draw_generator = torch.Generator(device="cpu").manual_seed(
            _sample_seed(base_seed, int(batch["source_row_idx"][user]))
        )
        forbidden = forbidden_lists[user]
        accepted, pieces = 0, []
        for _ in range(8):
            width = max(negatives - accepted + 64, 128)
            draws = torch.searchsorted(cumulative, torch.rand(width, generator=draw_generator)).clamp_max(catalog_size - 1)
            if forbidden.numel():
                ordered = forbidden.sort().values
                positions = torch.searchsorted(ordered, draws)
                clipped = positions.clamp_max(ordered.numel() - 1)
                draws = draws[~((positions < ordered.numel()) & (ordered[clipped] == draws))]
            pieces.append(draws)
            accepted += int(draws.numel())
            if accepted >= negatives:
                break
        if accepted < negatives:
            raise ValueError("too few draws survived the task target exclusion set")
        kept = torch.cat(pieces)[:negatives]
        kept_lists.append(kept)
        per_user_rows.append(torch.cat((positives_lists[user], kept)))
    all_negative_rows = torch.cat(kept_lists)
    all_negative_log = torch.log(
        float(negatives) * proposal.index_select(0, all_negative_rows).clamp_min(1e-30)
        / allowed_total.repeat_interleave(negatives)
    )
    negative_logs, offset = [], 0
    for user, rows in enumerate(per_user_rows):
        values = torch.zeros(rows.numel(), dtype=torch.float32)
        positive_count = len(positives_lists[user])
        values[positive_count:] = all_negative_log[offset : offset + negatives]
        negative_logs.append(values)
        offset += negatives
    global_rows = torch.unique(torch.cat(per_user_rows), sorted=True)
    max_width = max(int(rows.numel()) for rows in per_user_rows)
    candidate_index = torch.zeros((batch_size, max_width), dtype=torch.long)
    candidate_mask = torch.zeros((batch_size, max_width), dtype=torch.bool)
    positive_mask = torch.zeros((batch_size, max_width), dtype=torch.bool)
    negative_log_count = torch.zeros((batch_size, max_width), dtype=torch.float32)
    target_index = torch.zeros_like(batch["target_rows"])
    for user, rows in enumerate(per_user_rows):
        width = int(rows.numel())
        candidate_index[user, :width] = torch.searchsorted(global_rows, rows)
        candidate_mask[user, :width] = True
        positive_mask[user, :len(positives_lists[user])] = True
        negative_log_count[user, :width] = negative_logs[user]
        for position in target_mask[user].nonzero(as_tuple=False).flatten().tolist():
            target_index[user, position] = per_user_target[user][int(batch["target_rows"][user, position] - 1)]
    physical_rows = catalog.catalog_rows.index_select(0, global_rows)
    content = catalog.embeddings.index_select(0, physical_rows).contiguous()
    if content.device.type == "cpu" and torch.cuda.is_available():
        content = content.pin_memory()
    return {
        "content": content,
        "pids": catalog.pids.index_select(0, global_rows),
        "rows": global_rows,
        "candidate_index": candidate_index,
        "candidate_mask": candidate_mask,
        "positive_mask": positive_mask,
        "negative_log_count": negative_log_count,
        "target_index": target_index,
    }


def _build_samples_worker(payload):
    """ProcessPool worker: build the samples of one deterministic row range."""
    cfg, split, start, stop, history_only, recent = payload
    return _build_samples(cfg, split, start, stop, history_only=history_only, recent=recent)


def load_samples(cfg, split: str, *, history_only: bool = False, limit: int | None = None,
                 workers: int | None = None, recent: int | None = None) -> list[Sample]:
    """Build one split's samples, optionally across worker processes.

    Row ranges are fixed and results are concatenated in order, so the output is
    byte-for-byte identical to the single-process path. Workers rebuild their own
    TaskSamples because the memmapped item arrays cannot cross process boundaries.
    """
    from concurrent.futures import ProcessPoolExecutor

    total = len(TaskSamples(Path(str(cfg.data.root)), str(cfg.task), split,
                            dataset=artifacts.dataset(cfg)))
    count = total if limit is None else min(int(limit), total)
    workers = int(cfg.deyi.get("load_workers", 1)) if workers is None else int(workers)
    if workers <= 1 or count < 2048:
        return _build_samples(cfg, split, 0, count, history_only=history_only, recent=recent)
    chunk = max(512, -(-count // (workers * 4)))
    spans = [(start, min(start + chunk, count)) for start in range(0, count, chunk)]
    payloads = [(cfg, split, start, stop, history_only, recent) for start, stop in spans]
    samples: list[Sample] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(_build_samples_worker, payloads):
            samples.extend(part)
    return samples
