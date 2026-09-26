"""DeYi encoder: old history -> K unit-norm predictive interest memories.

Implements ``00-Reconstruct/01_deyi_encoder.md``:

    H_old --Contextualize--> U --Retrieve & Refine--> M --Normalize--> Z

with two invariants the code makes explicit rather than incidental:

* ``Query Identity != Memory Content`` — the learnable anchors only build the query used to
  ask the history; the exported memory is always something read out of ``U``.
* old history never mixes with recent history — ``Z`` is a function of the old history only;
  recent history is used once, to pick which memory to read.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def model_config(cfg):
    """The flat namespace ``DeYi`` reads, built from one source for train and encode.

    Keeping this in the model module means a second caller can never construct the encoder
    from a differently-shaped config (``cfg.deyi`` carries ``state_dim``, not ``dim``).
    """
    from omegaconf import OmegaConf

    return OmegaConf.create({
        "dim": int(cfg.deyi.state_dim),
        "num_interests": int(cfg.deyi.num_interests),
        "max_history": int(cfg.max_history[cfg.task]),
        "max_recent": int(cfg.recent[cfg.task]),
        "history_layers": int(cfg.deyi.history_layers),
        "interest_layers": int(cfg.deyi.interest_layers),
        "heads": int(cfg.deyi.heads),
        "ffn": int(cfg.deyi.ffn),
        "dropout": float(cfg.deyi.dropout),
        "temporal_buckets": int(cfg.deyi.temporal_buckets),
        "behavior_dim": int(cfg.deyi.behavior_dim),
        "route_dim": int(cfg.deyi.route_dim),
        "route_temperature": float(cfg.deyi.route_temperature),
        # Which H_old -> M mapping this run instantiates (04_baseline.md). "deyi" is the
        # production encoder; the two below are the learned controlled baselines.
        "method": str(cfg.deyi.get("method", "deyi")),
    })


def temporal_bucket_index(width: int, half: int) -> torch.Tensor:
    """Signed logarithmic buckets over event order distance ``t - s``.

    Index 0 is the diagonal (same event); positive distances get ``1..half`` and negative
    ones ``half+1..2*half``, so a per-head bias table distinguishes "before" from "after"
    instead of collapsing both into an absolute distance.
    """
    order = torch.arange(width)
    delta = order[None, :] - order[:, None]
    magnitude = delta.abs().clamp(min=1)
    bucket = torch.floor(torch.log2(magnitude.float())).long().clamp(max=half - 1)
    sign = (delta < 0).long()
    index = sign * half + bucket + 1
    return torch.where(delta == 0, torch.zeros_like(index), index)


class BiasedHistoryLayer(nn.Module):
    """Bidirectional self-attention plus a learned per-head relative temporal bias."""

    def __init__(self, hidden: int, heads: int, ffn: int, dropout: float, half_buckets: int):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.attention_norm = nn.LayerNorm(hidden)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)
        self.bias = nn.Embedding(2 * half_buckets + 1, heads)
        nn.init.zeros_(self.bias.weight)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn, bias=False), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn, hidden, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, states, mask, bucket_index, causal=False):
        batch, width, _ = states.shape
        values = self.attention_norm(states)
        shape = (batch, width, self.heads, self.head_dim)
        query = self.query(values).view(shape).transpose(1, 2)
        key = self.key(values).view(shape).transpose(1, 2)
        value = self.value(values).view(shape).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores + self.bias(bucket_index).permute(2, 0, 1)[None]
        if causal:
            # Chronicle-Core (04_baseline.md §6.3): the state at an anchor may only depend on the
            # prefix available at that anchor, so the attention is lower-triangular on top of the
            # ordinary padding mask.
            allowed = torch.ones(width, width, dtype=torch.bool, device=scores.device).tril()
            scores = scores.masked_fill(~allowed, float("-inf"))
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        update = (weights @ value).transpose(1, 2).reshape(batch, width, self.hidden)
        states = states + self.dropout(self.output(update))
        return states + self.dropout(self.ffn(self.ffn_norm(states)))


class InterestLayer(nn.Module):
    """One ``Construct Query -> Retrieve Evidence -> Update Memory`` step."""

    def __init__(self, hidden: int, heads: int, ffn: int, dropout: float, first: bool):
        super().__init__()
        self.first = bool(first)
        if not self.first:
            self.memory_to_query = nn.LayerNorm(hidden)
            self.memory_to_query_projection = nn.Linear(hidden, hidden, bias=False)
        self.query_norm = nn.LayerNorm(hidden)
        self.cross_query_norm = nn.LayerNorm(hidden)
        self.cross_history_norm = nn.LayerNorm(hidden)
        self.cross = nn.MultiheadAttention(hidden, heads, dropout=dropout, bias=False,
                                           batch_first=True)
        self.evidence_projection = nn.Linear(hidden, hidden, bias=False)
        self.memory_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn, bias=False), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn, hidden, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, anchors, memory, history, history_mask):
        if self.first:
            query = anchors
        else:
            query = self.query_norm(
                anchors + self.memory_to_query_projection(self.memory_to_query(memory))
            )
        history_values = self.cross_history_norm(history)
        evidence = self.cross(
            self.cross_query_norm(query), history_values, history_values,
            key_padding_mask=~history_mask, need_weights=False,
        )[0]
        accumulated = memory + self.evidence_projection(evidence)
        return accumulated + self.dropout(self.ffn(self.memory_norm(accumulated)))


class DeYi(nn.Module):
    CHECKPOINT_FORMAT = "deyi_reconstruct_v1"
    STATE_FORMAT = "deyi_predictive_memory_states_v1"
    RADIUS_ALIGNMENT = "none_in_deyi_export"
    DESIGN = "contextualize_retrieve_refine_v1"

    def __init__(self, cfg) -> None:
        super().__init__()
        self.dim = int(cfg.dim)
        self.num_interests = int(cfg.num_interests)
        if self.num_interests < 1:
            raise ValueError("num_interests must be positive")
        self.max_history = int(cfg.max_history)
        self.max_recent = int(cfg.max_recent)
        self.history_layers = int(cfg.history_layers)
        self.interest_layers = int(cfg.interest_layers)
        self.heads = int(cfg.heads)
        self.ffn = int(cfg.ffn)
        dropout = float(cfg.dropout)
        self.behavior_dim = int(cfg.behavior_dim)
        self.half_buckets = int(cfg.get("temporal_buckets", 64))

        self.behavior_projection = nn.Linear(self.behavior_dim, self.dim, bias=False)
        self.recency_embedding = nn.Embedding(self.max_history + 1, self.dim, padding_idx=0)
        self.event_norm = nn.LayerNorm(self.dim)
        self.buckets = None

        self.history_blocks = nn.ModuleList([
            BiasedHistoryLayer(self.dim, self.heads, self.ffn, dropout, self.half_buckets)
            for _ in range(self.history_layers)
        ])
        self.anchors = nn.Parameter(torch.empty(1, self.num_interests, self.dim))
        nn.init.normal_(self.anchors, std=0.02)
        self.interest_blocks = nn.ModuleList([
            InterestLayer(self.dim, self.heads, self.ffn, dropout, first=(index == 0))
            for index in range(self.interest_layers)
        ])
        self.output_norm = nn.LayerNorm(self.dim)

        # Recent-conditioned readout: a low-capacity semantic mean, deliberately not a causal
        # transformer head, so the predictive signal has to live in Z rather than in a strong
        # recent reader.
        self.route_dim = int(cfg.route_dim)
        self.route_temperature = float(cfg.route_temperature)
        self.read_recent = nn.Linear(self.dim, self.route_dim, bias=False)
        self.read_state = nn.Linear(self.dim, self.route_dim, bias=False)

        self.method = str(cfg.get("method", "deyi"))
        if self.method not in ("deyi", "chronicle", "hicogen"):
            raise ValueError("unknown memory method %r" % self.method)
        if self.method == "hicogen":
            # Shared cluster readout of 04_baseline.md §5.3: one projection and one query for
            # every cluster, so all four states come from the same readout geometry.
            self.cluster_projection = nn.Linear(self.dim, self.dim, bias=False)
            self.cluster_query = nn.Parameter(torch.empty(self.dim))
            nn.init.normal_(self.cluster_query, std=0.02)
            self.cluster_temperature = float(cfg.get("cluster_temperature", 1.0))

    # ------------------------------------------------------------------ forward
    def _bucket_index(self, width: int, device):
        if self.buckets is None or self.buckets.shape[0] < width:
            self.buckets = temporal_bucket_index(max(width, 1), self.half_buckets)
        return self.buckets[:width, :width].to(device)

    def _events(self, content, behavior, recency):
        # ``content`` is e(i): the consumer already L2-normalized the PCA coordinate.
        values = content + self.behavior_projection(behavior) + self.recency_embedding(recency)
        return self.event_norm(values)

    def contextualize(self, content, behavior, recency, mask):
        mask = mask.bool()
        has_history = mask.any(-1)
        safe_mask = mask.clone()
        # A row with no events at all would make softmax see an all-masked row; keep one key
        # alive so the block stays finite. The output is zeroed below anyway.
        safe_mask[:, 0] |= ~has_history
        states = self._events(content, behavior, recency) * mask.unsqueeze(-1)
        index = self._bucket_index(states.shape[1], states.device)
        for block in self.history_blocks:
            states = block(states, safe_mask, index)
        return states, safe_mask, has_history

    def interests(self, history, history_mask):
        memory = torch.zeros(history.shape[0], self.num_interests, self.dim,
                             device=history.device, dtype=history.dtype)
        anchors = self.anchors.expand(history.shape[0], -1, -1)
        for block in self.interest_blocks:
            memory = block(anchors, memory, history, history_mask)
        return memory

    def readout(self, states, state_mask, recent_content, recent_mask):
        weight = recent_mask.unsqueeze(-1).to(states.dtype)
        count = weight.sum(1).clamp_min(1.0)
        mean = (recent_content * weight).sum(1) / count
        recent_key = F.normalize(self.read_recent(mean.float()), dim=-1)
        state_key = F.normalize(self.read_state(states.float()), dim=-1)
        logits = (recent_key[:, None, :] * state_key).sum(-1) / self.route_temperature
        alpha = torch.softmax(logits, dim=-1)
        # Old history non-empty but recent empty: the readout has no opinion, so fall back to
        # a uniform mixture over the slots.
        has_recent = recent_mask.bool().any(-1)
        uniform = torch.full_like(alpha, 1.0 / self.num_interests)
        alpha = torch.where(has_recent[:, None], alpha, uniform)
        return alpha * state_mask.to(alpha.dtype), logits

    # -------------------------------------------------------------- controlled baselines
    def _encode_chronicle(self, content, behavior, recency, mask):
        """Chronicle-Core: four temporally anchored causal states (04_baseline.md §6).

        Four learned query tokens are **interleaved after their anchors** — appending them at the
        end would let ``q_1`` attend to the whole history and void the prefix-only semantics of
        §6.3 — and the states are the hidden states at those query positions.
        """
        mask = mask.bool()
        has_history = mask.any(-1)
        events = self._events(content, behavior, recency)
        rows, width, dim = events.shape
        steps = torch.arange(1, self.num_interests + 1, device=events.device)
        length = mask.sum(-1)
        anchors = torch.ceil(length[:, None].float() * steps[None, :].to(events.dtype)
                             / float(self.num_interests)).long()
        total = width + self.num_interests
        combined = events.new_zeros(rows, total, dim)
        combined_mask = torch.zeros(rows, total, dtype=torch.bool, device=events.device)
        events_index = torch.arange(1, width + 1, device=events.device)
        inserted = (anchors[:, None, :] < events_index[None, :, None]).sum(-1)     # (B, W)
        event_position = (events_index - 1)[None, :] + inserted
        query_position = anchors + steps[None, :] - 1
        combined.scatter_(1, event_position[..., None].expand(-1, -1, dim), events)
        combined_mask.scatter_(1, event_position, mask)
        query_valid = (anchors >= 1) & has_history[:, None]
        combined.scatter_(1, query_position[..., None].expand(-1, -1, dim),
                          self.anchors.expand(rows, -1, -1))
        combined_mask.scatter_(1, query_position, query_valid)
        states = combined * combined_mask.unsqueeze(-1)
        safe_mask = combined_mask.clone()
        safe_mask[:, 0] |= ~has_history
        index = self._bucket_index(total, events.device)
        for block in self.history_blocks:
            states = block(states, safe_mask, index, causal=True)
        gathered = states.gather(1, query_position[..., None].expand(-1, -1, dim))
        memory = F.normalize(self.output_norm(gathered).float(), dim=-1).to(gathered.dtype)
        state_mask = query_valid
        return {"states": memory * state_mask.unsqueeze(-1), "state_mask": state_mask}

    def _agglomerative(self, content, mask):
        """Average-linkage agglomerative clustering on cosine distance -> <= K clusters.

        ``04_baseline.md`` §5.2, run inside the model so no extra data stage is needed: ``W <= 84``
        for the product task, so the O(W^2) distance matrix and the ``W - K`` Lance-Williams merges
        are cheap. Repeated events are kept (no de-duplication) and only ``H_old`` is read.
        """
        unit = F.normalize(content.float(), dim=-1)
        rows, width, _ = unit.shape
        finite = mask[:, :, None] & mask[:, None, :]
        # Explicit float32: under autocast the matmul returns bf16 while ``size`` stays fp32, and
        # the Lance-Williams update would then mix dtypes into the in-place scatter.
        distance = (1.0 - unit @ unit.transpose(-2, -1)).float().masked_fill(~finite, float("inf"))
        distance.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))
        size = mask.float()
        labels = torch.arange(width, device=unit.device).expand(rows, width).clone()
        labels = labels.masked_fill(~mask, -1)
        target = min(self.num_interests, width)
        for _ in range(width):
            flat = distance.reshape(rows, -1)
            index = flat.argmin(-1)
            # A row stops as soon as it holds ``target`` clusters. The loop bound is global, so
            # without this a short history would keep merging well past K clusters.
            live = (torch.isfinite(flat.gather(-1, index[:, None])[:, 0])
                    & (size.gt(0).sum(-1) > target))
            if not bool(live.any()):
                break
            left = torch.where(live, index // width, torch.zeros_like(index))
            right = torch.where(live, index % width, torch.zeros_like(index))
            left_size = size.gather(1, left[:, None])
            right_size = size.gather(1, right[:, None])
            left_row = distance.gather(1, left[:, None, None].expand(rows, 1, width)).squeeze(1)
            right_row = distance.gather(1, right[:, None, None].expand(rows, 1, width)).squeeze(1)
            merged = ((left_size * left_row + right_size * right_row)
                      / (left_size + right_size).clamp_min(1e-9)).float()
            merged = merged.masked_fill(~live[:, None], float("inf"))
            # Explicit source shapes: ``scatter_`` does not broadcast a (rows, width) source into
            # the (rows, 1, width) / (rows, width, 1) index of the symmetric update.
            merged_row, merged_column = merged.unsqueeze(1), merged.unsqueeze(-1)
            index_row = left[:, None, None].expand(rows, 1, width)
            index_column = left[:, None, None].expand(rows, width, 1)
            absent_row = torch.full_like(merged_row, float("inf"))
            absent_column = torch.full_like(merged_column, float("inf"))
            distance.scatter_(1, index_row, merged_row)
            distance.scatter_(2, index_column, merged_column)
            distance.scatter_(1, right[:, None, None].expand(rows, 1, width), absent_row)
            distance.scatter_(2, right[:, None, None].expand(rows, width, 1), absent_column)
            diagonal = left[:, None, None].expand(rows, 1, 1)
            distance.scatter_(1, diagonal, torch.full_like(merged_row[:, :, :1], float("inf")))
            distance.scatter_(2, diagonal, torch.full_like(merged_column[:, :1, :], float("inf")))
            size.scatter_(1, left[:, None],
                          (left_size + right_size).masked_fill(~live[:, None], 0.0))
            size.scatter_(1, right[:, None], torch.zeros_like(right_size))
            labels = torch.where((labels == right[:, None]) & live[:, None],
                                 left[:, None], labels)
        return labels

    def _encode_hicogen(self, content, mask):
        """HiCoGen-Core: semantic clusters read out by one shared attention (04_baseline.md §5).

        Clusters are ordered by their most recent member (§5.3) so the slot embedding keeps a
        deterministic meaning, and a row with fewer clusters than slots is padded with zeros.
        """
        mask = mask.bool()
        has_history = mask.any(-1)
        rows, width, dim = content.shape
        unit = F.normalize(content.float(), dim=-1)
        labels = self._agglomerative(content, mask)
        positions = torch.arange(width, device=unit.device).expand(rows, width)
        last_of_cluster = torch.full((rows, width), -1, dtype=torch.long, device=unit.device)
        last_of_cluster.scatter_reduce_(
            1, labels.clamp_min(0),
            torch.where(labels >= 0, positions, torch.full_like(positions, -1)),
            reduce="amax", include_self=True)
        order = torch.argsort(-last_of_cluster, dim=-1, stable=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(width, device=unit.device).expand(rows, width))
        slot_of_cluster = torch.where(last_of_cluster < 0, torch.full_like(rank, -1), rank)
        slot = slot_of_cluster.gather(1, labels.clamp_min(0))
        score = (self.cluster_projection(unit) @ self.cluster_query) / self.cluster_temperature
        memory = unit.new_zeros(rows, self.num_interests, dim)
        state_mask = torch.zeros(rows, self.num_interests, dtype=torch.bool, device=unit.device)
        for candidate in range(self.num_interests):
            member = mask & (slot == candidate)
            occupied = member.any(-1)
            if not bool(occupied.any()):
                continue
            weights = torch.softmax(score.masked_fill(~member, float("-inf")), dim=-1)
            weights = torch.nan_to_num(weights.float(), nan=0.0)
            memory[:, candidate] = torch.einsum("bw,bwd->bd", weights, unit)
            state_mask[:, candidate] = occupied
        states = F.normalize(self.output_norm(memory).float(), dim=-1).to(memory.dtype)
        state_mask = state_mask & has_history[:, None]
        return {"states": states * state_mask.unsqueeze(-1), "state_mask": state_mask}

    def encode(self, content, behavior, recency, mask):
        """Export path: Z only. recent history never enters this function."""
        if self.method == "chronicle":
            return self._encode_chronicle(content, behavior, recency, mask)
        if self.method == "hicogen":
            return self._encode_hicogen(content, mask)
        history, safe_mask, has_history = self.contextualize(content, behavior, recency, mask)
        memory = self.interests(history, safe_mask)
        states = F.normalize(self.output_norm(memory).float(), dim=-1).to(memory.dtype)
        state_mask = has_history[:, None].expand(-1, self.num_interests)
        return {"states": states * state_mask.unsqueeze(-1), "state_mask": state_mask}

    def forward(self, content, behavior, recency, mask, recent_content, recent_mask):
        encoded = self.encode(content, behavior, recency, mask)
        alpha, route_logits = self.readout(encoded["states"], encoded["state_mask"],
                                           recent_content, recent_mask)
        return {"states": encoded["states"], "state_mask": encoded["state_mask"],
                "alpha": alpha, "route_logits": route_logits}
