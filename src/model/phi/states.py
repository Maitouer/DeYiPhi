"""The frozen DeYi side: the arm's states, its recent-conditioned readout, and the split rows.

Two invariants are enforced here rather than assumed, because a phi run that silently reads
the wrong rows produces a vocabulary that looks fine and is not:

* the states artifact must belong to the arm this config names (dataset, task, K, recent);
* the states' ``source_row_idx`` must be the canonical rows of the split being used, so a
  permutation cannot be absorbed silently.

``alpha`` re-runs the frozen teacher's own readout (``01_deyi_encoder.md`` §9) from the
checkpoint instead of re-deriving it: route attribution is defined as the teacher's posterior
responsibility, so it has to use the teacher's weights and its ``tau_g``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class TeacherArm:
    """Where one frozen DeYi arm lives, and how to read its predictive quantities."""

    def __init__(self, cfg):
        from src.model.deyi import artifacts as deyi_artifacts

        self.cfg = cfg
        self.dataset = deyi_artifacts.dataset(cfg)
        self.task = str(cfg.task)
        self.num_interests = int(cfg.deyi.num_interests)
        self.recent = int(cfg.deyi.recent)
        self.root = deyi_artifacts.output_root(cfg)
        self.dim = int(cfg.deyi.state_dim)
        self._checkpoint = None

    # ------------------------------------------------------------------ artifacts
    def states_path(self, split: str) -> Path:
        from src.model.deyi import artifacts as deyi_artifacts

        return deyi_artifacts.states(self.cfg, split)

    def checkpoint_path(self) -> Path:
        from src.model.deyi import artifacts as deyi_artifacts

        return deyi_artifacts.checkpoint(self.cfg)

    def payload(self, split: str) -> dict:
        """The DeYi states of one split, memory-mapped: the fp16 payload must not sit in RAM."""
        path = self.states_path(split)
        if not path.is_file():
            raise FileNotFoundError(
                "the frozen DeYi %s states are missing: %s (run the deyi encode stage for this "
                "arm first)" % (split, path))
        return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)

    def rows(self, split: str) -> int:
        return int(self.payload(split)["states"].shape[0])

    def checkpoint(self) -> dict:
        if self._checkpoint is None:
            path = self.checkpoint_path()
            if not path.is_file():
                raise FileNotFoundError("the frozen DeYi checkpoint is missing: %s" % path)
            self._checkpoint = torch.load(str(path), map_location="cpu", mmap=True,
                                          weights_only=False)
        return self._checkpoint

    # ------------------------------------------------------------------ predictive constants
    def item_temperature(self) -> float:
        """``tau_p``: the teacher's own prediction temperature, never a phi choice."""
        value = self.checkpoint().get("item_temperature")
        if value is None or float(value) <= 0:
            raise ValueError("the checkpoint carries no usable item_temperature")
        return float(value)

    def route_temperature(self) -> float:
        value = self.checkpoint().get("route_temperature")
        return float(value) if value else 0.5

    def readout_weights(self):
        state = self.checkpoint().get("model_state") or {}
        recent = state.get("read_recent.weight")
        memory = state.get("read_state.weight")
        if recent is None or memory is None:
            raise ValueError(
                "the frozen checkpoint carries no recent readout (read_recent/read_state); "
                "route attribution needs the teacher's own weights")
        return (torch.as_tensor(recent, dtype=torch.float32).clone(),
                torch.as_tensor(memory, dtype=torch.float32).clone())

    def identity(self) -> dict:
        from src.model.phi import artifacts as phi_artifacts

        return {"dataset": self.dataset, "task": self.task,
                "num_interests": self.num_interests, "recent": self.recent,
                "checkpoint": phi_artifacts.file_marker(self.checkpoint_path()),
                "states": {split: phi_artifacts.file_marker(self.states_path(split))
                           for split in ("train", "test")
                           if self.states_path(split).is_file()}}

    # ------------------------------------------------------------------ readout
    @torch.no_grad()
    def alpha(self, states: torch.Tensor, state_mask: torch.Tensor, recent_unit: torch.Tensor,
              recent_mask: torch.Tensor, *, device=None, batch: int = 4096):
        """The teacher's recent-conditioned interest weights ``alpha_rk``.

        Mirrors ``DeYi.readout`` exactly: a low-capacity semantic-mean readout (``01_deyi_encoder.md``
        §9), including its published fallback -- old history non-empty but recent empty means the
        readout has no opinion, so the mixture is uniform over the slots.
        """
        target = torch.device("cpu") if device is None else device
        weights_recent, weights_state = self.readout_weights()
        weights_recent = weights_recent.to(target)
        weights_state = weights_state.to(target)
        tau_g = self.route_temperature()
        rows = int(states.shape[0])
        alpha = torch.zeros((rows, self.num_interests), dtype=torch.float32)
        for start in range(0, rows, int(batch)):
            stop = min(start + int(batch), rows)
            memory = states[start:stop].to(target, torch.float32)
            mask = state_mask[start:stop].to(target).bool()
            content = recent_unit[start:stop].to(target, torch.float32)
            keep = recent_mask[start:stop].to(target).bool()
            weight = keep.unsqueeze(-1).to(torch.float32)
            count = weight.sum(1).clamp_min(1.0)
            mean = (content * weight).sum(1) / count
            recent_key = F.normalize((weights_recent @ mean.T).T, dim=-1)
            state_key = F.normalize(memory @ weights_state.T, dim=-1)
            logits = (recent_key[:, None, :] * state_key).sum(-1) / tau_g
            value = torch.softmax(logits, dim=-1)
            has_recent = keep.any(-1)
            uniform = torch.full_like(value, 1.0 / self.num_interests)
            value = torch.where(has_recent[:, None], value, uniform)
            alpha[start:stop] = (value * mask.to(torch.float32)).cpu()
        return alpha


class SplitRows:
    """One split's canonical rows, joined to the frozen states through ``source_row_idx``."""

    def __init__(self, cfg, arm: TeacherArm, split: str):
        from src.data.dataset import open_dataset

        self.split_name = split
        dataset = open_dataset(str(cfg.data.root), str(cfg.task), arm.dataset)
        self.split = dataset.split(str(cfg.task), split)
        self.sources = np.asarray(self.split.source_row_idx, dtype=np.int64)
        self.uid = np.asarray(self.split.uid, dtype=np.int64)
        self.target = np.asarray(self.split.target, dtype=np.int64)
        self.target_length = np.asarray(self.split.target_length, dtype=np.int64)
        self.rows = int(self.sources.size)
        self.recent = int(arm.recent)

    def positions(self, state_source_rows) -> np.ndarray:
        """Split positions of the given canonical rows, asserted rather than assumed."""
        wanted = np.asarray(state_source_rows, dtype=np.int64)
        order = np.argsort(self.sources, kind="stable")
        sorted_rows = self.sources[order]
        found = np.searchsorted(sorted_rows, wanted)
        if (found >= len(order)).any() or not np.array_equal(sorted_rows[found], wanted):
            raise ValueError(
                "the frozen DeYi states do not map onto the canonical %s split" % self.split_name)
        return order[found]

    def recent_rows(self, positions, budget: int | None = None):
        """The recent window of each row: ``(physical item rows, mask)``; 0 is padding."""
        budget = self.recent if budget is None else int(budget)
        return self.split.history_rows(np.asarray(positions, dtype=np.int64),
                                       mode="recent", recent=budget)

    def target_rows(self, positions, limit: int):
        """The future window of each row: ``(physical item rows, mask)``; 0 is padding."""
        positions = np.asarray(positions, dtype=np.int64)
        width = min(int(limit), self.target.shape[1])
        values = self.target[positions][:, :width].astype(np.int64)
        keep = np.arange(width)[None, :] < self.target_length[positions][:, None]
        return values, keep

    def uids(self, positions) -> np.ndarray:
        return self.uid[np.asarray(positions, dtype=np.int64)]
