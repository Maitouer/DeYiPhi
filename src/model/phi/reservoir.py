"""The user-balanced codebook reservoir (``03_phi.md`` §5).

The shared vocabulary is meant to hold *population-level reusable predictive concepts*, not to
reproduce every training state, so phi estimates its predictive geometry on a fixed-budget sample
of the fit states. Two properties of that sample are what keep the vocabulary from drifting:

* the weight ``w_rk = a_r * m_rk / sum_j m_rj`` gives every user one unit of total mass, so a user
  with a long history cannot outvote a user with a short one;
* ``m_rk`` is the *existence* mask, never the recent-conditioned ``alpha``. Whether a long-term
  interest exists is a property of the user; whether it is active right now is context, and that
  belongs to route attribution (``02_phi.md`` §5.1).
"""

from __future__ import annotations

import torch


def code_weights(uid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``w_rk^code`` for every (row, slot); masked slots carry exactly zero."""
    count = mask.sum(dim=1, keepdim=True).clamp_min(1)
    users, inverse = torch.unique(torch.as_tensor(uid, dtype=torch.long), return_inverse=True)
    rows_per_user = torch.bincount(inverse, minlength=users.numel()).clamp_min(1)
    balance = (1.0 / rows_per_user.double())[inverse]
    return (balance[:, None] * mask.to(torch.float64) / count.to(torch.float64))


def sample(weight: torch.Tensor, size: int, generator: torch.Generator):
    """Weighted reservoir sampling without replacement (Efraimidis-Spirakis keys).

    ``key = log(u) / w`` is the log form of ``u^(1/w)``; keeping the ``size`` largest keys draws
    exactly the sample the weights describe, with a seeded generator the draw is reproducible.
    """
    flat = weight.reshape(-1).double()
    draws = torch.rand(flat.numel(), generator=generator, dtype=torch.float64)
    key = torch.where(flat > 0, draws.clamp_min(1e-300).log() / flat.clamp_min(1e-300),
                      torch.full_like(flat, float("-inf")))
    take = min(int(size), int((flat > 0).sum()))
    if take < 1:
        raise ValueError("the fit split produced no valid interest state")
    chosen = key.topk(take).indices
    order = torch.argsort(chosen)                       # keep the reservoir in row order
    return chosen[order]


def build(cfg, arm, *, size: int, seed: int, progress=None) -> dict:
    """The reservoir artifact: the drawn states plus everything the next stages key on."""
    from . import states as states_module

    payload = arm.payload("train")
    states = payload["states"]
    mask = payload["mask"].bool()
    rows, slots = int(mask.shape[0]), int(mask.shape[1])
    if slots != arm.num_interests:
        raise ValueError("the frozen states carry %d slots but the config names K=%d"
                         % (slots, arm.num_interests))
    split_rows = states_module.SplitRows(cfg, arm, "train")
    positions = split_rows.positions(payload["source_row_idx"].numpy())
    uid = torch.from_numpy(split_rows.uids(positions).astype("int64"))
    weight = code_weights(uid, mask)
    generator = torch.Generator().manual_seed(int(seed))
    chosen = sample(weight, int(size), generator)
    index = torch.stack([chosen // slots, chosen % slots], dim=1)
    gathered = states[index[:, 0], index[:, 1]]
    if progress is not None:
        progress.line("reservoir", "drew %d states from %d rows (%d users)"
                      % (int(chosen.numel()), rows, int(uid.unique().numel())), force=True)
    return {"format": "phi_reservoir_v1", "states": gathered.clone(),
            "row": index[:, 0].clone(), "slot": index[:, 1].clone(),
            "weight": weight.reshape(-1)[chosen].clone(), "uid": uid[index[:, 0]].clone(),
            "count": mask.sum(dim=1).clone(), "rows": rows, "slots": slots}

