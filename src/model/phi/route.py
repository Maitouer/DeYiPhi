"""Target route attribution (``03_phi.md`` §15; the form is v200's, ``02_phi.md`` §7).

Which shared interest actually drove one observed future interaction? With the vocabulary frozen,
the answer is a token-level posterior

    beta_rg(y) ∝ alpha_bar_rg * exp(c_g^T e(y) / tau_p - A_g)

where ``alpha_bar`` aggregates the teacher's recent readout over the slots that landed on the same
token and ``A_g`` is the token's own pre-computed normalizer. Nothing here scans the catalog, so
each target costs ``O(K d)``.

Two properties are deliberate and are asserted rather than described:

* the route is **not** an item property -- the same item can carry different routes for different
  users, because the posterior depends on the user's own memory and recent context;
* the route is *supervision only*. It is built from ``e(y)`` and the target identity, exactly like
  the ``SID(y)`` labels are, and the target never enters the encoder.
"""

from __future__ import annotations

import torch

from . import artifacts as phi_artifacts

FORMAT = "phi_routes_v1"


def aggregate_alpha(codes: torch.Tensor, alpha: torch.Tensor, vocabulary: int):
    """Aggregate the teacher's slot weights onto the tokens (``03_phi.md`` §15.1).

    Several slots may land on the same token; the token's weight is their sum, so ``alpha_bar``
    still sums to one over the user's unique tokens. Masked slots (``codes < 0``) contribute
    nothing.
    """
    rows, slots = int(codes.shape[0]), int(codes.shape[1])
    if tuple(alpha.shape) != (rows, slots):
        raise ValueError("alpha must be [rows, slots] like the codes")
    device = alpha.device
    aggregated = torch.zeros((rows, int(vocabulary)), dtype=torch.float32, device=device)
    valid = (codes >= 0).to(device)
    aggregated.scatter_add_(1, codes.clamp_min(0).to(device),
                            alpha.float() * valid.to(torch.float32))
    present = aggregated > 0
    # 1e-30 rather than 1e-300: this tensor is float32, where 1e-300 underflows to zero and
    # ``log`` of it would be -inf on the very entries ``present`` says are absent anyway.
    log_alpha = torch.where(present, aggregated.clamp_min(1e-30).log(),
                            torch.full_like(aggregated, float("-inf")))
    return log_alpha, present


def posterior(target_unit: torch.Tensor, center: torch.Tensor, normaliser: torch.Tensor,
              log_alpha: torch.Tensor, present: torch.Tensor, temperature: float,
              target_mask: torch.Tensor) -> torch.Tensor:
    """``beta_rg(y)``: the token posterior for every (row, target, token).

    Uses the token's own pre-computed normalizer ``A_g``, so no catalog is visited: each target
    costs ``O(K d)``.
    """
    # fp32 throughout: the posterior is a softmax over at most K=4 candidates whose logits are
    # bounded by 1/tau_p, so fp64 would buy nothing and cost a CPU route stage several minutes.
    scores = torch.einsum("btd,gd->btg", target_unit.float(), center.float())
    logits = (log_alpha[:, None, :] + scores / float(temperature)
              - normaliser.float()[None, None, :])
    logits = logits.masked_fill(~present[:, None, :], float("-inf"))
    logits = logits.masked_fill(~target_mask[:, :, None], float("-inf"))
    # A row with no target and no available token has nothing to normalise over; softmax over an
    # all-masked axis is NaN, and a NaN that only ever lands on masked entries is still a bug
    # waiting for the next reader. Give those rows a finite (uniform) pivot instead, and zero the
    # entries that carry no target so a consumer cannot read a route where there is none.
    empty = ~torch.isfinite(logits).any(dim=-1, keepdim=True)
    logits = torch.where(empty, torch.zeros_like(logits), logits)
    return torch.softmax(logits, dim=-1) * target_mask[:, :, None].to(torch.float32)


def run(cfg, arm, items, codebook, split: str, *, batch: int = 2048, device=None,
        progress=None) -> dict:
    from . import states as states_module

    tokens = torch.load(str(phi_artifacts.tokens(cfg, split)), map_location="cpu",
                        weights_only=False)
    payload = arm.payload(split)
    states = payload["states"]
    mask = payload["mask"].bool()
    rows = int(mask.shape[0])
    slots = int(mask.shape[1])
    if int(tokens["codes"].shape[0]) != rows:
        raise ValueError("the token artifact does not describe the same rows as the states")
    codes = tokens["codes"].long()
    split_rows = states_module.SplitRows(cfg, arm, split)
    positions = split_rows.positions(payload["source_row_idx"].numpy())
    target_limit = int(cfg.data.target)
    target_rows, target_mask = split_rows.target_rows(positions, target_limit)
    width = int(target_rows.shape[1])
    vocabulary = int(codebook.size)
    temperature = arm.item_temperature()
    normaliser = codebook.log_partition.float()
    center = codebook.center.float()

    route = torch.full((rows, width), -1, dtype=torch.int16)
    confidence = torch.zeros((rows, width), dtype=torch.float16)
    entropy_sum = 0.0
    entropy_count = 0
    for start in range(0, rows, int(batch)):
        stop = min(start + int(batch), rows)
        memory = states[start:stop].float()
        keep = mask[start:stop]
        recent_rows, recent_mask = split_rows.recent_rows(positions[start:stop])
        recent_unit = items.unit(recent_rows.reshape(-1), device=device,
                                 dtype=torch.float32).reshape(
            stop - start, recent_rows.shape[1], items.dim)
        alpha = arm.alpha(memory, keep, recent_unit, torch.from_numpy(recent_mask),
                          device=device, batch=stop - start)
        block_codes = codes[start:stop]
        log_alpha, present = aggregate_alpha(block_codes, alpha, vocabulary)
        keep_target = torch.from_numpy(target_mask[start:stop]).to(alpha.device)
        flat = torch.from_numpy(target_rows[start:stop].reshape(-1))
        gathered = items.unit(flat.clamp_min(0).numpy(), device=device, dtype=torch.float32)
        gathered = gathered.reshape(stop - start, width, items.dim)
        probability = posterior(gathered.to(alpha.device), center.to(alpha.device),
                                normaliser, log_alpha, present, temperature, keep_target)
        best = probability.argmax(dim=-1)
        # A row with no valid interest state has no candidate token at all; ``posterior`` still
        # returns a finite pivot so nothing is NaN, and this is where "no route" is decided.
        live = present.any(dim=-1)
        assign = keep_target.cpu() & live.cpu()[:, None]
        route[start:stop] = torch.where(assign, best.to(torch.int16).cpu(),
                                        torch.full_like(best, -1, dtype=torch.int16).cpu())
        top = probability.max(dim=-1).values
        confidence[start:stop] = torch.where(assign, top.to(torch.float16).cpu(),
                                            torch.zeros_like(top, dtype=torch.float16).cpu())
        # ``xlogy`` gives exactly zero for ``p = 0``; ``p * log(p)`` does not, and the NaN it
        # produces would poison the whole split's entropy average.
        entry = -torch.special.xlogy(probability, probability).sum(-1)
        entry = entry.cpu()
        entropy_sum += float(entry[assign].sum())
        entropy_count += int(assign.sum())
        if progress is not None:
            progress.tick(stop, rows, "%s routes %d/%d" % (split, stop, rows))
    return {"format": FORMAT, "split": split, "rows": rows, "targets": width,
            "vocabulary": vocabulary,
            "source_row_idx": payload["source_row_idx"].clone(),
            "target_rows": torch.from_numpy(target_rows.astype("int64")),
            "target_mask": torch.from_numpy(target_mask),
            "route": route, "confidence": confidence,
            "route_entropy": (entropy_sum / entropy_count) if entropy_count else 0.0,
            "targets_with_route": int((route >= 0).sum())}
