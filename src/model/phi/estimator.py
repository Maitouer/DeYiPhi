"""Head-tail predictive statistics (``03_phi.md`` §6-§8).

The two quantities the whole vocabulary is built from are

    A(x)  = log sum_{i in C} exp(x^T e(i) / tau_p)          the predictive log-partition
    mu_x  = E_{i ~ p_x}[e(i)] = M_x / Z_x                   the item-semantic expectation

and neither may be obtained by scanning a multi-million-row catalog for every state. Phi v300
therefore splits the sum into a **deterministic high-probability head** (the exact top-M items,
which carry most of the mass at ``tau_p = 0.15``) and an **unbiased importance-sampled tail**
over a shared proposal ``q``:

    Z_hat = Z_head + (1/N) sum_n 1[j_n not in H_x] a_x(j_n) / q(j_n)

Every reported number is a property of that estimator, so the calibration set (exact full-catalog
scan on a few hundred states) is part of the stage rather than an optional extra: the documents
require the approximation to be *measurable*, not to be trusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class Estimate:
    """Per-state predictive statistics plus the two stability reads the documents ask for."""

    log_partition: torch.Tensor            # (S,) float32, A_hat
    moment: torch.Tensor                   # (S, d) float32, mu_hat
    head_mass: torch.Tensor                # (S,) float32, Z_head / Z_hat
    tail_ess: torch.Tensor                 # (S,) float32, ESS of the tail weights
    head: int
    tail: int
    fallback: list = field(default_factory=list)
    head_mass_floor: float = 0.0
    tail_ess_floor: float = 0.0

    def status(self):
        head = self.head_mass
        ess = self.tail_ess
        return {
            "states": int(head.numel()),
            "head": int(self.head), "tail": int(self.tail),
            "head_mass": {"min": float(head.min()), "median": float(head.median()),
                          "mean": float(head.mean())},
            "tail_ess": {"min": float(ess.min()), "median": float(ess.median()),
                         "mean": float(ess.mean())},
            "fallback": list(self.fallback),
        }


def _running_top(values: torch.Tensor, width: int, best_v, best_i, base: int):
    """Merge one catalog block's best ``width`` scores into the running top-``width``."""
    take = min(int(width), int(values.shape[1]))
    block_v, block_i = values.topk(take, dim=-1)
    block_i = block_i + int(base)
    if best_v is None:
        return block_v, block_i
    merged_v = torch.cat([best_v, block_v], dim=1)
    merged_i = torch.cat([best_i, block_i], dim=1)
    top_v, order = merged_v.topk(min(int(width), merged_v.shape[1]), dim=-1)
    return top_v, merged_i.gather(1, order)


def head_search(queries: torch.Tensor, units, *, head: int, item_block: int, query_block: int,
                device, dtype, progress=None):
    """Exact top-``head`` catalog items per query, streamed over the catalog.

    The merge keeps the exact global top-M (a block's top-M merged into the running top-M is the
    global top-M), so this *is* the deterministic head the documents describe -- there is no ANN
    recall to lose. An approximate index would replace the per-block top-M with its own candidate
    list; that is the only place ``M_ann`` would enter.
    """
    total = int(queries.shape[0])
    index = torch.empty((total, int(head)), dtype=torch.long)
    score = torch.empty((total, int(head)), dtype=torch.float32)
    for start in range(0, total, int(query_block)):
        stop = min(start + int(query_block), total)
        wide = torch.as_tensor(queries[start:stop], dtype=torch.float32, device=device)
        block_query = wide.to(dtype)
        best_v = best_i = None
        for item_start in range(0, units.catalog.size, int(item_block)):
            item_stop = min(item_start + int(item_block), units.catalog.size)
            rows = units.block(item_start, item_stop)
            values = (block_query @ rows.T.to(block_query.dtype)).float()
            best_v, best_i = _running_top(values, int(head), best_v, best_i, item_start)
        index[start:stop] = best_i.cpu()
        score[start:stop] = best_v.cpu()
        if progress is not None:
            progress.tick(stop, total, "head %d/%d" % (stop, total))
    return index, score


def _tail_membership(head_index: torch.Tensor, tail_index: torch.Tensor) -> torch.Tensor:
    """``1[j not in H_x]`` per (query, tail draw), resolved exactly through the sorted head."""
    sorted_head, _ = head_index.sort(dim=-1)
    position = torch.searchsorted(sorted_head.contiguous(), tail_index.contiguous(), right=False)
    position = position.clamp_max(sorted_head.shape[1] - 1)
    return sorted_head.gather(1, position) != tail_index


def estimate(queries: torch.Tensor, units, catalog, temperature: float, *, head: int, tail: int,
             item_block: int, query_block: int, device, dtype, generator, progress=None,
             moment_block: int = 1024):
    """``(A_hat, mu_hat)`` for every query, with the head/tail stability diagnostics."""
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    total = int(queries.shape[0])
    dim = int(units.items.dim)
    log_partition = torch.empty(total, dtype=torch.float32)
    moment = torch.empty((total, dim), dtype=torch.float32)
    head_mass = torch.empty(total, dtype=torch.float32)
    tail_ess = torch.empty(total, dtype=torch.float32)
    index, score = head_search(queries, units, head=int(head), item_block=int(item_block),
                               query_block=int(query_block), device=device, dtype=dtype,
                               progress=progress)
    for start in range(0, total, int(query_block)):
        stop = min(start + int(query_block), total)
        query = torch.as_tensor(queries[start:stop], dtype=torch.float32, device=device)
        head_rows = units.gather(index[start:stop], device=device).float()
        # Recompute the head logits from the gathered rows in fp32: the search accumulated them
        # in the compute dtype, and a log-partition must not inherit that rounding.
        # ``query`` is 2-D and the head rows are 3-D, so this is a per-query contraction rather
        # than a plain matmul: ``query @ head_rows.transpose(1, 2)`` would broadcast the query
        # into a (queries, queries, head) tensor instead of scoring each query's own head.
        head_score = torch.einsum("qd,qmd->qm", query, head_rows) / float(temperature)
        weight = head_score.exp().double()
        head_z = weight.sum(dim=1)
        # The moment is accumulated in bounded sub-blocks of queries: ``head_rows.double()`` for a
        # whole query block was 34 GiB at M=512 and killed the job. ``mu`` is a mean of unit
        # vectors, so fp32 rows are exact enough; only the partition keeps fp64 weights.
        head_m = torch.empty((stop - start, dim), dtype=torch.float32, device=head_rows.device)
        for sub in range(0, stop - start, int(moment_block)):
            sub_stop = min(sub + int(moment_block), stop - start)
            head_m[sub:sub_stop] = torch.einsum(
                "qm,qmd->qd", weight[sub:sub_stop].float(),
                head_rows[sub:sub_stop].to(torch.float32))

        drawn = catalog.draw(int(tail), generator).to(device)
        tail_rows = units.gather(drawn, device=device).float()
        tail_score = (query @ tail_rows.T) / float(temperature)
        log_weight = tail_score.double() - catalog.log_proposal.to(device)[drawn]
        keep = _tail_membership(index[start:stop].to(device), drawn[None, :].expand(stop - start, -1))
        log_weight = log_weight.masked_fill(~keep, float("-inf"))
        tail_weight = log_weight.exp()
        tail_z = tail_weight.sum(dim=1) / float(tail)
        tail_m = torch.einsum("qn,nd->qd", tail_weight,
                              tail_rows.double()) / float(tail)

        total_z = head_z + tail_z
        if bool((total_z <= 0).any()):
            raise FloatingPointError("a state received no predictive mass from head or tail")
        log_partition[start:stop] = total_z.log().float().cpu()
        moment[start:stop] = ((head_m.double() + tail_m) / total_z[:, None]).float().cpu()
        head_mass[start:stop] = (head_z / total_z).float().cpu()
        tail_ess[start:stop] = _ess(tail_weight).float().cpu()
        if progress is not None:
            progress.tick(stop, total, "statistics %d/%d" % (stop, total))
    return Estimate(log_partition=log_partition, moment=moment, head_mass=head_mass,
                    tail_ess=tail_ess, head=int(head), tail=int(tail))


def _ess(weight: torch.Tensor) -> torch.Tensor:
    summed = weight.sum(dim=1)
    squared = weight.square().sum(dim=1)
    return summed.square() / squared.clamp_min(1e-300)


def refine(queries: torch.Tensor, units, catalog, temperature: float, estimate_in: Estimate,
           subset: torch.Tensor, *, rounds: int, factor: float, item_block: int, query_block: int,
           device, dtype, generator, progress=None):
    """Re-estimate the states whose head mass or tail ESS failed, with a wider estimator.

    Most states run at the cheap configured width; only the difficult ones pay more, which is the
    whole point of reporting the two diagnostics instead of trusting a fixed ``M``/``N``.
    """
    keep = estimate_in
    pending = torch.as_tensor(subset, dtype=torch.long)
    for round_index in range(int(rounds)):
        if int(pending.numel()) == 0:
            break
        head = int(keep.head * float(factor) ** (round_index + 1))
        tail = int(keep.tail * float(factor) ** (round_index + 1))
        if progress is not None:
            progress.line("estimate", "fallback round %d: states=%d head=%d tail=%d"
                          % (round_index + 1, int(pending.numel()), head, tail), force=True)
        updated = estimate(queries[pending], units, catalog, temperature, head=head, tail=tail,
                           item_block=item_block, query_block=query_block, device=device,
                           dtype=dtype, generator=generator, progress=progress)
        keep.log_partition[pending] = updated.log_partition
        keep.moment[pending] = updated.moment
        keep.head_mass[pending] = updated.head_mass
        keep.tail_ess[pending] = updated.tail_ess
        keep.fallback.append({"round": round_index + 1, "states": int(pending.numel()),
                              "head": head, "tail": tail})
        failing = ((updated.head_mass < float(keep.head_mass_floor))
                   | (updated.tail_ess < float(keep.tail_ess_floor)))
        pending = pending[failing]
    return keep


def exact(queries: torch.Tensor, units, catalog, temperature: float, *, item_block: int,
          query_block: int, device, dtype, progress=None):
    """Exact ``(A, mu)`` by streaming the whole catalog; the calibration reference."""
    total = int(queries.shape[0])
    dim = int(units.items.dim)
    log_partition = torch.empty(total, dtype=torch.float32)
    moment = torch.empty((total, dim), dtype=torch.float32)
    for start in range(0, total, int(query_block)):
        stop = min(start + int(query_block), total)
        query = torch.as_tensor(queries[start:stop], dtype=torch.float32, device=device)
        mass = torch.zeros(stop - start, dtype=torch.float64, device=device)
        first = torch.zeros((stop - start, dim), dtype=torch.float64, device=device)
        for item_start in range(0, units.catalog.size, int(item_block)):
            item_stop = min(item_start + int(item_block), units.catalog.size)
            rows = units.block(item_start, item_stop).float()
            weight = ((query @ rows.T) / float(temperature)).exp()
            mass += weight.double().sum(dim=1)
            first += (weight @ rows).double()
        log_partition[start:stop] = mass.log().float().cpu()
        moment[start:stop] = (first / mass[:, None]).float().cpu()
        if progress is not None:
            progress.tick(stop, total, "exact calibration %d/%d" % (stop, total))
    return log_partition, moment


def calibration_report(log_estimate: torch.Tensor, moment_estimate: torch.Tensor,
                       log_exact: torch.Tensor, moment_exact: torch.Tensor,
                       moment_population: torch.Tensor | None = None) -> dict:
    """The four reads ``03_phi.md`` §8 asks for, plus the optional first-moment cosine."""
    delta = (log_estimate.double() - log_exact.double()).abs()
    estimate64, exact64 = moment_estimate.double(), moment_exact.double()
    cosine = torch.nn.functional.cosine_similarity(estimate64, exact64, dim=-1)
    relative = (estimate64 - exact64).norm(dim=-1) / exact64.norm(dim=-1).clamp_min(1e-30)
    report = {
        "states": int(log_estimate.numel()),
        "partition_error": {"mean": float(delta.mean()), "max": float(delta.max()),
                            "p95": float(delta.quantile(0.95))},
        "moment_cosine": {"mean": float(cosine.mean()), "min": float(cosine.min())},
        "moment_relative_error": {"mean": float(relative.mean()), "max": float(relative.max())},
    }
    if moment_population is not None:
        population = moment_population.double()
        report["moment_population_cosine"] = {
            "mean": float(torch.nn.functional.cosine_similarity(
                exact64, population, dim=-1).mean())}
    return report


def assignment_agreement(distance_estimate: torch.Tensor, distance_exact: torch.Tensor) -> dict:
    """Top-1 and top-2 agreement of the predictive assignment under both geometries."""
    top1 = (distance_estimate.argmin(dim=1) == distance_exact.argmin(dim=1)).float().mean()
    best_estimate = distance_estimate.argmin(dim=1)
    top2 = distance_exact.topk(2, dim=1, largest=False).indices
    top2 = (top2 == best_estimate[:, None]).any(dim=1).float().mean()
    return {"top1": float(top1), "top2": float(top2)}


def information(retained_bits: float) -> float:
    """Bits helper: the documents report predictive KL in nats, the audit in bits."""
    return float(retained_bits) / math.log(2.0)
