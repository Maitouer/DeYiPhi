"""Predictive-KL medoid codebook (``03_phi.md`` §9-§10, §13).

For a state ``s`` and a codeword ``c`` the predictive divergence has an exact low-dimensional
form (``02_phi.md`` §4.5)

    D_KL(p_s || p_c) = C_s + A(c) - (1/tau_p) c^T mu_s

with ``C_s`` depending on the state only. Assignment is therefore ``argmin_g [A(c_g) -
c_g^T mu_s / tau_p]``, a single matrix product over cached sufficient statistics, and the same
expression -- restricted to the cluster's own members -- is the medoid update.

The medoid constraint ``c_g in {z_s}`` is what keeps the update catalog-free: every codeword is
itself a legal DeYi predictive distribution, so its ``A`` is already cached and no free prototype
has to be optimised against the catalog. Only the *L* final codewords get an exact normalizer
(§13), which is ``O(L|C|d)`` instead of ``O(|S||C|d)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Codebook:
    medoid: torch.Tensor            # (L,) reservoir positions, c_g = states[medoid[g]]
    center: torch.Tensor            # (L, d) float32: the codeword itself, c_g = z_{m_g}
    log_partition: torch.Tensor     # (L,) float32, A of the medoid (estimated, then exact)
    label: torch.Tensor             # (S,) int64 reservoir assignment
    moment: torch.Tensor = None     # (L, d) optional exact E_{i ~ p_{c_g}}[e(i)] (diagnostics)
    history: list = field(default_factory=list)
    exact: bool = False
    converged: bool = False
    metric: str = "predictive"     # which distortion this partition was fitted under

    @property
    def size(self) -> int:
        return int(self.center.shape[0])


def distance(moment: torch.Tensor, center: torch.Tensor, log_partition: torch.Tensor,
             temperature: float) -> torch.Tensor:
    """``A(c_g) - c_g^T mu_s / tau_p`` for every (state, codeword) pair."""
    return (log_partition[None, :].double()
            - (moment.double() @ center.double().T) / float(temperature))


def scores_for(states: torch.Tensor, moment: torch.Tensor, center: torch.Tensor,
               log_partition: torch.Tensor, temperature: float,
               metric: str = "predictive") -> torch.Tensor:
    """The ``(S, L)`` distortion table in the geometry this run was configured with.

    ``05_analysis_design.md`` §12 asks for one clustering framework with three distortion
    choices, so this is the *only* place the metric enters: everything downstream (medoid update,
    router distillation, tokenization, audit) reads the resulting partition.

    ``predictive`` is the proposed criterion, ``euclidean``/``cosine`` are the generic
    vector-space quantizers. Note that on unit-norm predictive states ``||z-c||^2 = 2 - 2 z^T c``,
    so the two geometric criteria induce the same partition by construction; that equivalence is
    itself a result worth reporting rather than a bug.
    """
    if metric == "predictive":
        return distance(moment, center, log_partition, temperature)
    z = states.double()
    c = center.double()
    if metric == "euclidean":
        return ((z * z).sum(1, keepdim=True) - 2.0 * (z @ c.T) + (c * c).sum(1)[None, :])
    if metric == "cosine":
        z = torch.nn.functional.normalize(z, dim=-1)
        c = torch.nn.functional.normalize(c, dim=-1)
        return 1.0 - z @ c.T
    raise ValueError("unknown codebook metric %r" % metric)


def _weighted_mean(moment: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, size: int):
    total = torch.zeros((size, moment.shape[1]), dtype=torch.float64)
    mass = torch.zeros(size, dtype=torch.float64)
    total.index_add_(0, labels, moment.double() * weight.double()[:, None])
    mass.index_add_(0, labels, weight.double())
    return total / mass.clamp_min(1e-30)[:, None], mass


def fit(states: torch.Tensor, moment: torch.Tensor, log_partition: torch.Tensor,
        weight: torch.Tensor, size: int, temperature: float, *, iterations: int = 25,
        seed: int = 2026, restart: bool = True, progress=None,
        metric: str = "predictive") -> Codebook:
    """Alternate predictive-KL assignment with the weighted-medoid update until it is stable.

    ``center`` is the codeword *state* ``z_t`` and not its moment: the assignment's cross term is
    ``c_g^T mu_s``, and ``mu_t^T mu_s`` would be a different (and wrong) divergence.
    """
    count = int(moment.shape[0])
    size = int(size)
    if size < 1 or size > count:
        raise ValueError("codebook size must be within [1, number of reservoir states]")
    generator = torch.Generator().manual_seed(int(seed))
    medoid = _initialise(states, moment, log_partition, weight, size, temperature, generator, metric)
    center = states[medoid].float()
    normaliser = log_partition[medoid].float()
    label = torch.zeros(count, dtype=torch.long)
    history = []
    best = None
    best_objective, stale = float("inf"), 0
    for step in range(int(iterations)):
        scores = scores_for(states, moment, center, normaliser, temperature, metric)
        label = scores.argmin(dim=1)
        mean, mass = _weighted_mean(moment, weight, label, size)
        # Geometric criteria derive the medoid from the cluster's mean *state*, the predictive one
        # from its mean moment; both are weighted means of reservoir quantities.
        mean_state = None if metric == "predictive" else _weighted_mean(states, weight, label, size)[0]
        objective = float((scores.gather(1, label[:, None]).squeeze(1).double()
                           * weight.double()).sum())
        moved = restarts = 0
        taken = torch.zeros(count, dtype=torch.bool)
        taken[medoid] = True
        for index in range(size):
            members = torch.nonzero(label == index, as_tuple=False).squeeze(1)
            if int(members.numel()) == 0:
                if restart:
                    medoid[index] = _worst_state(scores, label, weight, taken)
                    taken[medoid[index]] = True
                    center[index] = states[medoid[index]].float()
                    normaliser[index] = log_partition[medoid[index]].float()
                    moved += 1
                    restarts += 1
                continue
            # The medoid is the member closest to the cluster's own mean moment, measured in
            # exactly the objective the assignment minimises: the candidate *state* ``z_t``
            # crosses the cluster's mean moment, so ``z_t^T mu_bar_g`` and not ``mu_t^T mu_bar_g``.
            if metric == "predictive":
                value = (log_partition[members].double()
                         - (states[members].double() @ mean[index]) / float(temperature))
            elif metric == "euclidean":
                value = (states[members].double().pow(2).sum(1)
                         - 2.0 * (states[members].double() @ mean_state[index]))
            else:   # cosine: the member with the largest mean similarity to its cluster
                value = -(states[members].double()
                          @ torch.nn.functional.normalize(mean_state[index], dim=-1))
            choice = members[int(value.argmin())]
            if int(choice) != int(medoid[index]):
                moved += 1
            medoid[index] = choice
            center[index] = states[choice].float()
            normaliser[index] = log_partition[choice].float()
        # Score the configuration the update produced and keep the best one visited: an empty
        # cluster restart is a heuristic that can move the objective either way, so the returned
        # codebook has to be the best configuration rather than the last one.
        updated_scores = scores_for(states, moment, center, normaliser, temperature, metric)
        updated_label = updated_scores.argmin(dim=1)
        updated = float((updated_scores.gather(1, updated_label[:, None]).squeeze(1).double()
                         * weight.double()).sum())
        if best is None or updated < best["objective"]:
            best = {"objective": updated, "step": step, "converged": moved == 0,
                    "medoid": medoid.clone(), "center": center.clone(),
                    "normaliser": normaliser.clone(), "label": updated_label.clone()}
        history.append({"step": step, "objective": objective, "updated": updated,
                        "moved": moved, "restarts": restarts,
                        "empty": int((mass <= 0).sum()), "labels": int(torch.unique(label).numel())})
        if progress is not None:
            progress.line("codebook", "step=%d objective=%.6g moved=%d clusters=%d"
                          % (step, objective, moved, int(torch.unique(label).numel())), force=True)
        # Convergence is the objective, not the number of moved medoids: a restart that keeps
        # re-seeding empty clusters never reaches ``moved == 0`` while changing nothing, which is
        # exactly how an earlier production run burned 25 iterations at a constant objective.
        if updated < best_objective - 1e-9:
            best_objective, stale = updated, 0
        else:
            stale += 1
        if (moved == 0 and step > 0) or stale >= 3:
            break
    return Codebook(medoid=best["medoid"], center=best["center"],
                    log_partition=best["normaliser"], label=best["label"], history=history,
                    converged=bool(best["converged"]), metric=str(metric))


def _initialise(states: torch.Tensor, moment: torch.Tensor, log_partition: torch.Tensor,
                weight: torch.Tensor, size: int, temperature: float, generator,
                metric: str = "predictive") -> torch.Tensor:
    """Weighted k-means++ in the configured geometry, with the first centre drawn by weight."""
    count = int(moment.shape[0])
    probability = weight.double().clamp_min(0.0)
    probability = (probability / probability.sum()) if float(probability.sum()) > 0 else (
        torch.full((count,), 1.0 / count, dtype=torch.float64))
    available = torch.ones(count, dtype=torch.bool)
    first = int(torch.multinomial(probability, 1, generator=generator))
    chosen = [first]
    available[first] = False
    best = scores_for(states, moment, states[chosen].float(), log_partition[chosen].float(),
                      temperature, metric)
    best = best.min(dim=1).values.clamp_min(0.0)
    while len(chosen) < int(size):
        score = probability * best
        score = torch.where(available, score, torch.zeros_like(score))
        if float(score.sum()) <= 0:
            # The predictive distortion is exhausted (many states share one geometry): fall back
            # to drawing an *unused* state by weight rather than re-picking a chosen one, which is
            # what produced dozens of empty clusters at L=256.
            score = torch.where(available, probability, torch.zeros_like(probability))
        if float(score.sum()) <= 0:
            break
        pick = int(torch.multinomial(score / score.sum(), 1, generator=generator))
        chosen.append(pick)
        available[pick] = False
        candidate = scores_for(states, moment, states[[pick]].float(),
                               log_partition[[pick]].float(), temperature,
                               metric).squeeze(1).clamp_min(0.0)
        best = torch.minimum(best, candidate)
    while len(chosen) < int(size):
        chosen.append(chosen[len(chosen) % max(len(chosen), 1)])
    return torch.tensor(chosen, dtype=torch.long)


def _worst_state(scores: torch.Tensor, label: torch.Tensor, weight: torch.Tensor,
                 taken: torch.Tensor) -> int:
    """The state a dead cluster restarts from: the largest weighted distortion not already a centre.

    Excluding the current medoids is what stops the churn: re-seeding from a state that already
    owns a cluster simply empties the restarted one again on the next assignment.
    """
    distortion = scores.gather(1, label[:, None]).squeeze(1).double().clamp_min(0.0)
    value = distortion * weight.double()
    value = torch.where(taken, torch.full_like(value, -1.0), value)
    return int(value.argmax())


def state_kl(states: torch.Tensor, moment: torch.Tensor, log_partition: torch.Tensor,
             codebook: Codebook, temperature: float) -> torch.Tensor:
    """``D_KL(p_s || p_{c_g})`` for the assigned codeword, using the exact decomposition."""
    self_moment = (states.double() * moment.double()).sum(-1)
    own = self_moment / float(temperature) - log_partition.double()
    assigned = codebook.log_partition.double()[codebook.label]
    cross = (moment.double() * codebook.center.double()[codebook.label]).sum(-1)
    return (own + assigned - cross / float(temperature)).clamp_min(0.0).float()


def diagnostics(codebook: Codebook, states: torch.Tensor, moment: torch.Tensor,
                log_partition: torch.Tensor, weight: torch.Tensor, uid: torch.Tensor,
                temperature: float, *, split_users=None) -> dict:
    """What ``03_phi.md`` §23.2 asks a codebook to report.

    Usage statistics are diagnostics, never objectives: the documents explicitly refuse a
    usage-balance or entropy term, so a long tail is reported rather than suppressed.
    """
    kl = state_kl(states, moment, log_partition, codebook, temperature)
    mass = torch.zeros(codebook.size, dtype=torch.float64)
    mass.index_add_(0, codebook.label, weight.double())
    share = mass / mass.sum().clamp_min(1e-30)
    # UID support and its effective sample size, in one pass: one row per (token, user).
    stride = int(uid.max()) + 1
    pair = codebook.label.long() * stride + uid.long()
    unique_pair, inverse = torch.unique(pair, return_inverse=True)
    pair_mass = torch.zeros(unique_pair.numel(), dtype=torch.float64)
    pair_mass.index_add_(0, inverse, weight.double())
    pair_label = unique_pair // stride
    support = torch.bincount(pair_label, minlength=codebook.size)
    summed = torch.bincount(pair_label, weights=pair_mass, minlength=codebook.size)
    squared = torch.bincount(pair_label, weights=pair_mass.square(), minlength=codebook.size)
    ess = summed.square() / squared.clamp_min(1e-300)
    report = {
        "size": codebook.size,
        "assigned_states": int(torch.unique(codebook.label).numel()),
        "dead_codes": int((mass <= 0).sum()),
        "predictive_kl": {"mean": float((kl.double() * weight.double()).sum()
                                        / weight.double().sum().clamp_min(1e-30)),
                          "median": float(kl.double().quantile(0.5)),
                          "p90": float(kl.double().quantile(0.9)),
                          "p95": float(kl.double().quantile(0.95)),
                          "max": float(kl.max())},
        "usage": {"entropy_bits": float(-(share[share > 0] * share[share > 0].log2()).sum()),
                  "max_share": float(share.max()),
                  "min_share": float(share.min())},
        "support": {"uid_min": int(support.min()), "uid_median": float(support.median()),
                    "uid_max": int(support.max()), "ess_min": float(ess.min()),
                    "ess_median": float(ess.median())},
        "history": codebook.history,
    }
    if split_users is not None:
        fit_users, held_out = split_users
        for name, mask in (("fit", fit_users), ("held_out_uid", held_out)):
            selected = kl[mask]
            weights = weight[mask]
            report.setdefault("split_kl", {})[name] = {
                "states": int(mask.sum()),
                "mean": float((selected.double() * weights.double()).sum()
                              / weights.double().sum().clamp_min(1e-30))}
    return report


def held_out_split(uid: torch.Tensor, fraction: float, seed: int):
    """A deterministic user-level split of the reservoir for the held-out KL read.

    The split is by user, not by state: a held-out read that shares users with the fit half
    measures nothing (``01_deyi_encoder.md`` makes the same distinction for its audits).
    """
    if fraction <= 0:
        ones = torch.ones(uid.numel(), dtype=torch.bool)
        return ones, torch.zeros(uid.numel(), dtype=torch.bool)
    values = np.asarray(uid.detach().cpu(), dtype=np.uint64)
    keys = _mix(values + np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF))
    order = np.argsort(keys, kind="stable")
    cut = int(round(uid.numel() * (1.0 - float(fraction))))
    held = torch.zeros(uid.numel(), dtype=torch.bool)
    held[torch.from_numpy(order[cut:].astype(np.int64))] = True
    return (~held), held


def _mix(values: np.ndarray) -> np.ndarray:
    """splitmix64: a cheap, deterministic 64-bit mix used only to order users."""
    state = values.copy()
    state ^= state >> np.uint64(30)
    state *= np.uint64(0xBF58476D1CE4E5B9)
    state ^= state >> np.uint64(27)
    state *= np.uint64(0x94D049BB133111EB)
    state ^= state >> np.uint64(31)
    return state


def bits_per_state(kl_nats: float) -> float:
    return float(kl_nats) / math.log(2.0)
