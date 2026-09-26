"""DeYi training objectives (``00-Reconstruct/01_deyi_encoder.md`` §10-§19).

    L = L_pred + lambda_user * L_user + lambda_div * L_div

* ``L_pred``  predictive compression: every interest forms its own catalog distribution and
  recent history only mixes them. Future targets are equally weighted and never deduplicated;
  only the positive *candidate* set is deduplicated.
* ``L_user``  cross-user discrimination: with the focus user's recent context and future, the
  focus user's own memory must explain that future better than another user's memory does.
* ``L_div``   interest-set geometry: ``-(2/(K(K-1))) log det(Z Z^T)`` in float64, which is the
  log-volume of the slot set rather than a sum of pairwise penalties.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_GRAM_JITTER = 1e-6


class CandidateAffinity:
    """The candidate side of the affinity, materialised once per micro-batch.

    ``items[candidate_index]`` is ``(B, W, d)`` — about 2 GB in fp32 at the shipped batch
    sizes — and it is identical for every slot set scored against this candidate list. The
    cross-user term scores up to ``user_negatives`` extra slot sets, so gathering per call
    multiplied both the peak memory and the cuBLAS workspace pressure by nine. Gather once
    here, in bounded row chunks, and reuse it for every slot set.
    """

    def __init__(self, candidate_content, candidate_index, chunk_size=16):
        self.index = candidate_index
        self.chunk_size = int(chunk_size)
        self.items = F.normalize(candidate_content.float(), dim=-1)

    def __call__(self, slot_states):
        slots = F.normalize(slot_states.float(), dim=-1)
        batch = slots.shape[0]
        scores = slots.new_empty(batch, slots.shape[1], self.index.shape[1])
        for start in range(0, batch, self.chunk_size):
            stop = min(start + self.chunk_size, batch)
            gathered = self.items[self.index[start:stop]]
            scores[start:stop] = torch.einsum("bkd,bcd->bkc", slots[start:stop], gathered)
        return scores


def _slot_log_probability(logits, negative_log_count, positive_mask, candidate_mask,
                          target_index):
    """``log p(y | z_k)`` for every target event: ``(B, K, T)``."""
    corrected = logits - negative_log_count.float()[:, None, :]
    corrected = torch.where(positive_mask[:, None, :], logits, corrected)
    corrected = corrected.masked_fill(~candidate_mask[:, None, :], float("-inf"))
    log_partition = torch.logsumexp(corrected, dim=-1)
    gather = target_index[:, None, :].expand(-1, logits.shape[1], -1)
    return logits.gather(2, gather) - log_partition[:, :, None]


def _mixture_log_probability(slot_log_probability, alpha):
    """``log sum_k alpha_k p(y | z_k)``: mixture in probability space, never in logit space."""
    weight = alpha.clamp_min(1e-30).log()[:, :, None]
    return torch.logsumexp(weight + slot_log_probability, dim=1)


def _row_mean(log_probability, valid):
    count = valid.sum(1).clamp_min(1).to(log_probability.dtype)
    return (log_probability * valid).sum(1) / count


def user_row_counts(uids):
    """How many supervised rows each user contributes, as a tensor aligned with ``uids``.

    Rows are weighted by ``1 / n_u`` so a user with many future windows does not outvote a
    user with few; the weight is computed per optimizer batch, which is exact within a batch
    and stable across batches because row membership only changes on shuffle.
    """
    values, inverse = np.unique(np.asarray(uids), return_inverse=True)
    counts = np.bincount(inverse, minlength=len(values)).astype(np.float64)
    return torch.as_tensor(counts[inverse], dtype=torch.float32)


def deyi_objective(
    model,
    *,
    states,
    state_mask,
    alpha,
    target_mask,
    target_index,
    candidate_content,
    candidate_index,
    candidate_mask,
    positive_mask,
    negative_log_count,
    row_weight,
    recent_content,
    recent_mask,
    other_index,
    item_temperature,
    lambda_user,
    lambda_div,
) -> dict[str, torch.Tensor]:
    if item_temperature <= 0:
        raise ValueError("item_temperature must be positive")
    # A row with no memory has no catalog distribution, and a row with no future target has
    # nothing to predict; neither may enter the objective.
    # Row-level eligibility, and the per-target mask it implies.
    valid_rows = target_mask.bool().any(dim=-1) & state_mask.any(dim=-1)
    valid = target_mask.bool() & valid_rows[:, None]
    if not valid.any():
        raise ValueError("no row has both memory and future targets")

    affinity = CandidateAffinity(candidate_content, candidate_index)
    logits = affinity(states) / float(item_temperature)
    slot_log_probability = _slot_log_probability(
        logits, negative_log_count, positive_mask, candidate_mask, target_index)
    mixture = _mixture_log_probability(slot_log_probability, alpha)
    prediction = -_row_mean(mixture, valid)
    own_score = _row_mean(mixture, valid)

    user_term = prediction.new_zeros(())
    if float(lambda_user) > 0 and other_index is not None:
        candidates_per_row = int(other_index.shape[1]) if other_index.dim() == 2 else 0
        if candidates_per_row:
            scores = [own_score]
            for slot in range(candidates_per_row):
                picked = other_index[:, slot]
                keep = picked >= 0
                safe = picked.clamp_min(0)
                # Borrow another row's memory but keep the focus row's recent context: the
                # readout must be recomputed, which is the whole point of the objective.
                other_states = states[safe]
                other_mask = state_mask[safe]
                other_alpha, _ = model.readout(other_states, other_mask, recent_content,
                                               recent_mask)
                other_logits = affinity(other_states) / float(item_temperature)
                other_slot = _slot_log_probability(other_logits, negative_log_count,
                                                   positive_mask, candidate_mask, target_index)
                other_mix = _mixture_log_probability(other_slot, other_alpha)
                score = _row_mean(other_mix, valid)
                # Unavailable negatives must never win the softmax (and must not be counted).
                scores.append(torch.where(keep, score, torch.full_like(score, -1e9)))
            stacked = torch.stack(scores, dim=1)
            user_term = -torch.log_softmax(stacked, dim=1)[:, 0]

    diversity = prediction.new_zeros(())
    if float(lambda_div) > 0 and int(state_mask.shape[1]) > 1:
        gram = (states.float() @ states.float().transpose(1, 2)).double()
        gram = gram + _GRAM_JITTER * torch.eye(gram.shape[-1], dtype=torch.float64,
                                               device=gram.device)
        log_volume = torch.linalg.slogdet(gram)[1]
        scale = 2.0 / (state_mask.shape[1] * (state_mask.shape[1] - 1))
        diversity = -scale * log_volume

    weight = row_weight.float() * valid_rows.to(row_weight.dtype)
    denominator = weight.sum().clamp_min(1e-12)
    total_prediction = (weight * prediction).sum() / denominator
    total_user = (weight * user_term).sum() / denominator
    total_diversity = (weight * diversity).sum() / denominator
    total = total_prediction + float(lambda_user) * total_user + float(lambda_div) * total_diversity

    with torch.no_grad():
        # Ranking is done in raw affinity space: the sampled-softmax proposal correction
        # belongs to the likelihood estimator and would penalise forced positives here.
        raw = affinity(states) / float(item_temperature)
        # Mixture over slots *per candidate*: (B, W). Rank each target's mixture score
        # against the same candidate list.
        mixture_raw = torch.logsumexp(
            alpha.clamp_min(1e-30).log()[:, :, None] + raw, dim=1)
        at_target = mixture_raw.gather(1, target_index)
        ranks = 1 + (mixture_raw[:, None, :] > at_target[:, :, None]).sum(-1)
        recall_at_1 = ranks[valid].le(1).float().mean()
        recall_at_10 = ranks[valid].le(10).float().mean()
        mean_rank = ranks.float()[valid].mean()

    return {
        "loss": total,
        "prediction": total_prediction,
        "user": total_user,
        "diversity": total_diversity,
        "mean_rank": mean_rank,
        "recall_at_1": recall_at_1,
        "recall_at_10": recall_at_10,
    }
