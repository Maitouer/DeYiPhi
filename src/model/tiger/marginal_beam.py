"""Native-SID beam with a posterior mixture of G paths inside each prefix.

By default all G participate in the first SID decision. The budgeted mode limits
initial G and per-prefix G support explicitly. Posterior truncation retains
unnormalized joint probabilities: discarded mass is never redistributed.
"""
from contextlib import contextmanager

import torch

from .evaluate import DevicePrefixIndex, _ranked_topk


@contextmanager
def cached_cross_projections(model, hidden, enabled=True):
    """Reuse immutable encoder K/V projections; owner indices follow decoder paths."""
    state = {'owners': torch.arange(len(hidden), device=hidden.device)}
    restored = []
    try:
        if enabled:
            for block in model.decoder.block:
                attention = block.layer[1].EncDecAttention
                for linear in (attention.k, attention.v):
                    original = linear.forward
                    projected = original(hidden)
                    def forward(_input, projected=projected):
                        return projected.index_select(0, state['owners'])
                    restored.append((linear, original))
                    linear.forward = forward
        yield state
    finally:
        for linear, original in restored:
            linear.forward = original


def grouped_logsumexp(values, groups, count):
    indices = groups[:, None].expand_as(values)
    maximum = values.new_full((count, values.shape[1]), -torch.inf)
    maximum.scatter_reduce_(0, indices, values, reduce='amax', include_self=True)
    centered = values - maximum.index_select(0, groups)
    centered = centered.masked_fill(~torch.isfinite(values), -torch.inf)
    total = torch.zeros_like(maximum).scatter_add_(0, indices, centered.exp())
    return maximum + total.log()


def posterior_keep(log_weights, mass):
    """Minimal highest-probability support reaching mass, stable input tie order."""
    if not 0 < mass <= 1:
        raise ValueError('posterior mass must lie in (0, 1]')
    order = torch.argsort(log_weights, dim=1, descending=True, stable=True)
    sorted_weights = log_weights.gather(1, order)
    normalizer = torch.logsumexp(sorted_weights, dim=1, keepdim=True)
    probabilities = (sorted_weights-normalizer).exp().nan_to_num(0)
    if mass == 1:
        keep = torch.isfinite(sorted_weights)
    else:
        before = probabilities.cumsum(1)-probabilities
        keep = (before < mass) & torch.isfinite(sorted_weights)
    retained = (probabilities*keep).sum(1)
    return order, keep, retained


@torch.inference_mode()
def generate_marginal(model, batch, index, beams=32, device_index=None,
                      posterior_mass=.999, decode_chunk=1024, cache_cross=True,
                      initial_g=None, max_g=None):
    device = batch['history_raw'].device
    device_index = device_index or DevicePrefixIndex(index, device)
    encoder = ('history_raw', 'history_mask', 'continuous_states', 'continuous_mask',
               'history_anchor', 'interest_anchor', 'interest_anchor_mask',
               'interest_codes', 'interest_mask', 'interest_channels')
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        hidden, mask = model.encode_history(**{k:v for k,v in batch.items() if k in encoder})
        rows, groups = len(hidden), int(model.anchor_width)
        stats = []
        with cached_cross_projections(model, hidden, cache_cross) as cache:
            empty = torch.empty((rows, 0), dtype=torch.long, device=device)
            prior = model.step_logits(model.decode_chain(hidden, mask, empty, empty), 0).float().log_softmax(-1)
            prefixes = torch.zeros((rows, 4), dtype=torch.long, device=device)
            if initial_g is None:
                chosen_g = torch.arange(groups, device=device).expand(rows,-1)
            else:
                chosen_g = torch.argsort(prior,dim=1,descending=True,stable=True)[:,:initial_g]
            initial_weights = prior.gather(1,chosen_g)
            coverage = initial_weights.exp().sum(1)
            stats.append(dict(depth=0,prefixes=rows,states=chosen_g.numel(),
                              max_states=chosen_g.shape[1],min_retained=float(coverage.min()),
                              retained_sum=float(coverage.sum())))
            parents = torch.arange(rows, device=device).repeat_interleave(chosen_g.shape[1])
            gs = chosen_g.flatten()
            weights = initial_weights.flatten()
            previous_beams = 1
            for depth in range(4):
                # Joint scores for every retained (SID prefix, G) and next native digit.
                joint = torch.empty((len(gs), model.width), device=device, dtype=torch.float32)
                for start in range(0, len(gs), decode_chunk):
                    stop = min(start+decode_chunk, len(gs))
                    parent = parents[start:stop]
                    owner = torch.div(parent, previous_beams, rounding_mode='floor')
                    cache['owners'] = owner
                    decoded = model.decode_chain(hidden.index_select(0, owner), mask.index_select(0, owner),
                        gs[start:stop,None], prefixes[parent,:depth])
                    joint[start:stop] = model.step_logits(decoded, depth+1).float().log_softmax(-1)+weights[start:stop,None]
                marginal = grouped_logsumexp(joint, parents, len(prefixes))
                digits, valid = device_index.candidates(prefixes[:,:depth])
                candidate = marginal.gather(1, digits.long()).masked_fill(~valid, -torch.inf)
                width = digits.shape[1]
                flat = candidate.reshape(rows, previous_beams*width)
                scores, selected, _ = _ranked_topk(flat, torch.isfinite(flat), beams)
                parent = torch.div(selected, width, rounding_mode='floor')
                chosen_digits = digits.reshape(rows,-1).gather(1, selected).long().flatten()
                selected_parent = (parent+torch.arange(rows, device=device)[:,None]*previous_beams).flatten()
                new_prefixes = prefixes.index_select(0, selected_parent)
                new_prefixes[:,depth] = chosen_digits
                if depth == 3:
                    return new_prefixes.reshape(rows,beams,4).cpu().numpy(), scores.cpu().numpy(), stats
                # State rows are contiguous by parent throughout. Gather all G support
                # of the selected SID children, then sparsify by their own posterior.
                counts = torch.bincount(parents, minlength=len(prefixes))
                starts = counts.cumsum(0)-counts
                slots = torch.arange(int(counts.max()), device=device)
                state_index = starts[selected_parent,None]+slots[None]
                available = slots[None] < counts[selected_parent,None]
                state_index = state_index.clamp_max(len(gs)-1)
                child_weights = joint[state_index, chosen_digits[:,None]].masked_fill(~available, -torch.inf)
                order, keep, retained = posterior_keep(child_weights, posterior_mass)
                state_index = state_index.gather(1, order)
                child_weights = child_weights.gather(1, order)
                if max_g is not None:
                    keep &= torch.arange(keep.shape[1],device=device)[None] < max_g
                    probabilities = (child_weights-torch.logsumexp(child_weights,dim=1,keepdim=True)).exp().nan_to_num(0)
                    retained = (probabilities*keep).sum(1)
                active = torch.isfinite(scores.flatten())
                sizes = keep.sum(1)
                stats.append(dict(depth=depth+1, prefixes=int(active.sum()),
                    states=int(sizes.sum()), max_states=int(sizes.max()),
                    min_retained=float(retained[active].min()),
                    retained_sum=float(retained[active].sum())))
                parents, columns = torch.where(keep)
                gs = gs[state_index[parents, columns]]
                weights = child_weights[parents, columns]
                prefixes, previous_beams = new_prefixes, beams
    raise AssertionError('unreachable')
