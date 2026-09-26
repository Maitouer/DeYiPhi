import itertools

import numpy as np
import torch
from omegaconf import OmegaConf

from src.model.tiger.evaluate import PrefixIndex
from src.model.tiger.marginal_beam import generate_marginal, posterior_keep, cached_cross_projections
from src.model.tiger.model import Tiger


class Toy:
    width = 4
    anchor_width = 3

    def __init__(self):
        rng = torch.Generator().manual_seed(19)
        self.logits = torch.randn((2000, 4), generator=rng)

    def encode_history(self, history_raw, history_mask):
        return history_raw[:,:1,:1].float(), history_mask[:,:1]

    def decode_chain(self, hidden, mask, gs, prefix):
        key = hidden[:,0,0].long()
        if gs.shape[1]:
            key = key*3+gs[:,0]+1
        for digit in prefix.unbind(1):
            key = key*4+digit+1
        return key[:,None,None].float()

    def step_logits(self, decoded, depth):
        return self.logits[decoded[:,0,0].long(), :3 if depth == 0 else 4]


def test_exact_mixture_matches_independent_enumeration():
    model = Toy()
    catalog = np.array([p for p in itertools.product(range(4), repeat=4) if sum(p)%3 != 0])
    index = PrefixIndex(catalog, np.arange(len(catalog)), 4)
    batch = dict(history_raw=torch.arange(2).reshape(2,1,1).expand(2,1,4),
                 history_mask=torch.ones((2,1),dtype=torch.bool))
    actual, scores, _ = generate_marginal(model,batch,index,beams=3,posterior_mass=1,
                                         decode_chunk=7,cache_cross=False)
    for row in range(2):
        prior = model.logits[row,:3].log_softmax(0).numpy()
        candidates = [((), prior)]
        for depth in range(4):
            expanded = []
            for prefix, weights in candidates:
                conditionals = []
                for g in range(3):
                    key = row*3+g+1
                    for digit in prefix:
                        key = key*4+digit+1
                    conditionals.append(model.logits[key].log_softmax(0).numpy())
                for digit in index.allowed(prefix):
                    joint = weights+np.array(conditionals)[:,digit]
                    expanded.append((prefix+(int(digit),),joint))
            expanded.sort(key=lambda x:-float(np.logaddexp.reduce(x[1])))
            candidates = expanded[:3]
        np.testing.assert_array_equal(actual[row], [p for p,w in candidates])
        np.testing.assert_allclose(scores[row], [np.logaddexp.reduce(w) for p,w in candidates], atol=2e-6)


def test_posterior_minimal_support_and_no_renormalization():
    weights = torch.tensor([[.6,.3,.1], [.9995,.0004,.0001]]).log()
    order, keep, retained = posterior_keep(weights,.999)
    assert keep.sum(1).tolist() == [3,1]
    torch.testing.assert_close(retained, torch.tensor([1.,.9995]))
    assert weights.gather(1,order)[1,0] == weights[1,0]


def test_budgeted_mixture_is_unique_and_has_bounded_support():
    model = Toy()
    catalog = np.array(list(itertools.product(range(4),repeat=4)))
    index = PrefixIndex(catalog,np.arange(len(catalog)),4)
    batch = dict(history_raw=torch.zeros((1,1,4),dtype=torch.long),
                 history_mask=torch.ones((1,1),dtype=torch.bool))
    codes,scores,stats = generate_marginal(model,batch,index,beams=3,cache_cross=False,
        initial_g=2,max_g=1)
    assert len(set(map(tuple,codes[0]))) == 3
    assert np.isfinite(scores).all()
    assert stats[0]['max_states'] == 2
    assert all(x['max_states'] <= 1 for x in stats[1:])
    assert all(0 < x['min_retained'] <= 1.000001 for x in stats)


def test_cached_cross_projection_matches_decoder_and_restores():
    cfg = OmegaConf.create(dict(codebook=dict(width=4),model=dict(d_model=16,d_ff=32,d_kv=8,
        num_heads=2,num_layers=1,dropout=0,attn_implementation='eager'),phi=dict(variant='tree',num_g1=2)))
    model = Tiger(cfg).eval()
    hidden = torch.randn(2,5,16)
    mask = torch.ones(2,5,dtype=torch.bool)
    owners = torch.tensor([1,0,1,1])
    g = torch.tensor([[0],[1],[2],[0]])
    sid = torch.tensor([[0,1],[2,1],[1,0],[3,2]])
    with torch.inference_mode():
        expected = model.decode_chain(hidden[owners],mask[owners],g,sid)
        original = model.decoder.block[0].layer[1].EncDecAttention.k.forward
        with cached_cross_projections(model,hidden) as cache:
            cache['owners'] = owners
            actual = model.decode_chain(hidden[owners],mask[owners],g,sid)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        assert model.decoder.block[0].layer[1].EncDecAttention.k.forward == original
