"""phi (v6) beam: the chain decodes one region then four constrained codebook digits."""

import numpy as np
import torch
from omegaconf import OmegaConf

from src.model.tiger import evaluate as tiger_eval
from src.model.tiger.model import Tiger


def _config(width=8, num_g1=4):
    return OmegaConf.create({
        "seed": 2026,
        "codebook": {"width": width},
        "model": {"d_model": 32, "d_ff": 64, "d_kv": 16, "num_heads": 2, "num_layers": 2,
                  "dropout": 0.0, "attn_implementation": "eager"},
        "train": {"prefix_sampling": 0.0},
        "deyi": {"enabled": False},
        "phi": {"variant": "tree", "num_g1": num_g1, "digit_width": width},
    })


_CATALOG = np.array([[0, 1, 2, 3], [0, 2, 1, 3], [1, 0, 3, 2], [1, 1, 1, 1]], dtype=np.int64)


def _catalog(width):
    codes = _CATALOG
    pids = np.array([10, 20, 30, 40], dtype=np.int64)
    return tiger_eval.PrefixIndex(codes, pids, width)


def test_phi_beam_decodes_region_then_constrained_codes():
    width = 8
    torch.manual_seed(0)
    model = Tiger(_config(width=width))
    model.eval()
    index = _catalog(width)
    rows, items = 3, 5
    batch = {
        "history_raw": torch.zeros((rows, items, 4), dtype=torch.long),
        "history_mask": torch.ones((rows, items), dtype=torch.bool),
        # One region per item; each slot carries (region, first codebook digit).
        "history_anchor": torch.zeros((rows, items), dtype=torch.long),
        "interest_anchor": torch.zeros((rows, 3, 2), dtype=torch.long),
        "interest_anchor_mask": torch.ones((rows, 3), dtype=torch.bool),
    }
    codes, scores = tiger_eval.generate(model, batch, index, beams=8)
    assert codes.shape == (rows, 8, 4)
    assert scores.shape == (rows, 8)
    assert np.isfinite(scores[:, 0]).all()
    # Every beam must decode to a code that exists in the catalog.
    keys, pids = index.lookup(codes.reshape(-1, 4))
    known = set((index.lookup(np.asarray([row]))[0][0] for row in _CATALOG))
    assert all(int(key) in known for key in keys.tolist()), codes[:, 0]


def test_phi_chain_heads_cover_five_levels():
    """v6: the region head (regions + ``none``) then the four codebook heads."""
    model = Tiger(_config())
    assert model.anchor_width == 5
    widths = [model.chain_head(depth).out_features for depth in range(5)]
    assert widths == [5, 8, 8, 8, 8]
