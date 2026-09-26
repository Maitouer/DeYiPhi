"""Full-state tokenization through the frozen router (``03_phi.md`` §12).

After the router is frozen, tokenizing a state needs no ANN search, no catalog sampling, no
softmax partition and no ``mu``: the cost is ``O(S * d * L)`` and no longer touches
``|C|`` at all. That decoupling is the point of the whole v300 construction.
"""

from __future__ import annotations

import torch

from . import router as router_module

FORMAT = "phi_tokens_v1"


def run(arm, module, split: str, *, vocabulary: int, batch: int = 8192, device=None,
        progress=None) -> dict:
    """``g_rk`` for every row and slot of one split; invalid slots keep ``-1``."""
    payload = arm.payload(split)
    states = payload["states"]
    mask = payload["mask"].bool()
    rows, slots = int(mask.shape[0]), int(mask.shape[1])
    codes = torch.full((rows, slots), -1, dtype=torch.int16)
    confidence = torch.zeros((rows, slots), dtype=torch.float16)
    target = device if device is not None else next(module.parameters()).device
    for start in range(0, rows, int(batch)):
        stop = min(start + int(batch), rows)
        block = states[start:stop].float().to(target)
        flat = block.reshape(-1, block.shape[-1])
        output = router_module.logits(module, flat, batch=int(batch), device=device)
        probability = torch.softmax(output, dim=-1)
        best = probability.argmax(dim=-1).to(torch.int16).reshape(stop - start, slots).cpu()
        top = probability.max(dim=-1).values.reshape(stop - start, slots).to(torch.float16).cpu()
        keep = mask[start:stop]
        codes[start:stop] = torch.where(keep, best, torch.full_like(best, -1))
        confidence[start:stop] = torch.where(keep, top, torch.zeros_like(top))
        if progress is not None:
            progress.tick(stop, rows, "%s tokens %d/%d" % (split, stop, rows))
    return {"format": FORMAT, "split": split, "rows": rows, "slots": slots,
            "vocabulary": int(vocabulary), "source_row_idx": payload["source_row_idx"].clone(),
            "codes": codes, "mask": mask.clone(), "confidence": confidence,
            "valid_slots": int(mask.sum())}
