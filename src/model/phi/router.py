"""The distilled fast router (``03_phi.md`` §11).

Once the vocabulary exists, tokenizing every state through the head-tail estimator would make
the cost grow with the number of states again. Phi v300 therefore distils the reservoir's
predictive-KL labels into a small MLP

    R_phi(g | z) = softmax(MLP(z))

and tokenizes the full data with that. The router lives entirely offline: it is not part of the
generative model, is not executed at recommendation time, and never updates DeYi.

Its validation is reported as **predictive regret** rather than top-1 disagreement alone: a
router that disagrees on a tie costs nothing, while one that routes a state to a token with
visibly worse predictive geometry costs a lot, and only the regret separates the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from . import codebook as codebook_module


class MLP(nn.Module):
    """``linear -> GELU`` repeated, with ``layers`` linear maps and no activation on the last."""

    def __init__(self, dim: int, hidden: int, size: int, layers: int = 2):
        super().__init__()
        blocks = []
        width_in = int(dim)
        for index in range(int(layers) - 1):
            blocks += [nn.Linear(width_in, int(hidden)), nn.GELU()]
            width_in = int(hidden)
        blocks.append(nn.Linear(width_in, int(size)))
        self.network = nn.Sequential(*blocks)

    def forward(self, values):
        return self.network(values)


@dataclass
class Router:
    state_dict: dict
    config: dict
    validation: dict

    def build(self) -> MLP:
        module = MLP(int(self.config["dim"]), int(self.config["hidden"]),
                     int(self.config["size"]), int(self.config["layers"]))
        module.load_state_dict(self.state_dict)
        module.eval()
        return module


@torch.no_grad()
def logits(module: MLP, states: torch.Tensor, *, batch: int = 8192, device=None) -> torch.Tensor:
    states = torch.as_tensor(states, dtype=torch.float32)
    target = next(module.parameters()).device if device is None else device
    output = torch.empty((states.shape[0], int(module.network[-1].out_features)))
    for start in range(0, states.shape[0], int(batch)):
        stop = min(start + int(batch), states.shape[0])
        output[start:stop] = module(states[start:stop].to(target)).cpu()
    return output


def train(states: torch.Tensor, label: torch.Tensor, weight: torch.Tensor, *, dim: int, size: int,
          hidden: int = 256, layers: int = 2, epochs: int = 80, batch_size: int = 4096,
          learning_rate: float = 1e-3, weight_decay: float = 0.0, seed: int = 2026,
          validation: torch.Tensor | None = None, moment: torch.Tensor | None = None,
          reference: "codebook_module.Codebook" | None = None, temperature: float = 0.15,
          device=None, progress=None) -> Router:
    """Weighted cross-entropy distillation of the predictive-KL labels, with early stop."""
    target = torch.device("cpu") if device is None else device
    torch.manual_seed(int(seed))
    module = MLP(int(dim), int(hidden), int(size), int(layers)).to(target)
    optimiser = torch.optim.Adam(module.parameters(), lr=float(learning_rate),
                                 weight_decay=float(weight_decay))
    values = torch.as_tensor(states, dtype=torch.float32)
    labels = torch.as_tensor(label, dtype=torch.long)
    weights = torch.as_tensor(weight, dtype=torch.float32).clamp_min(0.0)
    if bool(validation is not None) and bool(validation.any()) and int((~validation).sum()) < 8:
        validation = None
    fit_index = torch.nonzero(~validation, as_tuple=False).squeeze(1) if validation is not None \
        else torch.arange(values.shape[0])
    hold_index = torch.nonzero(validation, as_tuple=False).squeeze(1) if validation is not None \
        else torch.zeros(0, dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    best = {"state": None, "validation_ce": float("inf"), "epoch": -1}
    history = []
    for epoch in range(int(epochs)):
        module.train()
        order = fit_index[torch.randperm(fit_index.numel(), generator=generator)]
        total = 0.0
        seen = 0
        for start in range(0, order.numel(), int(batch_size)):
            index = order[start:start + int(batch_size)]
            output = module(values[index].to(target))
            loss = nn.functional.cross_entropy(output, labels[index].to(target), reduction="none")
            weight_batch = weights[index].to(target)
            loss = (loss * weight_batch).sum() / weight_batch.sum().clamp_min(1e-12)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss) * int(index.numel())
            seen += int(index.numel())
        entry = {"epoch": epoch, "train_ce": total / max(seen, 1)}
        if hold_index.numel():
            module.eval()
            with torch.no_grad():
                output = logits(module, values[hold_index], batch=batch_size, device=target)
                hold_weight = weights[hold_index]
                ce = nn.functional.cross_entropy(output, labels[hold_index],
                                                 reduction="none")
                entry["validation_ce"] = float((ce * hold_weight).sum()
                                               / hold_weight.sum().clamp_min(1e-12))
            if entry["validation_ce"] < best["validation_ce"]:
                best = {"state": {name: value.detach().cpu().clone()
                                  for name, value in module.state_dict().items()},
                        "validation_ce": entry["validation_ce"], "epoch": epoch}
        history.append(entry)
        if progress is not None and (epoch % 10 == 0 or epoch == int(epochs) - 1):
            progress.line("router", "epoch=%d %s" % (epoch, entry), force=True)
    if best["state"] is None:
        best["state"] = {name: value.detach().cpu().clone()
                         for name, value in module.state_dict().items()}
    state = Router(state_dict=best["state"],
                   config={"dim": int(dim), "hidden": int(hidden), "size": int(size),
                           "layers": int(layers)},
                   validation={})
    state.validation = _validate(state, values, hold_index, moment, reference, temperature,
                                 device=target, batch=batch_size, history=history)
    return state


def _validate(config_router: Router, states: torch.Tensor, hold_index: torch.Tensor,
              moment, reference, temperature: float, *, device, batch: int, history) -> dict:
    """Top-1/top-2 agreement and the predictive routing regret on the held-out states."""
    report = {"epochs": len(history), "history": history[-5:]}
    if int(hold_index.numel()) == 0 or moment is None or reference is None:
        report["note"] = "no router-validation split was available"
        return report
    module = config_router.build().to(device)
    with torch.no_grad():
        output = logits(module, states[hold_index], batch=batch, device=device).to(device)
        predicted = output.argmax(dim=-1)
        scores = codebook_module.distance(moment[hold_index].to(device), reference.center.to(device),
                                          reference.log_partition.to(device), temperature)
        best = scores.argmin(dim=1)
        shown = scores.topk(min(2, scores.shape[1]), dim=1, largest=False).indices
        regret = scores.gather(1, predicted[:, None]).squeeze(1) - scores.gather(
            1, best[:, None]).squeeze(1)
        probability = torch.softmax(output, dim=-1)
    report.update({
        "states": int(hold_index.numel()),
        "top1_agreement": float((predicted == best).float().mean()),
        "top2_agreement": float((shown == predicted[:, None]).any(dim=1).float().mean()),
        "regret": {"mean": float(regret.mean()), "p95": float(regret.quantile(0.95)),
                   "max": float(regret.max()),
                   "zero_fraction": float((regret <= 1e-9).float().mean())},
        "confidence": {"mean": float(probability.max(dim=-1).values.mean())},
    })
    return report

