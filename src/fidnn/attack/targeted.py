"""Targeted bit search: flip INT8 weight bits until one input is read as a chosen class.

T-BFA style (Rakin et al. 2021). A demo addition, not a SPEC attacker level; see README.md.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from fidnn.attack import bfa
from fidnn.inject.bitflip import flip_quantized_weight_
from fidnn.inject.targets import is_quantized


@dataclass
class TargetedResult:
    success: bool
    target: int
    prediction: int                     # the input's class after the kept flips
    flips: list[bfa.Flip] = field(default_factory=list)


def _loss(model: nn.Module, x: torch.Tensor, target: torch.Tensor,
          keep: tuple[torch.Tensor, torch.Tensor] | None, keep_weight: float) -> torch.Tensor:
    loss = F.cross_entropy(model(x), target)
    if keep is not None and len(keep[1]):
        loss = loss + keep_weight * F.cross_entropy(model(keep[0]), keep[1])
    return loss


def search(int8: nn.Module, fp32: nn.Module, x: torch.Tensor, target: int,
           keep: tuple[torch.Tensor, torch.Tensor] | None = None, budget: int = 20,
           candidates_per_layer: int = 10, keep_weight: float = 1.0,
           avoid: set[tuple[str, int, int]] | None = None, log=print) -> TargetedResult:
    """Greedy loss-decreasing search on `int8`, which is mutated in place (pass a scratch copy).

    `x` is one input (1, C, H, W). `keep` = (inputs, labels) whose answers should not move, which
    is what makes the attack targeted rather than a collapse. `avoid` holds (layer, index, bit)
    triples already flipped elsewhere, since flipping one again would undo it.
    """
    layers = {n: m for n, m in int8.named_modules() if is_quantized(m)}
    t = torch.tensor([target])
    used = set(avoid or ())
    flips: list[bfa.Flip] = []

    def predict() -> int:
        with torch.no_grad():
            return int(int8(x).argmax(1)[0])

    pred = predict()
    for step in range(1, budget + 1):
        if pred == target:
            break
        twin, convs = bfa.surrogate(fp32, int8)
        twin.zero_grad(set_to_none=True)
        _loss(twin, x, t, keep, keep_weight).backward()
        with torch.no_grad():
            before = float(_loss(int8, x, t, keep, keep_weight))

        best, best_loss = None, before
        for name, module in layers.items():
            grad = convs[name].weight.grad
            if grad is None:
                continue
            # negated gradient: rank_candidates then keeps the loss-decreasing flips
            for idx, bit, _ in bfa.rank_candidates(module, -grad.detach().flatten(),
                                                   candidates_per_layer):
                if (name, idx, bit) in used:
                    continue
                flip_quantized_weight_(module, [idx], [bit])
                with torch.no_grad():
                    loss = float(_loss(int8, x, t, keep, keep_weight))
                flip_quantized_weight_(module, [idx], [bit])
                if loss < best_loss:
                    best, best_loss = bfa.Flip(name, idx, bit), loss
        if best is None:
            log(f"step {step}: no flip moves the input towards class {target}; stopping")
            break
        flip_quantized_weight_(layers[best.layer], [best.flat_index], [best.bit])
        used.add((best.layer, best.flat_index, best.bit))
        flips.append(best)
        pred = predict()
        log(f"step {step}: {best.layer}[{best.flat_index}] bit {best.bit}, "
            f"loss {before:.3f} → {best_loss:.3f}, prediction {pred}")
    return TargetedResult(pred == target, target, pred, flips)
