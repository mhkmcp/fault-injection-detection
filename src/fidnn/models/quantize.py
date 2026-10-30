"""INT8 post-training static quantisation, FX, CPU-only (SPEC §3). Engine: see models/README.md."""

import copy
import warnings
from collections.abc import Iterable

import torch
from torch import nn
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

SUPPORTED_ENGINES = torch.backends.quantized.supported_engines
# x86 (fbgemm + oneDNN) on Intel/AMD Windows and Linux; qnnpack on Apple silicon and ARM
BACKEND = next((e for e in ("x86", "onednn", "fbgemm", "qnnpack") if e in SUPPORTED_ENGINES),
               None)


def quantize_ptq(model: nn.Module, calib_batches: Iterable[torch.Tensor]) -> nn.Module:
    """INT8 copy of `model`. See `prepare_fx` / `convert_fx`; calibrate on `clean_fit` only."""
    if BACKEND is None:
        raise RuntimeError(f"no quantised engine in this torch build: {SUPPORTED_ENGINES}")
    torch.backends.quantized.engine = BACKEND
    model = copy.deepcopy(model).cpu().eval()
    batches = list(calib_batches)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = prepare_fx(model, get_default_qconfig_mapping(BACKEND),
                              example_inputs=(batches[0],))
        with torch.no_grad():
            for x in batches:
                prepared(x)
        return convert_fx(prepared).eval()
