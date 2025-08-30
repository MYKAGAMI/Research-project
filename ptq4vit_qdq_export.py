"""
ptq4vit_qdq_export.py  (per-tensor weight Q/DQ export)
-----------------------------------------------------
Export a ViT/DeiT model, calibrated by PTQ4ViT, into an ONNX graph with explicit
QuantizeLinear/DequantizeLinear (Q/DQ). We use per-tensor weight quantization
to keep ONNX export compatible (PyTorch exporter does not support
aten::quantize_per_channel to ONNX directly in many versions/opsets).

This yields REAL INT8 execution in TensorRT/H100 while preserving
PTQ4ViT calibration as much as possible.

Author: ChatGPT (GPT-5 Thinking)
"""

from collections import OrderedDict
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- QDQ Linear Wrapper -----------------------------

class QDQLinearExport(nn.Module):
    """
    A wrapper around nn.Linear that injects QuantizeLinear/DequantizeLinear
    around inputs (uint8 affine) and weights (int8 per-tensor) during ONNX export
    by using aten::quantize_per_tensor/aten::dequantize ops.

    Runtime in PyTorch remains fp32; ONNX consumers (e.g., TensorRT) can fuse
    Q/DQ into real INT8 kernels.
    """
    def __init__(
        self,
        linear: nn.Linear,
        w_scale: float,             # per-tensor weight scale
        w_zero_point: int,          # usually 0 for symmetric int8 weights
        a_scale: float,             # per-tensor activation scale
        a_zero_point: int           # e.g., 128 for uint8 affine
    ):
        super().__init__()
        self.inner = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None)
        with torch.no_grad():
            self.inner.weight.copy_(linear.weight)
            if linear.bias is not None:
                self.inner.bias.copy_(linear.bias)

        # register buffers for constant folding & ONNX export
        self.register_buffer("w_scale", torch.tensor([w_scale], dtype=torch.float32))
        self.register_buffer("w_zero_point", torch.tensor([w_zero_point], dtype=torch.int32))
        self.register_buffer("a_scale", torch.tensor([a_scale], dtype=torch.float32))
        self.register_buffer("a_zero_point", torch.tensor([a_zero_point], dtype=torch.int32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Activation Q/DQ (uint8 affine)
        a_q = torch.quantize_per_tensor(
            x, float(self.a_scale.item()), int(self.a_zero_point.item()), torch.quint8
        )
        a_dq = torch.dequantize(a_q)

        # Weight Q/DQ (int8 per-tensor, symmetric)
        W = self.inner.weight
        w_q = torch.quantize_per_tensor(
            W, float(self.w_scale.item()), int(self.w_zero_point.item()), torch.qint8
        )
        w_dq = torch.dequantize(w_q)

        return F.linear(a_dq, w_dq, self.inner.bias)


# -------------------------- Helpers: module traversal -------------------------

def _named_modules_of_type(model: nn.Module, target_type) -> OrderedDict:
    out = OrderedDict()
    for name, m in model.named_modules():
        if isinstance(m, target_type):
            out[name] = m
    return out

def _set_module_by_name(root: nn.Module, name: str, new_m: nn.Module):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_m)


# ------------------ Collect activation stats via forward hooks -----------------

@torch.no_grad()
def _collect_activation_ranges(model: nn.Module, calib_loader, linear_names, device="cuda", max_batches: int = 50):
    """
    Collect per-tensor min/max of inputs to all Linear layers listed in linear_names.
    Returns dict: name -> (amin, amax)
    """
    stats: Dict[str, Tuple[float, float]] = {n: (float("+inf"), float("-inf")) for n in linear_names}

    handles = []
    def make_hook(nm):
        def hook(module, inputs):
            x = inputs[0].detach()
            mn = float(x.amin().cpu().item())
            mx = float(x.amax().cpu().item())
            old_mn, old_mx = stats[nm]
            stats[nm] = (min(old_mn, mn), max(old_mx, mx))
        return hook

    # register pre-forward hooks on each Linear
    for nm, m in model.named_modules():
        if nm in linear_names and isinstance(m, nn.Linear):
            handles.append(m.register_forward_pre_hook(make_hook(nm)))

    # run few batches
    it = 0
    for inp, _ in calib_loader:
        if it >= max_batches:
            break
        it += 1
        inp = inp.to(device)
        _ = model(inp)

    for h in handles:
        h.remove()

    return stats


# -------------------- Extract qparams from PTQ4ViT wrappers --------------------

def _try_ptq4vit_qparams(wrapped_modules: Dict[str, nn.Module], name: str):
    """
    Heuristic extraction of PTQ4ViT qparams from wrapper module.
    Returns (w_scale_tensor_or_scalar, a_scale_scalar) if possible, else None.
    NOTE: We'll convert weight scale to a single per-tensor scale later.
    """
    m = wrapped_modules.get(name, None)
    if m is None:
        return None

    w_scale = None
    a_scale = None

    for cand in ["w_scale", "weight_scale", "w_scales", "weight_scales"]:
        if hasattr(m, cand):
            w_scale = getattr(m, cand)
            break
    for cand in ["a_scale", "act_scale", "x_scale"]:
        if hasattr(m, cand):
            a_scale = float(torch.as_tensor(getattr(m, cand)).float().item())
            break

    if (w_scale is None) and hasattr(m, "w_clip"):
        # symmetric max => scale ≈ max/127  (rough heuristic)
        w_max = float(getattr(m, "w_clip"))
        w_scale = torch.tensor([w_max / 127.0], dtype=torch.float32)

    if (a_scale is None) and hasattr(m, "a_clip"):
        a_scale = float(getattr(m, "a_clip")) / 255.0  # assuming uint8 affine

    if (w_scale is None) or (a_scale is None):
        return None
    return w_scale, a_scale


# --------------- Fallback: compute qparams from min/max statistics ------------

def _symmetric_weight_scale_per_tensor(W: torch.Tensor, qmax: int = 127) -> float:
    # per-tensor symmetric: scale = max(|W|)/qmax
    m = float(W.detach().abs().amax().cpu().item())
    if m < 1e-12:
        m = 1e-12
    return m / qmax

def _affine_scale_zero(min_val: float, max_val: float, qmin: int = 0, qmax: int = 255):
    # per-tensor affine for activations
    rng = max_val - min_val
    if rng < 1e-12:
        rng = 1e-12
    scale = rng / (qmax - qmin)
    zp = int(round(qmin - min_val/scale))
    zp = max(qmin, min(qmax, zp))
    return float(scale), int(zp)


# ------------------------- Build QDQ-wrapped model ----------------------------

def _replace_linears_with_qdq(model: nn.Module,
                              wrapped_modules: Optional[Dict[str, nn.Module]],
                              act_ranges: Dict[str, Tuple[float, float]]) -> nn.Module:
    new_model = model  # replace in-place
    linears = _named_modules_of_type(new_model, nn.Linear)

    for name, lin in linears.items():
        W = lin.weight.detach().cpu()

        # Try to reuse PTQ4ViT qparams if present (then collapse weight scales to per-tensor)
        w_scale_tensor = None
        a_scale = None
        if wrapped_modules is not None:
            got = _try_ptq4vit_qparams(wrapped_modules, name)
            if got is not None:
                w_scale_tensor, a_scale = got
                w_scale_tensor = torch.as_tensor(w_scale_tensor, dtype=torch.float32).reshape(-1)

        # Weight per-tensor scale (collapse if we had per-channel)
        if w_scale_tensor is not None and w_scale_tensor.numel() > 0:
            w_scale = float(w_scale_tensor.abs().max().item())
        else:
            w_scale = _symmetric_weight_scale_per_tensor(W, qmax=127)

        # Activation per-tensor affine scale/zp
        if a_scale is None:
            mn, mx = act_ranges.get(name, (-1.0, 1.0))
            a_scale, a_zp = _affine_scale_zero(mn, mx, qmin=0, qmax=255)
        else:
            # have a_scale from PTQ4ViT; assume uint8 affine zp=128
            a_zp = 128

        qdq = QDQLinearExport(
            linear=lin,
            w_scale=float(max(w_scale, 1e-12)),
            w_zero_point=0,          # symmetric int8 weights
            a_scale=float(max(a_scale, 1e-12)),
            a_zero_point=int(a_zp)
        )
        _set_module_by_name(new_model, name, qdq)

    return new_model


# ----------------------------- Public API -------------------------------------

@torch.no_grad()
def export_qdq_onnx_from_model(model: nn.Module,
                               calib_loader,
                               wrapped_modules: Optional[Dict[str, nn.Module]] = None,
                               input_size: int = 224,
                               onnx_path: str = "model_qdq.onnx",
                               opset: int = 13,
                               device: str = "cuda"):
    """
    Convert a calibrated model into a Q/DQ ONNX (weights per-tensor INT8, activations per-tensor UINT8).
    """
    model = model.eval().to(device)

    # collect per-layer activation ranges
    linear_names = list(_named_modules_of_type(model, nn.Linear).keys())
    act_ranges = _collect_activation_ranges(model, calib_loader, linear_names, device=device, max_batches=50)

    # build QDQ-wrapped model
    qdq_model = _replace_linears_with_qdq(model, wrapped_modules, act_ranges).eval().to(device)

    # dummy input
    dummy = torch.randn(1, 3, input_size, input_size, device=device)

    # export
    dynamic_axes = {"input": {0: "N"}, "logits": {0: "N"}}
    torch.onnx.export(
        qdq_model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes
    )
    return onnx_path
