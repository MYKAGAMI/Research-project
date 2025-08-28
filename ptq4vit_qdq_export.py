"""
ptq4vit_qdq_export.py
---------------------
Export a ViT/DeiT model, calibrated by PTQ4ViT, into an ONNX graph with explicit
QuantizeLinear/DequantizeLinear (Q/DQ). This lets TensorRT run REAL INT8 kernels
on H100 while preserving (as much as possible) the quantization ranges learned by
your PTQ4ViT pipeline.

Usage (from your eval script, after PTQ4ViT calibration):
    from ptq4vit_qdq_export import export_qdq_onnx_from_model

    onnx_path = export_qdq_onnx_from_model(
        model=net,                               # the calibrated model (after wrap + calibrate)
        calib_loader=calib_loader,               # small loader for activation fallback if needed
        use_symmetric_weight=True,               # typical for weights
        use_symmetric_act=False,                 # typical to allow ZP for activations
        input_size=224,                          # 224 or 384
        onnx_path="vit_ptq4vit_qdq.onnx"
    )

Notes:
- We try to grab quantization params from PTQ4ViT-wrapped modules if available.
  If not found, we FALL BACK to min/max calibration collected via forward hooks.
- We replace nn.Linear with a QDQLinearExport which inserts Q/DQ ops around
  inputs & weights for ONNX. Forward still runs in fp32 for correctness.
- ONNX opset >= 13 recommended.

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
    around inputs and weights during ONNX export by using aten::quantize_per_* ops.
    Runtime in PyTorch remains fp32 (fake), but ONNX consumers (TensorRT) can fuse
    Q/DQ into real INT8 kernels.
    """
    def __init__(
        self,
        linear: nn.Linear,
        w_scales: torch.Tensor,     # per-channel or per-tensor weight scales (float32)
        w_zero_points: torch.Tensor,# same shape as w_scales, usually zeros for symmetric
        a_scale: float,             # per-tensor activation scale
        a_zero_point: int,          # per-tensor activation zp
        per_channel: bool = True
    ):
        super().__init__()
        self.inner = nn.Linear(linear.in_features, linear.out_features, bias=linear.bias is not None)
        # Copy weights/bias to avoid tying original params (safe for export)
        with torch.no_grad():
            self.inner.weight.copy_(linear.weight)
            if linear.bias is not None:
                self.inner.bias.copy_(linear.bias)

        # register buffers for constant folding & ONNX export
        self.register_buffer("w_scales", w_scales.float())
        self.register_buffer("w_zero_points", w_zero_points.int())
        self.register_buffer("a_scale", torch.tensor([a_scale], dtype=torch.float32))
        self.register_buffer("a_zero_point", torch.tensor([a_zero_point], dtype=torch.int32))

        self.per_channel = per_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Insert Q/DQ on activation (per-tensor, uint8 by default for activations)
        # NOTE: quantize_per_tensor expects zero_point dtype to match qdtype
        a_q = torch.quantize_per_tensor(x, float(self.a_scale.item()), int(self.a_zero_point.item()), torch.quint8)
        a_dq = torch.dequantize(a_q)

        # Insert Q/DQ on weight (per-channel qint8 along out_features=0)
        W = self.inner.weight
        if self.per_channel:
            # torch.quantize_per_channel requires scales/zero_points of shape [out_features]
            # axis=0 => per output channel
            w_q = torch.quantize_per_channel(
                W, self.w_scales, self.w_zero_points, axis=0, dtype=torch.qint8
            )
        else:
            w_q = torch.quantize_per_tensor(W, float(self.w_scales.item()), int(self.w_zero_points.item()), torch.qint8)
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

def _get_parent_name(name: str) -> Tuple[str, str]:
    parts = name.split(".")
    parent_name = ".".join(parts[:-1])
    child_name = parts[-1]
    return parent_name, child_name


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

def _try_ptq4vit_qparams(wrapped_modules: Dict[str, nn.Module], name: str, W: torch.Tensor):
    """
    Heuristic extraction of PTQ4ViT quant params from wrapper module.
    Returns (w_scales[OC], w_zero_points[OC], act_scale, act_zero_point) or None if not found.
    """
    m = wrapped_modules.get(name, None)
    if m is None:
        return None

    # Try common field names (best-effort; adjust if your PTQ4ViT wrappers differ)
    # We prefer symmetric weight int8 => zp=0, per-channel on out_features
    OC = W.shape[0]
    w_scales = None
    a_scale = None

    # weight scales candidates
    for cand in ["w_scale", "weight_scale", "w_scales", "weight_scales"]:
        if hasattr(m, cand):
            ws = getattr(m, cand)
            ws = torch.as_tensor(ws, dtype=torch.float32)
            if ws.numel() == OC:
                w_scales = ws
                break
            elif ws.numel() == 1:
                w_scales = ws.repeat(OC)
                break

    # activation scale candidates
    for cand in ["a_scale", "act_scale", "x_scale"]:
        if hasattr(m, cand):
            a_scale = float(torch.as_tensor(getattr(m, cand)).float().item())
            break

    if w_scales is None and hasattr(m, "w_clip"):  # e.g., symmetric max => scale = max/127
        w_max = getattr(m, "w_clip")
        w_scales = torch.full((OC,), float(w_max)/127.0, dtype=torch.float32)

    if a_scale is None and hasattr(m, "a_clip"):
        a_scale = float(getattr(m, "a_clip"))/255.0  # assuming uint8 affine

    if (w_scales is None) or (a_scale is None):
        return None

    w_zero_points = torch.zeros_like(w_scales, dtype=torch.int32)  # symmetric weight
    a_zero_point  = 128  # assume affine uint8; adjust if symmetric desired

    return (w_scales, w_zero_points, a_scale, a_zero_point)


# --------------- Fallback: compute qparams from min/max statistics ------------

def _symmetric_scale_per_channel(W: torch.Tensor, qmax: int = 127) -> torch.Tensor:
    # per output-channel (dim=0) scale = max(|W_i|) / qmax
    OC = W.shape[0]
    Wc = W.detach().abs().amax(dim=1) if W.dim() == 2 else W.detach().abs().amax(dim=(1,2,3))
    # Ensure shape [OC]
    Wc = Wc.reshape(OC)
    scales = (Wc / qmax).clamp(min=1e-12).to(torch.float32)
    return scales

def _affine_scale_zero(min_val: float, max_val: float, qmin: int = 0, qmax: int = 255):
    # per-tensor affine
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

        # 1) Try to get PTQ4ViT-provided qparams from wrapper
        wscale, wzp, ascale, azp = None, None, None, None
        if wrapped_modules is not None:
            got = _try_ptq4vit_qparams(wrapped_modules, name, W)
            if got is not None:
                wscale, wzp, ascale, azp = got

        # 2) Fallback to min/max
        if wscale is None:
            wscale = _symmetric_scale_per_channel(W, qmax=127)
            wzp = torch.zeros_like(wscale, dtype=torch.int32)
        if ascale is None or azp is None:
            mn, mx = act_ranges.get(name, (-1.0, 1.0))
            ascale, azp = _affine_scale_zero(mn, mx, qmin=0, qmax=255)

        qdq = QDQLinearExport(
            linear=lin,
            w_scales=wscale.float(),
            w_zero_points=wzp.int(),
            a_scale=float(ascale),
            a_zero_point=int(azp),
            per_channel=True
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
                               use_symmetric_weight: bool = True,
                               use_symmetric_act: bool = False,
                               device: str = "cuda"):
    """
    Convert a calibrated model into a Q/DQ ONNX.
    - If wrapped_modules provided, try to use its qparams; otherwise use fallback stats.
    - We only replace nn.Linear (major GEMM hotspots in ViT/DeiT).
      You can extend similarly for Conv if needed.
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
