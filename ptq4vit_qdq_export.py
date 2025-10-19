# ptq4vit_qdq_export.py
# 导出带显式 Q/DQ 的 ONNX（激活：uint8 per-tensor；权重：int8 per-tensor用于导出稳定性）
# 另外将每个 Linear 的“逐通道”权重 scale 保存为 .npz，供后处理写回 per-channel INT8 使用。

from collections import OrderedDict
from typing import Dict, Tuple, Optional
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- Q/DQ Linear wrapper -----------------------------
class QDQLinearExport(nn.Module):
    """
    激活：per-tensor uint8；权重：per-tensor int8（仅用于导出）。
    真正 per-channel 的权重 scale 另行保存为 .npz，便于后处理写回。
    """
    def __init__(self, linear: nn.Linear, w_scale: float, w_zero_point: int,
                 a_scale: float, a_zero_point: int):
        super().__init__()
        self.inner = nn.Linear(linear.in_features, linear.out_features,
                               bias=linear.bias is not None)
        with torch.no_grad():
            self.inner.weight.copy_(linear.weight)
            if linear.bias is not None:
                self.inner.bias.copy_(linear.bias)
        self.register_buffer("w_scale", torch.tensor([w_scale], dtype=torch.float32))
        self.register_buffer("w_zero_point", torch.tensor([w_zero_point], dtype=torch.int32))
        self.register_buffer("a_scale", torch.tensor([a_scale], dtype=torch.float32))
        self.register_buffer("a_zero_point", torch.tensor([a_zero_point], dtype=torch.int32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_q = torch.quantize_per_tensor(
            x, float(self.a_scale.item()), int(self.a_zero_point.item()), torch.quint8
        )
        a_dq = torch.dequantize(a_q)
        W = self.inner.weight
        w_q = torch.quantize_per_tensor(
            W, float(self.w_scale.item()), int(self.w_zero_point.item()), torch.qint8
        )
        w_dq = torch.dequantize(w_q)
        return F.linear(a_dq, w_dq, self.inner.bias)

# -------------------------- utils: traverse & hooks ----------------------------
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

@torch.no_grad()
def _collect_activation_ranges(model: nn.Module, calib_loader, linear_names,
                               device="cuda", max_batches: int = 50):
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
    for nm, m in model.named_modules():
        if nm in linear_names and isinstance(m, nn.Linear):
            handles.append(m.register_forward_pre_hook(make_hook(nm)))
    it = 0
    for inp, _ in calib_loader:
        if it >= max_batches: break
        it += 1
        inp = inp.to(device, non_blocking=True)
        _ = model(inp)
    for h in handles: h.remove()
    return stats

# -------------------- optional: get qparams from external wrappers -------------
def _try_ptq4vit_qparams(wrapped_modules: Optional[Dict[str, nn.Module]], name: str):
    if not wrapped_modules:
        return None
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
        w_scale = torch.tensor([float(getattr(m, "w_clip"))/127.0], dtype=torch.float32)
    if (a_scale is None) and hasattr(m, "a_clip"):
        a_scale = float(getattr(m, "a_clip"))/255.0
    if (w_scale is None) or (a_scale is None):
        return None
    return w_scale, a_scale

# ------------------------ fallbacks for qparams --------------------------------
def _symmetric_weight_scales_per_channel(W: torch.Tensor, qmax: int = 127) -> torch.Tensor:
    # 每个输出通道各取 |W|_max，得到对称量化的 scale
    OC = W.shape[0]
    Wc = W.detach().abs().amax(dim=1).reshape(OC)
    return (Wc / qmax).clamp(min=1e-12).to(torch.float32)

def _affine_scale_zero(min_val: float, max_val: float, qmin: int = 0, qmax: int = 255):
    rng = max_val - min_val
    if rng < 1e-12: rng = 1e-12
    scale = rng / (qmax - qmin)
    zp = int(round(qmin - min_val/scale))
    zp = max(qmin, min(qmax, zp))
    return float(scale), int(zp)

# ----------------------- wrap linears & dump scales ----------------------------
def _replace_linears_with_qdq_and_dump_scales(model: nn.Module,
                                              wrapped_modules: Optional[Dict[str, nn.Module]],
                                              act_ranges: Dict[str, Tuple[float, float]],
                                              dump_npz_path: str) -> nn.Module:
    new_model = model
    linears = _named_modules_of_type(new_model, nn.Linear)
    dump = {}
    for name, lin in linears.items():
        W = lin.weight.detach().cpu()
        w_scale_tensor = None
        a_scale = None

        # 先尝试从外部包装器读取（若有）
        got = _try_ptq4vit_qparams(wrapped_modules, name) if wrapped_modules else None
        if got is not None:
            w_scale_tensor, a_scale = got
            w_scale_tensor = torch.as_tensor(w_scale_tensor, dtype=torch.float32).reshape(-1)

        # 回退：按每通道 |W|_max 估计对称量化 scale
        if (w_scale_tensor is None) or (w_scale_tensor.numel() not in (1, W.shape[0])):
            w_scale_tensor = _symmetric_weight_scales_per_channel(W, qmax=127)

        dump[f"{name}.inner.weight"] = w_scale_tensor.cpu().numpy()  # 保存 per-channel scale

        # 导出 ONNX 时仍使用“per-tensor”以稳定图结构
        w_scale_pt = float(max(w_scale_tensor.abs().max().item(), 1e-12))
        if a_scale is None:
            mn, mx = act_ranges.get(name, (-1.0, 1.0))
            a_scale, a_zp = _affine_scale_zero(mn, mx, qmin=0, qmax=255)
        else:
            a_zp = 128  # 对称零点近似

        qdq = QDQLinearExport(
            linear=lin,
            w_scale=w_scale_pt, w_zero_point=0,
            a_scale=float(max(a_scale, 1e-12)), a_zero_point=int(a_zp)
        )
        _set_module_by_name(new_model, name, qdq)

    os.makedirs(os.path.dirname(dump_npz_path) or ".", exist_ok=True)
    np.savez(dump_npz_path, **dump)
    print(f"[Dump] per-channel weight scales saved to: {dump_npz_path}")
    return new_model

# ----------------------------- public API -------------------------------------
@torch.no_grad()
def export_qdq_onnx_from_model(model: nn.Module,
                               calib_loader,
                               wrapped_modules: Optional[Dict[str, nn.Module]] = None,
                               input_size: int = 224,
                               onnx_path: str = "model_qdq.onnx",
                               scales_npz: Optional[str] = None,
                               opset: int = 13,
                               device: str = "cuda",
                               keep_qdq: bool = True):
    """
    产物：
      - ONNX: 显式 Q/DQ（默认 keep_qdq=True 禁止常量折叠，避免 Q->DQ(Constant) 被折叠回 FP32）
      - NPZ : 每个 Linear 的 per-channel 权重 scale（写回时用）
    """
    model = model.eval().to(device)
    linear_names = list(_named_modules_of_type(model, nn.Linear).keys())
    act_ranges = _collect_activation_ranges(model, calib_loader, linear_names, device=device, max_batches=50)

    if scales_npz is None:
        base, _ = os.path.splitext(onnx_path)
        scales_npz = base + "_scales.npz"

    qdq_model = _replace_linears_with_qdq_and_dump_scales(
        model, wrapped_modules, act_ranges, scales_npz
    ).eval().to(device)

    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    dynamic_axes = {"input": {0: "N"}, "logits": {0: "N"}}

    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    torch.onnx.export(
        qdq_model, dummy, onnx_path,
        input_names=["input"], output_names=["logits"],
        opset_version=opset,
        do_constant_folding=not keep_qdq,  # keep_qdq=True -> 禁止折叠
        dynamic_axes=dynamic_axes
    )
    print(f"[OK] ONNX saved to: {onnx_path}")
    return onnx_path, scales_npz

# ------------------------------ CLI entrypoint --------------------------------
if __name__ == "__main__":
    import argparse, timm
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    parser = argparse.ArgumentParser("PTQ4ViT Q/DQ ONNX Export")
    parser.add_argument("--model", type=str, default="vit_b16")
    parser.add_argument("--pretrained", action="store_true", help="use timm pretrained weights")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to .pth checkpoint")
    parser.add_argument("--data-root", type=str, required=True, help="ImageNet val root (class-subdir layout)")
    parser.add_argument("--num-calib", type=int, default=1024, help="number of images for calibration")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--out", type=str, default="onnx/vit_qdq.onnx")
    parser.add_argument("--keep-qdq", action="store_true",
                        help="keep Q/DQ by disabling constant folding (recommended)")
    # 为保持与你之前命令兼容，支持 --fold，但这里我们将其解释为“保持 Q/DQ”（即禁用折叠）
    parser.add_argument("--fold", action="store_true",
                        help="(compat) same as --keep-qdq: keep Q/DQ; do NOT fold constants")
    args = parser.parse_args()

    # 目录准备
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # 1) 构建模型
    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=1000)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        # 去掉可能的 "module." 前缀
        if isinstance(sd, dict):
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[Load] missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print("[Warn] checkpoint format not recognized; skipped loading")

    # 2) 构建校准集 DataLoader（ImageNet val 常用变换）
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tval = transforms.Compose([
        transforms.Resize(int(args.input_size * 256 / 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        normalize,
    ])
    ds = datasets.ImageFolder(args.data_root, transform=tval)
    n = min(args.num_calib, len(ds))
    calib_ds = Subset(ds, list(range(n)))
    calib_loader = DataLoader(calib_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers, pin_memory=True)

    # 3) 导出
    device = "cuda" if torch.cuda.is_available() else "cpu"
    keep_qdq = args.keep_qdq or args.fold or True  # 缺省保持 Q/DQ，防止被折叠
    onnx_path, scales_npz = export_qdq_onnx_from_model(
        model=model,
        calib_loader=calib_loader,
        wrapped_modules=None,              # 若你使用过 PTQ4ViT 自定义 wrapper，可传 {name: module}
        input_size=args.input_size,
        onnx_path=args.out,
        scales_npz=None,
        opset=args.opset,
        device=device,
        keep_qdq=keep_qdq,
    )
    print(f"[DONE] ONNX: {onnx_path}")
    print(f"[DONE] SCALES: {scales_npz}")

"""
使用示例：
1) 预训练权重（无需 .pth）：
   python ptq4vit_qdq_export.py \
     --model vit_b16 --pretrained \
     --data-root /mnt/ino-raid4/usrs/feng/ImageNet/val \
     --num-calib 1024 --batch-size 32 --workers 8 \
     --input-size 224 --opset 13 \
     --out onnx/vit_b16_qdq_perC_sym.onnx --keep-qdq

2) 本地权重：
   python ptq4vit_qdq_export.py \
     --model vit_b16 --checkpoint /home/feng/weights/vit_b16.pth \
     --data-root /mnt/ino-raid4/usrs/feng/ImageNet/val \
     --num-calib 1024 --batch-size 32 --workers 8 \
     --input-size 224 --opset 13 \
     --out onnx/vit_b16_qdq_perC_sym.onnx --keep-qdq
"""






