# test_6_models_ptq4vit_qdq.py
# A variant of your test script that can export REAL Q/DQ ONNX from PTQ4ViT-calibrated model,
# then (optionally) build a TensorRT engine via trtexec.

import sys
sys.path.insert(0,'.')

import torch
from tqdm import tqdm
import argparse
from importlib import reload, import_module
import os, time, subprocess
import timm
import utils.datasets as datasets
import utils.net_wrap as net_wrap
from utils.quant_calib import HessianQuantCalibrator
from utils.models import get_net
import torch.nn.functional as F

from ptq4vit_qdq_export import export_qdq_onnx_from_model

DATA_ROOT = '/mnt/ino-raid4/usrs/feng/ImageNet'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_samples", type=int, default=50000)
    p.add_argument("--export-qdq", action="store_true", help="导出 Q/DQ ONNX（真实量化图），基于 PTQ4ViT 校准结果/或回退统计")
    p.add_argument("--onnx-out", type=str, default="vit_ptq4vit_qdq.onnx")
    p.add_argument("--trtexec", action="store_true", help="使用 trtexec 构建 TensorRT 引擎（需要已安装 TensorRT）")
    p.add_argument("--trt-plan", type=str, default="vit_ptq4vit_qdq.plan")
    p.add_argument("--trt-ws", type=int, default=4096, help="TensorRT workspace (MB)")
    p.add_argument("--input-size", type=int, default=None, help="手动指定输入分辨率（默认自动根据模型）")
    return p.parse_args()

def test_classification(net, test_loader, max_iteration=None, desc=None):
    pos = tot = i = 0
    max_iteration = len(test_loader) if max_iteration is None else max_iteration
    net.eval()
    with torch.no_grad():
        q = tqdm(test_loader, desc=desc)
        for inp, tgt in q:
            i += 1
            inp, tgt = inp.cuda(), tgt.cuda()
            out = net(inp)
            pos += (out.argmax(1) == tgt).sum().item()
            tot += inp.size(0)
            q.set_postfix({"acc": pos/tot})
            if i >= max_iteration: break
    acc = pos/tot
    print(f"Final Accuracy: {acc:.4f} ({pos}/{tot})")
    return acc

def _get_model_input_res(name: str) -> int:
    m = timm.create_model(name, pretrained=True)
    sz = 224
    if hasattr(m, "default_cfg") and "input_size" in m.default_cfg:
        inp = m.default_cfg["input_size"]
        if isinstance(inp, (list, tuple)) and len(inp) == 3:
            sz = int(inp[1])
    del m
    return sz

def main():
    args = parse_args()
    model_names = [
        "vit_small_patch16_224",
        "vit_base_patch16_224",
        "vit_base_patch16_384",
        "deit_small_patch16_224",
        "deit_base_patch16_224",
        "deit_base_patch16_384",
    ]

    for name in model_names:
        print("\n" + "="*60)
        print("Model:", name)
        print("="*60)

        # 1) baseline
        net_fp = timm.create_model(name, pretrained=True).cuda().eval()
        g = datasets.ViTImageNetLoaderGenerator(DATA_ROOT, 'imagenet', 32, 32, 16, kwargs={"model": net_fp})
        test_loader = g.test_loader()
        max_it = args.test_samples // 32 if args.test_samples > 0 else None
        acc_fp = test_classification(net_fp, test_loader, max_iteration=max_it, desc=f"Baseline-{name}")
        del net_fp; torch.cuda.empty_cache()

        # 2) PTQ4ViT calibration (fake-quant path, for comparing accuracy only)
        net = get_net(name)
        quant_cfg = import_module("configs.PTQ4ViT"); reload(quant_cfg)
        wrapped = net_wrap.wrap_modules_in_net(net, quant_cfg)
        calib_loader = g.calib_loader(num=32)

        print("Calibrating (PTQ4ViT)...")
        cal_start = time.time()
        quant_calibrator = HessianQuantCalibrator(net, wrapped, calib_loader, sequential=False, batch_size=4)
        quant_calibrator.batching_quant_calib()
        print(f"PTQ4ViT calib time: {(time.time()-cal_start)/60:.2f} min")

        acc_fake = test_classification(net, test_loader, max_iteration=max_it, desc=f"PTQ4ViT-fake-{name}")
        print(f"Baseline {acc_fp:.4f} -> PTQ4ViT-fake {acc_fake:.4f} (Δ={(acc_fp-acc_fake):.4f})")

        # 3) (Optional) Export REAL Q/DQ ONNX using PTQ4ViT ranges (or fallback)
        if args.export_qdq:
            input_size = args.input_size or _get_model_input_res(name)
            onnx_path = args.onnx_out.replace(".onnx", f"_{name}_{input_size}.onnx")
            print(f"Export Q/DQ ONNX => {onnx_path}")
            onnx_path = export_qdq_onnx_from_model(
                model=net, calib_loader=calib_loader, wrapped_modules=wrapped,
                input_size=input_size, onnx_path=onnx_path, opset=13
            )
            print(f"[OK] ONNX saved: {onnx_path}")

            if args.trtexec:
                plan_path = args.trt_plan.replace(".plan", f"_{name}_{input_size}.plan")
                cmd = [
                    "trtexec",
                    f"--onnx={onnx_path}",
                    "--int8",
                    "--fp16",
                    f"--saveEngine={plan_path}",
                    f"--workspace={args.trt_ws}"
                ]
                print("Running:", " ".join(cmd))
                try:
                    subprocess.run(cmd, check=True)
                    print(f"[OK] TRT engine saved: {plan_path}")
                except Exception as e:
                    print("trtexec failed:", e)

        # cleanup
        del net, wrapped, quant_calibrator, test_loader, g, calib_loader
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
