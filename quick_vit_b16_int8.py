# (文件：quick_vit_b16_int8.py)
import sys
sys.path.insert(0, '.')

import os, time, subprocess, shutil, importlib, argparse
import torch
from tqdm import tqdm
import timm
import torch.nn.functional as F

import utils.datasets as datasets
import utils.net_wrap as net_wrap
from utils.quant_calib import HessianQuantCalibrator
from utils.models import get_net

from ptq4vit_qdq_export import export_qdq_onnx_from_model

DATA_ROOT = '/mnt/ino-raid4/usrs/feng/ImageNet'
MODEL_NAME = 'vit_base_patch16_224'
INPUT_SIZE = 224

def parse_args():
    p = argparse.ArgumentParser("Quick INT8 test for vit_base_patch16_224")
    p.add_argument("--test-samples", type=int, default=10000)
    p.add_argument("--calib-num", type=int, default=32)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--export-qdq", action="store_true")
    p.add_argument("--onnx-out", type=str, default="vit_ptq4vit_qdq_vit_base_patch16_224.onnx")
    p.add_argument("--trtexec", action="store_true")
    p.add_argument("--trtexec-path", type=str, default="trtexec")
    p.add_argument("--trt-plan", type=str, default="vit_ptq4vit_qdq_vit_base_patch16_224.plan")
    p.add_argument("--trt-ws", type=int, default=8192)
    p.add_argument("--bench", action="store_true")
    return p.parse_args()

@torch.inference_mode()
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
    acc = pos / tot
    print(f"Final Accuracy: {acc:.4f} ({pos}/{tot})")
    return acc

def main():
    args = parse_args()
    print(f"\n=== Quick INT8 Test: {MODEL_NAME} ===")
    print(f"Dataset: {DATA_ROOT} | Input: {INPUT_SIZE} | Batch: {args.batch}")
    print(f"Samples: test={args.test_samples}, calib={args.calib_num}\n")

    # 1) Baseline
    net_fp = timm.create_model(MODEL_NAME, pretrained=True).cuda().eval()
    g = datasets.ViTImageNetLoaderGenerator(DATA_ROOT, 'imagenet', args.batch, args.batch, 16, kwargs={"model": net_fp})
    test_loader = g.test_loader()
    max_it = args.test_samples // args.batch if args.test_samples > 0 else None
    acc_fp = test_classification(net_fp, test_loader, max_iteration=max_it, desc="Baseline-FP32")
    del net_fp; torch.cuda.empty_cache()

    # 2) PTQ4ViT 假量化 + 校准
    net = get_net(MODEL_NAME)
    quant_cfg = importlib.import_module("configs.PTQ4ViT"); importlib.reload(quant_cfg)
    wrapped = net_wrap.wrap_modules_in_net(net, quant_cfg)
    calib_loader = g.calib_loader(num=args.calib_num)

    print("Calibrating (PTQ4ViT, small set)...")
    t0 = time.time()
    quant_calibrator = HessianQuantCalibrator(net, wrapped, calib_loader, sequential=False, batch_size=4)
    quant_calibrator.batching_quant_calib()
    print(f"PTQ4ViT calib time: {(time.time()-t0)/60:.2f} min")

    acc_fake = test_classification(net, test_loader, max_iteration=max_it, desc="PTQ4ViT-fake")
    print(f"Baseline {acc_fp:.4f} -> PTQ4ViT-fake {acc_fake:.4f} (Δ={(acc_fp-acc_fake):.4f})")

    # 3) 导出 Q/DQ ONNX + scales.npz
    if args.export_qdq:
        onnx_path = args.onnx_out
        print(f"Export Q/DQ ONNX => {onnx_path}")
        onnx_path, scales_npz = export_qdq_onnx_from_model(
            model=net,
            calib_loader=calib_loader,
            wrapped_modules=wrapped,
            input_size=INPUT_SIZE,
            onnx_path=onnx_path,
            opset=13
        )
        print(f"[OK] ONNX saved: {onnx_path}")
        print(f"[OK] Scales npz : {scales_npz}")

        # 4) 可选：构建 TRT 引擎
        if args.trtexec:
            exe = shutil.which(args.trtexec_path)
            if exe is None:
                print(f"[Skip] 未找到 trtexec（{args.trtexec_path}）。已生成 ONNX，可手动构建或指定 --trtexec-path。")
            else:
                plan_path = args.trt_plan
                cmd = [
                    exe,
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
                except subprocess.CalledProcessError as e:
                    print("trtexec failed:", e)

                if args.bench and os.path.isfile(plan_path):
                    bench_cmd = [
                        exe,
                        f"--loadEngine={plan_path}",
                        f"--shapes=input:{args.batch}x3x{INPUT_SIZE}x{INPUT_SIZE}",
                        "--useCudaGraph", "--warmUp=50", "--iterations=200", "--streams=1"
                    ]
                    print("Benchmark:", " ".join(bench_cmd))
                    try:
                        subprocess.run(bench_cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        print("Benchmark failed:", e)

    # 清理
    del net, wrapped, quant_calibrator, test_loader, g, calib_loader
    torch.cuda.empty_cache()
    print("\nDone.")

if __name__ == "__main__":
    main()
