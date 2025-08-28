import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from importlib import reload, import_module
import multiprocessing
import os
import time
from itertools import product
import timm
import utils.datasets as datasets
import utils.net_wrap as net_wrap
from utils.quant_calib import QuantCalibrator, HessianQuantCalibrator
from utils.models import get_net
import torch.nn.functional as F

# ==========  新增：Torch-TensorRT 相关 ==========
_TRTPRESENT = True
try:
    import torch_tensorrt
    from torch_tensorrt import ptq as trt_ptq
except Exception:
    _TRTPRESENT = False
# ==============================================

DATA_ROOT = '/mnt/ino-raid4/usrs/feng/ImageNet'  # 你的数据根目录

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--multiprocess", action='store_true')
    parser.add_argument("--test_samples", type=int, default=50000, help="Number of test samples")

    # ===== 新增：真实 INT8 (TensorRT) 相关开关与参数 =====
    parser.add_argument("--trt-int8", action="store_true",
                        help="为每个模型构建 TensorRT INT8 引擎并用其评估精度（真实低比特执行）")
    parser.add_argument("--calib-dir", type=str, default=os.path.join(DATA_ROOT, 'val'),
                        help="用于 PTQ 校准的数据目录（建议用 val）")
    parser.add_argument("--calib-samples", type=int, default=512,
                        help="用于校准的图像数量（从目录前N张）")
    parser.add_argument("--calib-batch", type=int, default=32,
                        help="校准 batch 大小")
    parser.add_argument("--calib-size", type=int, default=None,
                        help="校准输入分辨率（默认自动根据模型 224/384）")
    parser.add_argument("--trt-precisions", type=str, default="fp16,int8",
                        help="允许的精度(逗号分隔): 例如 fp16,int8 / fp32,int8")
    parser.add_argument("--engine-dir", type=str, default="./trt_engines",
                        help="保存 .ts 引擎的目录")
    # ====================================================

    args = parser.parse_args()
    return args

def test_classification(net, test_loader, max_iteration=None, description=None):
    pos = 0
    tot = 0
    i = 0
    max_iteration = len(test_loader) if max_iteration is None else max_iteration

    net.eval()
    with torch.no_grad():
        q = tqdm(test_loader, desc=description)
        for inp, target in q:
            i += 1
            inp = inp.cuda()
            target = target.cuda()
            out = net(inp)
            pos_num = torch.sum(out.argmax(1) == target).item()
            pos += pos_num
            tot += inp.size(0)
            q.set_postfix({"acc": pos/tot})
            if i >= max_iteration:
                break

    accuracy = pos/tot
    print(f"Final Accuracy: {accuracy:.4f} ({pos}/{tot})")
    return accuracy

# ===== 新增：使用 TensorRT 引擎做分类评估（真实 INT8 路径） =====
@torch.inference_mode()
def test_classification_trt(engine_path, test_loader, max_iteration=None, description=None, use_half_input=True):
    if not os.path.isfile(engine_path):
        raise FileNotFoundError(f"TensorRT 引擎不存在: {engine_path}")
    trt_mod = torch.jit.load(engine_path, map_location="cuda").eval()

    pos = 0
    tot = 0
    i = 0
    max_iteration = len(test_loader) if max_iteration is None else max_iteration

    q = tqdm(test_loader, desc=(description or "TRT-INT8"))
    for inp, target in q:
        i += 1
        inp = inp.cuda()
        if use_half_input:
            inp = inp.half()
        target = target.cuda()
        out = trt_mod(inp)
        pos_num = torch.sum(out.argmax(1) == target).item()
        pos += pos_num
        tot += inp.size(0)
        q.set_postfix({"acc": pos/tot})
        if i >= max_iteration:
            break

    acc = pos / tot
    print(f"[TRT] Final Accuracy: {acc:.4f} ({pos}/{tot})")
    return acc
# ========================================================

def init_config(config_name):
    """Initialize the config"""
    _, _, files = next(os.walk("./configs"))
    if config_name + ".py" in files:
        quant_cfg = import_module(f"configs.{config_name}")
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")
    reload(quant_cfg)
    return quant_cfg

def _get_model_input_res(name: str) -> int:
    """根据 timm 默认配置自动推断输入分辨率（224/384）"""
    m = timm.create_model(name, pretrained=True)
    sz = 224
    if hasattr(m, "default_cfg") and "input_size" in m.default_cfg:
        inp = m.default_cfg["input_size"]
        if isinstance(inp, (list, tuple)) and len(inp) == 3:
            sz = int(inp[1])  # (3, H, W) => H
    del m
    return sz

def test_baseline_accuracy(name, test_samples=5000):
    """Test baseline FP32 accuracy"""
    print(f"📊 Testing baseline accuracy for {name}")

    # Load original model
    net = timm.create_model(name, pretrained=True).cuda().eval()

    # Prepare dataset
    g = datasets.ViTImageNetLoaderGenerator(DATA_ROOT, 'imagenet', 32, 32, 16, kwargs={"model": net})
    test_loader = g.test_loader()

    # Calculate max iterations based on test_samples
    max_iteration = test_samples // 32 if test_samples > 0 else None

    # Test accuracy
    accuracy = test_classification(net, test_loader, max_iteration=max_iteration, description=f"Baseline-{name}")

    # Clean up
    del net, test_loader
    torch.cuda.empty_cache()

    return accuracy

def test_quantized_accuracy(name, calib_size=32, config_name="PTQ4ViT", bit_setting=(8,8), test_samples=5000):
    """Test quantized model accuracy (假量化路径，用于和 TRT 对比精度)"""
    print(f"🔧 Testing quantized accuracy for {name} with {config_name}")

    # Load configuration
    quant_cfg = init_config(config_name)

    # Configure bit settings
    quant_cfg.bit = bit_setting
    quant_cfg.w_bit = {n: bit_setting[0] for n in quant_cfg.conv_fc_name_list}
    quant_cfg.a_bit = {n: bit_setting[1] for n in quant_cfg.conv_fc_name_list}
    quant_cfg.A_bit = {n: bit_setting[1] for n in quant_cfg.matmul_name_list}
    quant_cfg.B_bit = {n: bit_setting[1] for n in quant_cfg.matmul_name_list}

    # Load model
    net = get_net(name)

    # Wrap quantization layers
    wrapped_modules = net_wrap.wrap_modules_in_net(net, quant_cfg)

    # Prepare dataset
    g = datasets.ViTImageNetLoaderGenerator(DATA_ROOT, 'imagenet', 32, 32, 16, kwargs={"model": net})
    test_loader = g.test_loader()
    calib_loader = g.calib_loader(num=calib_size)

    # Quantization calibration
    print(f"   Calibrating with {calib_size} samples...")
    calib_start_time = time.time()
    quant_calibrator = HessianQuantCalibrator(net, wrapped_modules, calib_loader, sequential=False, batch_size=4)
    quant_calibrator.batching_quant_calib()
    calib_end_time = time.time()
    calib_time = (calib_end_time - calib_start_time) / 60

    # Calculate max iterations based on test_samples
    max_iteration = test_samples // 32 if test_samples > 0 else None

    # Test accuracy
    accuracy = test_classification(net, test_loader, max_iteration=max_iteration, description=f"Quantized-{name}")

    # Clean up
    del net, wrapped_modules, test_loader, calib_loader, quant_calibrator
    torch.cuda.empty_cache()

    return accuracy, calib_time

# ===== 新增：构建 TRT INT8 引擎（真实加速） =====
def _parse_precisions(spec: str):
    mapping = {
        "fp16": torch.half, "half": torch.half, "float16": torch.half,
        "int8": torch.int8,
        "fp32": torch.float32, "float32": torch.float32,
        "bf16": torch.bfloat16,
    }
    out = set()
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if tok in mapping:
            out.add(mapping[tok])
    if not out:
        out = {torch.half, torch.int8}
    return out

def _make_calib_loader(imagenet_dir: str, n_imgs: int, bs: int, size: int):
    import torchvision as tv
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds  = tv.datasets.ImageFolder(imagenet_dir, transform=tfm)
    if (n_imgs is not None) and (len(ds.samples) > n_imgs):
        ds.samples = ds.samples[:n_imgs]
        if hasattr(ds, "targets") and len(ds.targets) > n_imgs:
            ds.targets = ds.targets[:n_imgs]
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

def build_trt_int8_engine(model_name: str,
                          input_size: int,
                          calib_dir: str,
                          calib_samples: int,
                          calib_batch: int,
                          precisions: str,
                          engine_dir: str) -> str:
    """
    用 Torch-TensorRT 对 timm 预训练模型进行 INT8 PTQ 编译，并保存 TorchScript 引擎 (.ts)
    返回引擎路径
    """
    assert _TRTPRESENT, "未检测到 torch-tensorrt，请先 `pip install torch-tensorrt`"
    os.makedirs(engine_dir, exist_ok=True)

    model = timm.create_model(model_name, pretrained=True).eval().to("cuda")
    enabled_precisions = _parse_precisions(precisions)

    # 校准数据
    calib_loader = _make_calib_loader(calib_dir, calib_samples, calib_batch, input_size)

    # Torch-TensorRT 的数据加载校准器
    calibrator = trt_ptq.DataLoaderCalibrator(
        calib_loader,
        cache_file=os.path.join(engine_dir, f"{model_name}_{input_size}_calib.cache"),
        use_cache=False,
        algo_type=trt_ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
        device=torch.device("cuda:0"),
    )

    # 编译（允许回退不支持算子）
    print(f"[TRT] Compiling INT8 engine for {model_name} @ {input_size} ...")
    trt_mod = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input((1,3,input_size,input_size))],
        enabled_precisions=enabled_precisions,      # {torch.half, torch.int8}
        calibrator=calibrator,
        require_full_compilation=False,
        device={"device_type": torch_tensorrt.DeviceType.GPU, "gpu_id": 0},
    )

    out_path = os.path.join(engine_dir, f"{model_name}_{input_size}_int8.ts")
    torch.jit.save(trt_mod, out_path)
    del model, trt_mod
    torch.cuda.empty_cache()
    print(f"[TRT] Saved engine: {out_path}")
    return out_path

def test_single_model(model_name, test_samples=5000, args=None):
    """Test baseline, quantized(PTQ4ViT-fake), and optional TRT-INT8 real accuracy"""
    print(f"\n{'='*60}")
    print(f"🚀 Testing Model: {model_name}")
    print(f"{'='*60}")

    results = {
        'model': model_name,
        'baseline_acc': None,
        'quantized_acc': None,
        'acc_drop': None,
        'acc_drop_percent': None,
        'calib_time': None,
        'trt_int8_acc': None,
    }

    try:
        # 1) Baseline
        baseline_acc = test_baseline_accuracy(model_name, test_samples)
        results['baseline_acc'] = baseline_acc

        # 2) PTQ4ViT 假量化（用于对比）
        quantized_acc, calib_time = test_quantized_accuracy(
            model_name,
            calib_size=32,
            config_name="PTQ4ViT",
            bit_setting=(8,8),
            test_samples=test_samples
        )
        results['quantized_acc'] = quantized_acc
        results['calib_time'] = calib_time
        results['acc_drop'] = baseline_acc - quantized_acc
        results['acc_drop_percent'] = (results['acc_drop'] / baseline_acc) * 100 if baseline_acc else None

        print(f"✅ {model_name} completed (FP32 vs PTQ4ViT-fake):")
        print(f"   Baseline : {baseline_acc:.4f}")
        print(f"   Quantized: {quantized_acc:.4f}")
        print(f"   Drop     : {results['acc_drop']:.4f} ({results['acc_drop_percent']:.2f}%)")
        print(f"   Calib t  : {calib_time:.2f} min")

        # 3) （可选）真实 INT8：TensorRT 引擎 + 精度评估
        if args is not None and args.trt_int8:
            if not _TRTPRESENT:
                print("⚠️ 未安装 torch-tensorrt，跳过 TRT INT8。请先 `pip install torch-tensorrt`")
            else:
                input_size = _get_model_input_res(model_name)
                if args.calib_size is not None:
                    input_size = int(args.calib_size)  # 允许手动覆盖

                engine_path = build_trt_int8_engine(
                    model_name=model_name,
                    input_size=input_size,
                    calib_dir=args.calib_dir,
                    calib_samples=args.calib_samples,
                    calib_batch=args.calib_batch,
                    precisions=args.trt_precisions,
                    engine_dir=args.engine_dir
                )

                # 用与 baseline 相同的 test_loader 做精度评估
                net_tmp = timm.create_model(model_name, pretrained=True).cuda().eval()
                g = datasets.ViTImageNetLoaderGenerator(DATA_ROOT, 'imagenet', 32, 32, 16, kwargs={"model": net_tmp})
                test_loader = g.test_loader()
                max_iteration = test_samples // 32 if test_samples > 0 else None

                trt_acc = test_classification_trt(
                    engine_path,
                    test_loader,
                    max_iteration=max_iteration,
                    description=f"TRT-INT8-{model_name}",
                    use_half_input=('fp16' in args.trt_precisions.lower() or 'half' in args.trt_precisions.lower())
                )
                results['trt_int8_acc'] = trt_acc
                print(f"🔥 TRT-INT8 accuracy: {trt_acc:.4f}")

                del net_tmp, test_loader
                torch.cuda.empty_cache()

        return results

    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return results

def print_results_table(results):
    """Print results in a formatted table"""
    print(f"\n{'='*110}")
    print("📊 ACCURACY COMPARISON RESULTS (FP32 vs PTQ4ViT-fake vs TRT-INT8)")
    print(f"{'='*110}")

    # Table header
    print(f"{'Model':<26} {'Baseline':<10} {'PTQ4ViT':<10} {'Drop':<9} {'Drop%':<9} {'TRT-INT8':<10} {'Time(min)':<10}")
    print(f"{'-'*26} {'-'*10} {'-'*10} {'-'*9} {'-'*9} {'-'*10} {'-'*10}")

    # Table rows
    for result in results:
        if result is not None:
            print(f"{result['model']:<26} "
                  f"{(result['baseline_acc'] or 0):<10.4f} "
                  f"{(result['quantized_acc'] or 0):<10.4f} "
                  f"{(result['acc_drop'] or 0):<9.4f} "
                  f"{(result['acc_drop_percent'] or 0):<9.2f} "
                  f"{(result.get('trt_int8_acc') or 0):<10.4f} "
                  f"{(result['calib_time'] or 0):<10.2f}")

    print(f"{'-'*110}")

    # Calculate averages (仅对存在的结果求平均)
    valid_results = [r for r in results if r is not None and r['baseline_acc'] is not None]
    if valid_results:
        n = len(valid_results)
        avg_baseline = sum((r['baseline_acc'] or 0) for r in valid_results) / n
        avg_quantized = sum((r['quantized_acc'] or 0) for r in valid_results) / n
        avg_drop = sum((r['acc_drop'] or 0) for r in valid_results) / n
        avg_drop_percent = sum((r['acc_drop_percent'] or 0) for r in valid_results) / n
        # TRT 可能未启用
        trt_list = [r.get('trt_int8_acc') for r in valid_results if r.get('trt_int8_acc') is not None]
        avg_trt = (sum(trt_list) / len(trt_list)) if trt_list else 0.0
        avg_time = sum((r['calib_time'] or 0) for r in valid_results) / n

        print(f"{'Average':<26} "
              f"{avg_baseline:<10.4f} "
              f"{avg_quantized:<10.4f} "
              f"{avg_drop:<9.4f} "
              f"{avg_drop_percent:<9.2f} "
              f"{avg_trt:<10.4f} "
              f"{avg_time:<10.2f}")

    print(f"{'='*110}")

def main():
    args = parse_args()

    print("🎯 PTQ4ViT Accuracy & Real INT8 (TRT) Test")
    print(f"📁 Dataset path: {DATA_ROOT}")
    print(f"🔢 Test samples per model: {args.test_samples}")
    print(f"⚙️  TRT INT8: {'ON' if args.trt_int8 else 'OFF'}")
    if args.trt_int8 and not _TRTPRESENT:
        print("⚠️ 未检测到 torch-tensorrt，将无法生成真实 INT8 引擎。请先 `pip install torch-tensorrt`")
    print(f"{'='*60}")

    # 6 specific models to test
    model_names = [
        "vit_small_patch16_224",     # ViT-S/224
        "vit_base_patch16_224",      # ViT-B/224
        "vit_base_patch16_384",      # ViT-B/384
        "deit_small_patch16_224",    # DeiT-S/224
        "deit_base_patch16_224",     # DeiT-B/224
        "deit_base_patch16_384",     # DeiT-B/384
    ]

    print(f"📋 Testing {len(model_names)} models:")
    for i, name in enumerate(model_names, 1):
        print(f"   {i}. {name}")
    print()

    # Test all models
    results = []
    for i, model_name in enumerate(model_names, 1):
        print(f"\n🔄 Progress: {i}/{len(model_names)}")
        res = test_single_model(model_name, args.test_samples, args=args)
        results.append(res)

        # Print intermediate results
        if res and res['baseline_acc'] is not None and res['quantized_acc'] is not None:
            print(f"✅ {model_name}: Baseline {res['baseline_acc']:.4f} → PTQ4ViT {res['quantized_acc']:.4f} "
                  f"(Drop: {res['acc_drop_percent']:.2f}%) | TRT-INT8: {res.get('trt_int8_acc')}")

    # Print final results table
    print_results_table(results)

    print("\n🎉 All tests completed!")

if __name__ == '__main__':
    main()
