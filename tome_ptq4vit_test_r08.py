import sys
sys.path.insert(0,'.')

import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from importlib import reload, import_module
import os
import time
import timm
import tome  

import utils.datasets as datasets
import utils.net_wrap as net_wrap
from utils.quant_calib import QuantCalibrator, HessianQuantCalibrator
from utils.models import get_net
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_samples", type=int, default=50000, help="Number of test samples")
    args = parser.parse_args()
    return args

def test_classification(net, test_loader, max_iteration=None, description=None, batch_size=32):
    pos = 0
    tot = 0
    i = 0
    total_time = 0.0
    max_iteration = len(test_loader) if max_iteration is None else max_iteration
    
    net.eval()
    with torch.no_grad():
        q = tqdm(test_loader, desc=description)
        for inp, target in q:
            i += 1
            inp = inp.cuda()
            target = target.cuda()
            
            torch.cuda.synchronize()
            start_time = time.time()
            out = net(inp)
            torch.cuda.synchronize()
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # ms per batch
            total_time += inference_time
            
            pos_num = torch.sum(out.argmax(1) == target).item()
            pos += pos_num
            tot += inp.size(0)
            avg_time_ms = total_time / i if i > 0 else 0
            img_per_sec = (batch_size * 1000) / avg_time_ms if avg_time_ms > 0 else 0
            q.set_postfix({"acc": pos/tot, "img/s": img_per_sec})
            if i >= max_iteration:
                break
    
    accuracy = pos / tot
    avg_time_ms = total_time / i if i > 0 else 0
    avg_img_per_sec = (batch_size * i * 1000) / total_time if total_time > 0 else 0  # total images / total seconds
    print(f"Final Accuracy: {accuracy:.4f} ({pos}/{tot})")
    print(f"Average Throughput: {avg_img_per_sec:.2f} img/s")
    return accuracy, avg_img_per_sec

def init_config(config_name):
    """Initialize the config"""
    _, _, files = next(os.walk("./configs"))
    if config_name + ".py" in files:
        quant_cfg = import_module(f"configs.{config_name}")
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")
    reload(quant_cfg)
    return quant_cfg

def test_baseline_accuracy(name, test_samples=50000, apply_tome=False, r=16):
    """Test baseline FP32 accuracy, optionally with ToMe"""
    desc = "Baseline" if not apply_tome else f"Baseline+ToMe(r={r})"
    print(f"📊 Testing {desc} accuracy for {name}")
    
    # Load original model
    net = timm.create_model(name, pretrained=True).cuda().eval()
    
    if apply_tome and r > 0:  # Skip patch for r=0 to avoid ToMe bug
        tome.patch.timm(net)
        net.r = r
        print(f"   Applied ToMe with r={r}")
    elif apply_tome:
        print(f"   Skipped ToMe patch for r={r} (equivalent to no ToMe)")
    
    # Prepare dataset
    g = datasets.ViTImageNetLoaderGenerator('/mnt/ino-raid4/usrs/feng/ImageNet', 'imagenet', 32, 32, 16, kwargs={"model": net})
    test_loader = g.test_loader()
    
    # Calculate max iterations based on test_samples
    max_iteration = test_samples // 32 if test_samples > 0 else None
    
    # Test accuracy and speed
    accuracy, avg_throughput = test_classification(net, test_loader, max_iteration=max_iteration, description=f"{desc}-{name}")
    
    # Clean up
    del net, test_loader
    torch.cuda.empty_cache()
    
    return accuracy, avg_throughput

def test_quantized_accuracy(name, calib_size=32, config_name="PTQ4ViT", bit_setting=(8,8), test_samples=50000, apply_tome=False, r=16):
    """Test quantized model accuracy, optionally with ToMe"""
    desc = "Quantized" if not apply_tome else f"Quantized+ToMe(r={r})"
    print(f"🔧 Testing {desc} accuracy for {name} with {config_name}")
    
    # Load configuration
    quant_cfg = init_config(config_name)
    
    # Configure bit settings
    quant_cfg.bit = bit_setting
    quant_cfg.w_bit = {name: bit_setting[0] for name in quant_cfg.conv_fc_name_list}
    quant_cfg.a_bit = {name: bit_setting[1] for name in quant_cfg.conv_fc_name_list}
    quant_cfg.A_bit = {name: bit_setting[1] for name in quant_cfg.matmul_name_list}
    quant_cfg.B_bit = {name: bit_setting[1] for name in quant_cfg.matmul_name_list}

    # Load model
    net = get_net(name)

    if apply_tome and r > 0:  # Skip patch for r=0 to avoid ToMe bug
        tome.patch.timm(net)
        net.r = r
        print(f"   Applied ToMe with r={r} before quantization")
    elif apply_tome:
        print(f"   Skipped ToMe patch for r={r} (equivalent to no ToMe)")

    # Wrap quantization layers
    wrapped_modules = net_wrap.wrap_modules_in_net(net, quant_cfg)
    
    # Prepare dataset
    g = datasets.ViTImageNetLoaderGenerator('/mnt/ino-raid4/usrs/feng/ImageNet', 'imagenet', 32, 32, 16, kwargs={"model": net})
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
    
    # Test accuracy and speed
    accuracy, avg_throughput = test_classification(net, test_loader, max_iteration=max_iteration, description=f"{desc}-{name}")
    
    # Clean up
    del net, wrapped_modules, test_loader, calib_loader, quant_calibrator
    torch.cuda.empty_cache()
    
    return accuracy, avg_throughput, calib_time

def main():
    args = parse_args()
    
    print("🎯 PTQ4ViT with ToMe Test (ViT-B/224, r=0 and r=8)")
    print(f"📁 Dataset path: /mnt/ino-raid4/usrs/feng/ImageNet")
    print(f"🔢 Test samples: {args.test_samples}")
    print(f"{'='*60}")
    
    model_name = "vit_base_patch16_224"
    
    # Test baseline (FP32, no ToMe)
    baseline_acc, baseline_throughput = test_baseline_accuracy(model_name, args.test_samples, apply_tome=False)
    
    # Test quantized (PTQ4ViT W8A8, no ToMe)
    quantized_acc, quantized_throughput, calib_time_q = test_quantized_accuracy(
        model_name, calib_size=32, config_name="PTQ4ViT", bit_setting=(8,8),
        test_samples=args.test_samples, apply_tome=False
    )
    
    results = []
    
    for r in [0, 8]:  # Test r=0 and r=8
        print(f"\n🔄 Testing with r={r}")
        
        # Test baseline + ToMe (FP32 + ToMe)
        tome_acc, tome_throughput = test_baseline_accuracy(model_name, args.test_samples, apply_tome=True, r=r)
        
        # Test quantized + ToMe (ToMe + PTQ4ViT W8A8)
        tome_quantized_acc, tome_quantized_throughput, calib_time_tq = test_quantized_accuracy(
            model_name, calib_size=32, config_name="PTQ4ViT", bit_setting=(8,8),
            test_samples=args.test_samples, apply_tome=True, r=r
        )
        
        results.append({
            'r': r,
            'tome_acc': tome_acc,
            'tome_throughput': tome_throughput,
            'tome_quantized_acc': tome_quantized_acc,
            'tome_quantized_throughput': tome_quantized_throughput,
            'calib_time_tq': calib_time_tq
        })
    
    # Print results
    print(f"\n{'='*100}")
    print("📊 RESULTS FOR ViT-B/224")
    print(f"{'='*100}")
    print(f"{'Variant':<25} {'Accuracy':<10} {'Throughput (img/s)':<20} {'Calib Time (min)':<15}")
    print(f"{'-'*25} {'-'*10} {'-'*20} {'-'*15}")
    print(f"{'Baseline (no ToMe)':<25} {baseline_acc:<10.4f} {baseline_throughput:<20.2f} {'N/A':<15}")
    print(f"{'Quantized (no ToMe)':<25} {quantized_acc:<10.4f} {quantized_throughput:<20.2f} {calib_time_q:<15.2f}")
    for res in results:
        print(f"{'-'*70}")
        print(f"{'Baseline+ToMe (r=' + str(res['r']) + ')':<25} {res['tome_acc']:<10.4f} {res['tome_throughput']:<20.2f} {'N/A':<15}")
        print(f"{'Quantized+ToMe (r=' + str(res['r']) + ')':<25} {res['tome_quantized_acc']:<10.4f} {res['tome_quantized_throughput']:<20.2f} {res['calib_time_tq']:<15.2f}")
    print(f"{'='*100}")
    
    print("\n🎉 Test completed!")

if __name__ == '__main__':
    main()