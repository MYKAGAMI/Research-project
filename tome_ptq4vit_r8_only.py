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
import tome  # Import ToMe
import types  # For method replacement

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

def vit_attn_forward_with_tome(self, x, size=None):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    metric = attn.mean(1)

    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, metric

def test_quantized_tome_accuracy(name, calib_size=32, config_name="PTQ4ViT", bit_setting=(8,8), test_samples=50000, r=8):
    """Test quantized + ToMe (r=8) accuracy"""
    desc = f"Quantized+ToMe(r={r})"
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

    # Apply ToMe with r=8
    tome.patch.timm(net)
    net.r = r
    print(f"   Applied ToMe with r={r} before quantization")

    # Save FP32 model before quantization
    torch.save(net.state_dict(), 'fp32_tome_r8_vit_base.pth')
    print("FP32 model saved to 'fp32_tome_r8_vit_base.pth'")

    # Wrap quantization layers
    wrapped_modules = net_wrap.wrap_modules_in_net(net, quant_cfg)

    # Override attn forward to be compatible with both quantization and ToMe
    for block in net.blocks:
        block.attn.forward = types.MethodType(vit_attn_forward_with_tome, block.attn)
    print("   Overridden attn forward for ToMe compatibility")
    
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
    
    # Save quantized model after calibration
    torch.save(net.state_dict(), 'quantized_tome_r8_vit_base.pth')
    print("Quantized model saved to 'quantized_tome_r8_vit_base.pth'")

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
    
    print("🎯 PTQ4ViT + ToMe (r=8 only) Test for ViT-B/224")
    print(f"📁 Dataset path: /mnt/ino-raid4/usrs/feng/ImageNet")
    print(f"🔢 Test samples: {args.test_samples}")
    print(f"{'='*60}")
    
    model_name = "vit_base_patch16_224"
    
    # Only test Quantized + ToMe (r=8)
    tome_quantized_acc, tome_quantized_throughput, calib_time_tq = test_quantized_tome_accuracy(
        model_name, calib_size=32, config_name="PTQ4ViT", bit_setting=(8,8),
        test_samples=args.test_samples, r=8
    )
    
    # Print results
    print(f"\n{'='*100}")
    print("📊 RESULTS FOR Quantized+ToMe (r=8) on ViT-B/224")
    print(f"{'='*100}")
    print(f"{'Accuracy':<10} {'Throughput (img/s)':<20} {'Calib Time (min)':<15}")
    print(f"{'-'*10} {'-'*20} {'-'*15}")
    print(f"{tome_quantized_acc:<10.4f} {tome_quantized_throughput:<20.2f} {calib_time_tq:<15.2f}")
    print(f"{'='*100}")
    
    print("\n🎉 Test completed!")

if __name__ == '__main__':
    main()