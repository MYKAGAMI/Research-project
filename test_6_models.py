import sys
sys.path.insert(0,'.')

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--multiprocess", action='store_true')
    parser.add_argument("--test_samples", type=int, default=50000, help="Number of test samples")
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

def init_config(config_name):
    """Initialize the config"""
    _, _, files = next(os.walk("./configs"))
    if config_name + ".py" in files:
        quant_cfg = import_module(f"configs.{config_name}")
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")
    reload(quant_cfg)
    return quant_cfg

def test_baseline_accuracy(name, test_samples=5000):
    """Test baseline FP32 accuracy"""
    print(f"📊 Testing baseline accuracy for {name}")
    
    # Load original model
    net = timm.create_model(name, pretrained=True).cuda().eval()
    
    # Prepare dataset
    g = datasets.ViTImageNetLoaderGenerator('/mnt/ino-raid4/usrs/feng/ImageNet', 'imagenet', 32, 32, 16, kwargs={"model": net})
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
    """Test quantized model accuracy"""
    print(f"🔧 Testing quantized accuracy for {name} with {config_name}")
    
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
    
    # Test accuracy
    accuracy = test_classification(net, test_loader, max_iteration=max_iteration, description=f"Quantized-{name}")
    
    # Clean up
    del net, wrapped_modules, test_loader, calib_loader, quant_calibrator
    torch.cuda.empty_cache()
    
    return accuracy, calib_time

def test_single_model(model_name, test_samples=5000):
    """Test both baseline and quantized accuracy for a single model"""
    print(f"\n{'='*60}")
    print(f"🚀 Testing Model: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Test baseline accuracy
        baseline_acc = test_baseline_accuracy(model_name, test_samples)
        
        # Test quantized accuracy (PTQ4ViT W8A8)
        quantized_acc, calib_time = test_quantized_accuracy(
            model_name, 
            calib_size=32, 
            config_name="PTQ4ViT", 
            bit_setting=(8,8),
            test_samples=test_samples
        )
        
        # Calculate accuracy drop
        acc_drop = baseline_acc - quantized_acc
        acc_drop_percent = (acc_drop / baseline_acc) * 100
        
        result = {
            'model': model_name,
            'baseline_acc': baseline_acc,
            'quantized_acc': quantized_acc,
            'acc_drop': acc_drop,
            'acc_drop_percent': acc_drop_percent,
            'calib_time': calib_time
        }
        
        print(f"✅ {model_name} completed:")
        print(f"   Baseline: {baseline_acc:.4f}")
        print(f"   Quantized: {quantized_acc:.4f}")
        print(f"   Drop: {acc_drop:.4f} ({acc_drop_percent:.2f}%)")
        print(f"   Calibration time: {calib_time:.2f} min")
        
        return result
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_results_table(results):
    """Print results in a formatted table"""
    print(f"\n{'='*80}")
    print("📊 ACCURACY COMPARISON RESULTS")
    print(f"{'='*80}")
    
    # Table header
    print(f"{'Model':<20} {'Baseline':<10} {'PTQ4ViT':<10} {'Drop':<8} {'Drop%':<8} {'Time(min)':<10}")
    print(f"{'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
    
    # Table rows
    for result in results:
        if result is not None:
            print(f"{result['model']:<20} "
                  f"{result['baseline_acc']:<10.4f} "
                  f"{result['quantized_acc']:<10.4f} "
                  f"{result['acc_drop']:<8.4f} "
                  f"{result['acc_drop_percent']:<8.2f} "
                  f"{result['calib_time']:<10.2f}")
    
    print(f"{'-'*80}")
    
    # Calculate averages
    valid_results = [r for r in results if r is not None]
    if valid_results:
        avg_baseline = sum(r['baseline_acc'] for r in valid_results) / len(valid_results)
        avg_quantized = sum(r['quantized_acc'] for r in valid_results) / len(valid_results)
        avg_drop = sum(r['acc_drop'] for r in valid_results) / len(valid_results)
        avg_drop_percent = sum(r['acc_drop_percent'] for r in valid_results) / len(valid_results)
        avg_time = sum(r['calib_time'] for r in valid_results) / len(valid_results)
        
        print(f"{'Average':<20} "
              f"{avg_baseline:<10.4f} "
              f"{avg_quantized:<10.4f} "
              f"{avg_drop:<8.4f} "
              f"{avg_drop_percent:<8.2f} "
              f"{avg_time:<10.2f}")
    
    print(f"{'='*80}")

def main():
    args = parse_args()
    
    print("🎯 PTQ4ViT Accuracy Comparison Test")
    print(f"📁 Dataset path: /mnt/ino-raid4/usrs/feng/ImageNet")
    print(f"🔢 Test samples per model: {args.test_samples}")
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
        result = test_single_model(model_name, args.test_samples)
        results.append(result)
        
        # Print intermediate results
        if result:
            print(f"✅ {model_name}: Baseline {result['baseline_acc']:.4f} → Quantized {result['quantized_acc']:.4f} (Drop: {result['acc_drop_percent']:.2f}%)")
    
    # Print final results table
    print_results_table(results)
    
    print("\n🎉 All tests completed!")

if __name__ == '__main__':
    main()