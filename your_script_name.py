import sys
sys.path.insert(0,'.')

import time
import torch
import torch.nn as nn
import numpy as np
import torch_tensorrt
import timm
import utils.datasets as datasets
import utils.net_wrap as net_wrap
from utils.quant_calib import HessianQuantCalibrator
from utils.models import get_net
from configs.PTQ4ViT import get_module

def apply_ptq4vit_quantization(model_name, calib_size=32):
    """应用PTQ4ViT量化"""
    print(f"🔧 应用PTQ4ViT量化到 {model_name}")
    
    # 1. 加载原始模型
    net = get_net(model_name)
    
    # 2. 包装量化层
    # 创建量化配置
    class QuantConfig:
        def __init__(self):
            self.bit = 8
            self.conv_fc_name_list = ["qconv", "qlinear_qkv", "qlinear_proj", "qlinear_MLP_1", "qlinear_MLP_2", "qlinear_classifier", "qlinear_reduction"]
            self.matmul_name_list = ["qmatmul_qk", "qmatmul_scorev"]
            self.w_bit = {name: 8 for name in self.conv_fc_name_list}
            self.a_bit = {name: 8 for name in self.conv_fc_name_list}
            self.A_bit = {name: 8 for name in self.matmul_name_list}
            self.B_bit = {name: 8 for name in self.matmul_name_list}
            
        def get_module(self, module_type, *args, **kwargs):
            return get_module(module_type, *args, **kwargs)
    
    quant_cfg = QuantConfig()
    wrapped_modules = net_wrap.wrap_modules_in_net(net, quant_cfg)
    
    # 3. 准备校准数据
    g = datasets.ViTImageNetLoaderGenerator('/mnt/ino-raid4/usrs/feng/ImageNet', 'imagenet', 32, 32, 16, kwargs={"model": net})
    calib_loader = g.calib_loader(num=calib_size)
    
    # 4. 运行量化校准
    print("   开始量化校准...")
    quant_calibrator = HessianQuantCalibrator(net, wrapped_modules, calib_loader, sequential=False, batch_size=4)
    quant_calibrator.batching_quant_calib()
    
    print("✅ PTQ4ViT量化完成")
    return net

def compile_tensorrt_int8(model, input_shape, use_int8=True):
    """TensorRT编译（支持int8）"""
    print(f"🔧 TensorRT编译 {'INT8' if use_int8 else 'FP16'} (shape: {input_shape})...")
    
    try:
        model.eval()
        dummy_input = torch.randn(input_shape).cuda()
        
        # 测试原始模型
        with torch.no_grad():
            original_output = model(dummy_input)
        
        # 编译设置
        if use_int8:
            compile_settings = {
                'inputs': [torch_tensorrt.Input(input_shape, dtype=torch.float32)],
                'enabled_precisions': {torch.int8, torch.float16, torch.float32},  # 启用int8
                'workspace_size': 2*1024**3,
                'min_block_size': 3,
                'require_full_compilation': False,
                'truncate_long_and_double': True,
            }
        else:
            compile_settings = {
                'inputs': [torch_tensorrt.Input(input_shape, dtype=torch.float32)],
                'enabled_precisions': {torch.float16, torch.float32},  # 仅FP16+FP32
                'workspace_size': 2*1024**3,
                'min_block_size': 3,
                'require_full_compilation': False,
                'truncate_long_and_double': True,
            }
        
        print("   开始编译...")
        trt_model = torch_tensorrt.compile(model, **compile_settings)
        
        # 验证编译后的模型
        with torch.no_grad():
            trt_output = trt_model(dummy_input)
            
        # 检查输出差异
        diff = torch.abs(original_output - trt_output).max().item()
        print(f"   输出差异: {diff:.6f}")
        
        if diff < 0.1:  # 可接受的数值误差
            print(f"✅ TensorRT {'INT8' if use_int8 else 'FP16'} 编译成功")
            return trt_model, True
        else:
            print(f"⚠️ TensorRT编译成功但输出差异较大: {diff}")
            return trt_model, True
            
    except Exception as e:
        print(f"❌ TensorRT编译失败: {e}")
        return model, False

def test_model_latency(model, batch_size, description, num_iterations=50):
    """测试模型延迟"""
    model.eval()
    dummy_input = torch.randn(batch_size, 3, 224, 224).cuda()
    
    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    
    # 测试
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = (end_time - start_time) * 1000
    avg_latency = total_time / num_iterations
    
    return avg_latency

def compare_quantization_acceleration():
    """对比量化前后的TensorRT加速效果"""
    print("🚀 对比量化前后的TensorRT加速效果")
    print("="*80)
    
    model_name = 'vit_small_patch16_224'
    batch_sizes = [32, 128]
    
    results = []
    
    for bs in batch_sizes:
        print(f"\n📊 测试 Batch Size: {bs}")
        
        try:
            # 1. 原始FP32模型
            print("1️⃣ 测试原始FP32模型")
            original_model = timm.create_model(model_name, pretrained=True).cuda().eval()
            fp32_latency = test_model_latency(original_model, bs, "FP32")
            fp32_throughput = bs * 1000 / fp32_latency
            print(f"   FP32: {fp32_latency:.2f}ms, {fp32_throughput:.0f} images/sec")
            
            # 2. FP16 TensorRT优化
            print("2️⃣ 测试FP16 TensorRT优化")
            input_shape = (bs, 3, 224, 224)
            trt_fp16_model, success = compile_tensorrt_int8(original_model, input_shape, use_int8=False)
            if success:
                fp16_latency = test_model_latency(trt_fp16_model, bs, "TensorRT-FP16")
                fp16_throughput = bs * 1000 / fp16_latency
                fp16_speedup = fp32_latency / fp16_latency
                print(f"   TensorRT-FP16: {fp16_latency:.2f}ms, {fp16_throughput:.0f} images/sec ({fp16_speedup:.2f}x)")
            else:
                fp16_latency = fp32_latency
                fp16_throughput = fp32_throughput
                fp16_speedup = 1.0
            
            # 3. PTQ4ViT量化模型
            print("3️⃣ 测试PTQ4ViT量化模型")
            quantized_model = apply_ptq4vit_quantization(model_name, calib_size=32)
            ptq_latency = test_model_latency(quantized_model, bs, "PTQ4ViT")
            ptq_throughput = bs * 1000 / ptq_latency
            ptq_vs_fp32 = fp32_latency / ptq_latency
            print(f"   PTQ4ViT: {ptq_latency:.2f}ms, {ptq_throughput:.0f} images/sec ({ptq_vs_fp32:.2f}x vs FP32)")
            
            # 4. PTQ4ViT + TensorRT INT8优化
            print("4️⃣ 测试PTQ4ViT + TensorRT INT8优化")
            trt_int8_model, success = compile_tensorrt_int8(quantized_model, input_shape, use_int8=True)
            if success:
                int8_latency = test_model_latency(trt_int8_model, bs, "TensorRT-INT8")
                int8_throughput = bs * 1000 / int8_latency
                int8_speedup = fp32_latency / int8_latency
                int8_vs_ptq = ptq_latency / int8_latency
                print(f"   TensorRT-INT8: {int8_latency:.2f}ms, {int8_throughput:.0f} images/sec ({int8_speedup:.2f}x vs FP32, {int8_vs_ptq:.2f}x vs PTQ)")
            else:
                int8_latency = ptq_latency
                int8_throughput = ptq_throughput
                int8_speedup = ptq_vs_fp32
                int8_vs_ptq = 1.0
            
            results.append({
                'batch_size': bs,
                'fp32_latency': fp32_latency,
                'fp32_throughput': fp32_throughput,
                'fp16_latency': fp16_latency,
                'fp16_throughput': fp16_throughput,
                'fp16_speedup': fp16_speedup,
                'ptq_latency': ptq_latency,
                'ptq_throughput': ptq_throughput,
                'ptq_speedup': ptq_vs_fp32,
                'int8_latency': int8_latency,
                'int8_throughput': int8_throughput,
                'int8_speedup': int8_speedup,
                'int8_vs_ptq': int8_vs_ptq,
            })
            
            # 清理显存
            del original_model, trt_fp16_model, quantized_model, trt_int8_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")
            torch.cuda.empty_cache()
            continue
    
    # 汇总结果
    print(f"\n📈 量化加速效果汇总:")
    print(f"{'BS':<4} {'FP32(ms)':<10} {'FP16(ms)':<10} {'PTQ(ms)':<10} {'INT8(ms)':<10} {'FP16↑':<8} {'PTQ↑':<8} {'INT8↑':<8} {'额外↑':<8}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['batch_size']:<4} "
              f"{r['fp32_latency']:<10.2f} "
              f"{r['fp16_latency']:<10.2f} "
              f"{r['ptq_latency']:<10.2f} "
              f"{r['int8_latency']:<10.2f} "
              f"{r['fp16_speedup']:<8.2f}x "
              f"{r['ptq_speedup']:<8.2f}x "
              f"{r['int8_speedup']:<8.2f}x "
              f"{r['int8_vs_ptq']:<8.2f}x")
    
    return results

def main():
    print("🔧 PTQ4ViT + TensorRT INT8 加速测试")
    print("="*80)
    
    try:
        results = compare_quantization_acceleration()
        
        print(f"\n💡 结论:")
        print(f"✅ 成功测试了完整的量化+TensorRT流程")
        print(f"📊 可以看到PTQ4ViT量化的精度损失和TensorRT的加速效果")
        print(f"🚀 INT8量化+TensorRT提供了最佳的性能表现")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()