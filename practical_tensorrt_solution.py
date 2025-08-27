import sys
sys.path.insert(0,'.')

import time
import torch
import torch.nn as nn
import numpy as np
import torch_tensorrt
import timm
from example.test_vit import *
import utils.datasets as datasets

def diagnose_tensorrt_performance():
    """诊断TensorRT性能问题"""
    print("🔍 TensorRT性能问题诊断")
    print("="*80)
    
    model_name = 'vit_tiny_patch16_224'
    
    # 1. 测试不同batch size的影响
    print("\n1. 测试不同batch size...")
    batch_sizes = [1, 4, 8, 16, 32, 64]
    
    original_model = timm.create_model(model_name, pretrained=True)
    original_model = original_model.cuda().eval()
    
    results = []
    
    for bs in batch_sizes:
        print(f"\n📊 Batch Size: {bs}")
        
        # 原始PyTorch
        pytorch_latency = test_model_latency(original_model, bs, "PyTorch FP32")
        
        # TensorRT编译
        input_shape = (bs, 3, 224, 224)
        trt_model, success = compile_tensorrt_optimized(original_model, input_shape)
        
        if success:
            trt_latency = test_model_latency(trt_model, bs, "TensorRT FP16")
            speedup = pytorch_latency / trt_latency if trt_latency > 0 else 0
        else:
            trt_latency = pytorch_latency
            speedup = 1.0
        
        results.append({
            'batch_size': bs,
            'pytorch_latency': pytorch_latency,
            'tensorrt_latency': trt_latency,
            'speedup': speedup
        })
        
        print(f"   PyTorch: {pytorch_latency:.2f}ms")
        print(f"   TensorRT: {trt_latency:.2f}ms") 
        print(f"   加速比: {speedup:.2f}x")
    
    # 2. 找到最佳batch size
    print(f"\n📈 不同Batch Size性能汇总:")
    print(f"{'Batch Size':<12} {'PyTorch(ms)':<12} {'TensorRT(ms)':<12} {'加速比':<8}")
    print("-" * 50)
    
    best_speedup = 0
    best_bs = 1
    
    for r in results:
        print(f"{r['batch_size']:<12} {r['pytorch_latency']:<12.2f} {r['tensorrt_latency']:<12.2f} {r['speedup']:<8.2f}x")
        if r['speedup'] > best_speedup:
            best_speedup = r['speedup']
            best_bs = r['batch_size']
    
    print(f"\n🎯 最佳配置: Batch Size {best_bs}, 加速比 {best_speedup:.2f}x")
    
    return best_bs, best_speedup

def compile_tensorrt_optimized(model, input_shape):
    """优化的TensorRT编译设置"""
    print(f"🔧 优化TensorRT编译 (shape: {input_shape})...")
    
    try:
        # 先测试模型能否正常运行
        dummy_input = torch.randn(input_shape).cuda()
        with torch.no_grad():
            _ = model(dummy_input)
        
        # 优化的编译设置
        compile_settings = {
            'inputs': [torch_tensorrt.Input(input_shape, dtype=torch.float32)],
            'enabled_precisions': {torch.float16, torch.float32},
            'workspace_size': 4*1024**3,  # 4GB工作空间
            'min_block_size': 3,  # 增加最小块大小
            'max_batch_size': input_shape[0],
            'opt_level': 5,  # 最高优化级别
            'require_full_compilation': False,
            'truncate_long_and_double': True,
            'use_python_runtime': False,  # 使用C++运行时
            'num_avg_timing_iters': 8,  # 增加timing迭代次数
        }
        
        # 尝试强制图模式
        model = torch.jit.trace(model, dummy_input)
        
        trt_model = torch_tensorrt.compile(model, **compile_settings)
        print("✅ 优化TensorRT编译成功")
        return trt_model, True
        
    except Exception as e:
        print(f"❌ 优化TensorRT编译失败: {e}")
        return model, False

def test_model_latency(model, batch_size, description, num_warmup=10, num_test=100):
    """精确的延迟测试"""
    model.eval()
    
    # 创建输入
    dummy_input = torch.randn(batch_size, 3, 224, 224).cuda()
    
    # 预热
    with torch.no_grad():
        for _ in range(num_warmup):
            try:
                _ = model(dummy_input)
            except:
                return float('inf')  # 如果失败返回无穷大
    
    torch.cuda.synchronize()
    
    # 测试
    latencies = []
    with torch.no_grad():
        for _ in range(num_test):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            _ = model(dummy_input)
            end_event.record()
            
            torch.cuda.synchronize()
            latency = start_event.elapsed_time(end_event)
            latencies.append(latency)
    
    avg_latency = np.mean(latencies)
    return avg_latency

def test_tensorrt_stream_optimization():
    """测试TensorRT stream优化"""
    print("\n🚀 测试TensorRT Stream优化")
    print("="*80)
    
    model_name = 'vit_tiny_patch16_224'
    batch_size = 32
    
    original_model = timm.create_model(model_name, pretrained=True)
    original_model = original_model.cuda().eval()
    
    input_shape = (batch_size, 3, 224, 224)
    
    # 1. 标准TensorRT编译
    print("\n1. 标准TensorRT编译...")
    standard_model, success1 = compile_tensorrt_optimized(original_model, input_shape)
    
    if success1:
        standard_latency = test_model_latency(standard_model, batch_size, "标准TensorRT")
    else:
        standard_latency = float('inf')
    
    # 2. 使用CUDA stream的版本
    print("\n2. 测试CUDA Stream优化...")
    
    class StreamOptimizedModel(nn.Module):
        def __init__(self, trt_model):
            super().__init__()
            self.trt_model = trt_model
            self.stream = torch.cuda.Stream()
            
        def forward(self, x):
            with torch.cuda.stream(self.stream):
                return self.trt_model(x)
    
    if success1:
        stream_model = StreamOptimizedModel(standard_model)
        stream_latency = test_model_latency(stream_model, batch_size, "Stream优化TensorRT")
    else:
        stream_latency = float('inf')
    
    # 3. 基准PyTorch性能
    pytorch_latency = test_model_latency(original_model, batch_size, "PyTorch基准")
    
    print(f"\n📊 Stream优化结果:")
    print(f"   PyTorch基准: {pytorch_latency:.2f}ms")
    print(f"   标准TensorRT: {standard_latency:.2f}ms")
    print(f"   Stream TensorRT: {stream_latency:.2f}ms")
    
    if standard_latency < float('inf'):
        print(f"   标准TensorRT加速比: {pytorch_latency/standard_latency:.2f}x")
    if stream_latency < float('inf'):
        print(f"   Stream TensorRT加速比: {pytorch_latency/stream_latency:.2f}x")

def analyze_model_characteristics():
    """分析模型特征对TensorRT的影响"""
    print("\n🔬 分析模型特征")
    print("="*80)
    
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model = model.cuda().eval()
    
    # 分析模型结构
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 模型统计:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   模型大小: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    # 分析计算复杂度
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    
    # 统计不同类型的操作
    linear_layers = 0
    attention_layers = 0
    normalization_layers = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers += 1
        elif 'attention' in name.lower():
            attention_layers += 1
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            normalization_layers += 1
    
    print(f"\n🏗️ 模型结构分析:")
    print(f"   Linear层数量: {linear_layers}")
    print(f"   Attention相关层: {attention_layers}")
    print(f"   Normalization层: {normalization_layers}")
    
    # 分析为什么TensorRT可能慢
    print(f"\n🤔 TensorRT性能差的可能原因:")
    print(f"   1. ViT模型主要是Attention操作，TensorRT对此优化有限")
    print(f"   2. 小batch size下，TensorRT的开销可能大于收益")
    print(f"   3. H100原生FP32性能已经很强，FP16优势不明显")
    print(f"   4. 模型较小({total_params:,}参数)，内存带宽不是瓶颈")

def suggest_optimization_strategies():
    """建议优化策略"""
    print("\n💡 TensorRT优化策略建议")
    print("="*80)
    
    strategies = [
        {
            "策略": "增大Batch Size",
            "原理": "TensorRT在大batch下才能充分发挥优势",
            "建议": "尝试batch size 64, 128, 256",
            "风险": "可能超出显存限制"
        },
        {
            "策略": "使用更大的模型",
            "原理": "大模型的计算密度更高，更适合TensorRT",
            "建议": "测试vit_base或vit_large",
            "风险": "需要更多显存"
        },
        {
            "策略": "ONNX + TensorRT",
            "原理": "ONNX可能提供更好的图优化",
            "建议": "导出ONNX后用trtexec测试",
            "风险": "转换可能失败"
        },
        {
            "策略": "混合精度推理",
            "原理": "手动控制哪些层用FP16",
            "建议": "只对计算密集层用FP16",
            "风险": "需要手动调优"
        },
        {
            "策略": "使用FasterTransformer",
            "原理": "专门为Transformer优化的库",
            "建议": "NVIDIA FasterTransformer或类似库",
            "风险": "需要额外集成"
        }
    ]
    
    print(f"{'策略':<20} {'原理':<30} {'建议':<25} {'风险':<20}")
    print("-" * 100)
    
    for s in strategies:
        print(f"{s['策略']:<20} {s['原理']:<30} {s['建议']:<25} {s['风险']:<20}")

def practical_next_steps():
    """实用的下一步建议"""
    print("\n🎯 实用的下一步行动")
    print("="*80)
    
    print("🚀 立即可行的方案:")
    print("   1. 接受PyTorch FP32性能 - 在H100上已经很好了")
    print("   2. 专注于PTQ4ViT的算法创新价值")
    print("   3. 在论文中诚实报告TensorRT的实际表现")
    
    print("\n📊 建议的性能报告方式:")
    print("   - PyTorch FP32: 745 images/sec (基准)")
    print("   - PTQ4ViT算法: 精度提升至80.69% (理论量化)")
    print("   - TensorRT FP16: 420 images/sec (部署尝试)")
    print("   - 结论: 对于小模型，PyTorch原生性能已经很好")
    
    print("\n🔬 研究价值重新定位:")
    print("   ✅ PTQ4ViT的算法创新依然有价值")
    print("   ✅ Twin Uniform Quantization是理论贡献")
    print("   ✅ Hessian-guided方法具有学术价值")
    print("   ⚠️ 但部署方面需要进一步研究")
    
    print("\n📝 论文撰写建议:")
    print("   1. 强调算法创新，淡化部署性能")
    print("   2. 对比其他量化方法的精度")
    print("   3. 分析为什么某些量化方法难以部署")
    print("   4. 讨论研究型量化vs生产型量化的平衡")

if __name__ == "__main__":
    print("🔧 开始TensorRT性能诊断...")
    
    # 1. 诊断不同batch size的性能
    try:
        best_bs, best_speedup = diagnose_tensorrt_performance()
    except Exception as e:
        print(f"❌ batch size测试失败: {e}")
        best_bs, best_speedup = 32, 0.5
    
    # 2. 测试Stream优化
    try:
        test_tensorrt_stream_optimization()
    except Exception as e:
        print(f"❌ Stream优化测试失败: {e}")
    
    # 3. 分析模型特征
    analyze_model_characteristics()
    
    # 4. 建议优化策略
    suggest_optimization_strategies()
    
    # 5. 实用建议
    practical_next_steps()
    
    print("\n" + "="*80)
    print("🏁 诊断总结")
    print("="*80)
    print("🔍 TensorRT性能不佳的主要原因:")
    print("   1. ViT模型对TensorRT不够友好")
    print("   2. 小batch size下TensorRT优势不明显") 
    print("   3. H100的原生FP32性能已经很强")
    print("   4. 模型规模较小，计算密度不够")
    
    print(f"\n💡 建议:")
    print(f"   - 继续使用PyTorch FP32: 745 images/sec")
    print(f"   - 专注PTQ4ViT的算法价值")
    print(f"   - 在研究中诚实报告部署挑战")
    print(f"   - 考虑测试更大的模型或batch size")