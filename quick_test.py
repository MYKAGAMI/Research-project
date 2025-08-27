import sys
sys.path.insert(0,'.')

from example.test_vit import *
import utils.net_wrap as net_wrap
import utils.datasets as datasets
from utils.quant_calib import HessianQuantCalibrator

def test_single_model():
    print("=== PTQ4ViT 量化测试 ===")
    
    # 选择一个小模型进行快速测试
    model_name = 'vit_tiny_patch16_224'
    config_name = "PTQ4ViT"
    
    print(f"测试模型: {model_name}")
    print(f"量化方法: {config_name}")
    
    # 初始化配置
    quant_cfg = init_config(config_name)
    
    # 加载模型
    net = get_net(model_name)
    
    # 包装量化层
    wrapped_modules = net_wrap.wrap_modules_in_net(net, quant_cfg)
    
    # 准备数据（使用你的数据集路径）
    g = datasets.ViTImageNetLoaderGenerator(
        '/mnt/ino-raid4/usrs/feng/ImageNet',
        'imagenet', 32, 32, 16, 
        kwargs={"model": net}
    )
    test_loader = g.test_loader()
    calib_loader = g.calib_loader(num=32)  # 使用32张图片校准
    
    # 量化校准
    print("开始量化校准...")
    quant_calibrator = HessianQuantCalibrator(
        net, wrapped_modules, calib_loader, 
        sequential=False, batch_size=2  # 减小batch_size避免内存问题
    )
    quant_calibrator.batching_quant_calib()
    
    # 测试精度（只测试前100个batch，加快速度）
    print("测试量化后的精度...")
    acc = test_classification(net, test_loader, max_iteration=100)
    
    print("\n=== 测试结果 ===")
    print(f"模型: {model_name}")
    print(f"量化方法: {config_name}")
    print(f"8-bit量化精度: {acc:.4f}")
    print("测试完成！")

if __name__ == "__main__":
    test_single_model()
