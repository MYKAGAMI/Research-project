import sys
sys.path.insert(0,'.')

import torch
import timm
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# 测试模型加载
try:
    from utils.models import get_net
    net = get_net('vit_tiny_patch16_224')
    print("✓ 模型加载成功!")
except Exception as e:
    print("✗ 模型加载失败:", e)

# 测试数据集加载
try:
    from utils import datasets
    g = datasets.ViTImageNetLoaderGenerator(
        '/mnt/ino-raid4/usrs/feng/ImageNet',
        'imagenet', 32, 32, 16, 
        kwargs={"model": net}
    )
    test_loader = g.test_loader()
    print("✓ 数据集加载成功!")
    print(f"数据集大小: {len(test_loader)}")
except Exception as e:
    print("✗ 数据集加载失败:", e)
