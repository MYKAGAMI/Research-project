import torch

# FP32 モデルロード
fp32 = torch.load('/home/feng/PTQ4ViT/fp32_tome_r8_vit_base.pth')
print("FP32 weight dtype:", fp32['patch_embed.proj.weight'].dtype)  # 期待: torch.float32

# INT8 モデルロード
int8 = torch.load('/home/feng/PTQ4ViT/quantized_tome_r8_vit_base.pth')
print("INT8 weight dtype:", int8['patch_embed.proj.weight'].dtype)  # 期待: torch.int8 または torch.qint8