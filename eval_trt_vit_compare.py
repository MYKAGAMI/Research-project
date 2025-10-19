#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_trt_vit_compare.py
- 自动识别 ImageNet 目录结构：val/<class_id>/*.jpg
- 可选对比 INT8 与 FP32/TF32 两个 TensorRT engine 的 速度 + 精度
- 也可只测其中一个 engine
"""

import os
import time
import argparse
import json
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

try:
    import tensorrt as trt
except Exception as e:
    raise RuntimeError("需要 TensorRT Python 包（通常随 TensorRT 安装）。") from e

# 尝试使用 PyCUDA（最常见）
try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    HAVE_PYCUDA = True
except Exception:
    HAVE_PYCUDA = False

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def log(s): print(s, flush=True)

def load_val_list(list_path: str, data_root: str) -> List[Tuple[str, int]]:
    """从 val_list.txt 读取：(相对/绝对路径, label)"""
    items = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            path, lab = line.split(maxsplit=1)
            if not os.path.isabs(path):
                path = os.path.join(data_root, path)
            items.append((path, int(lab)))
    return items

def auto_scan_imagenet_folder(data_root: str) -> Tuple[List[Tuple[str, int]], Dict[str,int], List[str]]:
    """
    自动扫描 ImageNet val 目录：data_root/class_id/*.jpg
    返回：
      items: [(img_path, label_idx), ...]
      class_to_idx: {'n01440764': 0, ...}
      classes: [class_name_sorted]
    """
    classes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    classes.sort()
    class_to_idx = {c:i for i,c in enumerate(classes)}

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    items = []
    for c in classes:
        cdir = os.path.join(data_root, c)
        for root, _, files in os.walk(cdir):
            for fn in files:
                if fn.lower().endswith(exts):
                    items.append((os.path.join(root, fn), class_to_idx[c]))
    items.sort()
    return items, class_to_idx, classes

def preprocess_image(path: str, size: int = 224) -> np.ndarray:
    """
    返回 NCHW float32 预处理图像（1,3,224,224），ImageNet mean/std 归一化
    """
    img = Image.open(path).convert("RGB")
    # 简单的短边缩放 + center crop（与大多数 ViT 推理一致）
    w, h = img.size
    scale = size * 256 // 224  # 按 224->256 的常见比率先缩放
    short = min(w, h)
    if short != 0:
        resize_to = int(scale * max(w, h) / short) if short == h else int(scale * max(h, w) / short)
    else:
        resize_to = scale
    # 更稳妥：直接按照短边到 256
    if w < h:
        new_w, new_h = 256, int(h * 256 / w)
    else:
        new_h, new_w = 256, int(w * 256 / h)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop 224
    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    arr = (np.array(img).astype(np.float32) / 255.0)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC->CHW
    arr = np.expand_dims(arr, 0)  # NCHW
    return arr

def build_batches(items: List[Tuple[str,int]], batch: int) -> List[Tuple[List[str], np.ndarray, np.ndarray]]:
    """
    将样本打成 batch，返回 [(paths, input_batch, labels_batch), ...]
    """
    batches = []
    for i in range(0, len(items), batch):
        chunk = items[i:i+batch]
        paths = [p for p,_ in chunk]
        labels = np.array([lab for _,lab in chunk], dtype=np.int64)
        inputs = [preprocess_image(p) for p in paths]
        x = np.concatenate(inputs, axis=0)  # (N,3,224,224)
        batches.append((paths, x, labels))
    return batches

def load_engine(engine_path: str):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    return engine

def create_context_and_io(engine):
    context = engine.create_execution_context()
    # 假设单输入单输出：
    assert engine.num_io_tensors == 2, "脚本假设 1 输入 1 输出"
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_name  = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    output_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    return context, input_name, output_name

def alloc_buffers(engine, context, input_name, output_name, batch):
    # 设置动态 batch（若 engine 是显式 batch）
    ishape = list(engine.get_tensor_shape(input_name))
    if ishape[0] == -1:
        ishape[0] = batch
    context.set_input_shape(input_name, tuple(ishape))

    input_nbytes  = np.prod(ishape) * np.dtype(np.float32).itemsize
    oshape = context.get_tensor_shape(output_name)
    if oshape[0] == -1:
        # 根据 batch 推断
        oshape = list(oshape)
        oshape[0] = batch
        oshape = tuple(oshape)
    output_nbytes = np.prod(oshape) * np.dtype(np.float32).itemsize

    if not HAVE_PYCUDA:
        raise RuntimeError(
            "需要 PyCUDA 以进行 GPU 内存分配与拷贝。请安装：\n"
            "pip install pycuda --extra-index-url https://pypi.nvidia.com"
        )

    d_input  = cuda.mem_alloc(int(input_nbytes))
    d_output = cuda.mem_alloc(int(output_nbytes))
    stream   = cuda.Stream()
    return d_input, d_output, stream, tuple(oshape)

def infer_batches(engine_path: str, batches, warmup: int, iters: int) -> Dict[str, float]:
    engine = load_engine(engine_path)
    context, in_name, out_name = create_context_and_io(engine)

    # 用首个 batch 的维度进行绑定创建
    first_batch = batches[0][1]
    batch = first_batch.shape[0]
    d_input, d_output, stream, oshape = alloc_buffers(engine, context, in_name, out_name, batch)

    context.set_tensor_address(in_name, int(d_input))
    context.set_tensor_address(out_name, int(d_output))

    # 统计
    total_images = 0
    correct_top1 = 0

    # 预热
    w = 0
    while w < warmup:
        for _, x, _ in batches:
            if x.shape[0] != batch:
                # 最后不足 batch 的一组，跳过预热/评测
                continue
            cuda.memcpy_htod_async(d_input, x, stream)
            context.enqueue_v3(stream.handle)
            stream.synchronize()
            w += 1
            if w >= warmup:
                break

    # 计时评测
    ran = 0
    t0 = time.time()
    while ran < iters:
        for _, x, labels in batches:
            if x.shape[0] != batches[0][1].shape[0]:
                continue  # 跳过非整 batch
            cuda.memcpy_htod_async(d_input, x, stream)
            context.enqueue_v3(stream.handle)
            # 拷回输出
            out_host = np.empty(oshape, dtype=np.float32)
            cuda.memcpy_dtoh_async(out_host, d_output, stream)
            stream.synchronize()

            # 计算 top1
            pred = out_host.argmax(axis=1)
            correct_top1 += (pred == labels).sum()
            total_images += labels.shape[0]

            ran += 1
            if ran >= iters:
                break
    t1 = time.time()
    wall = t1 - t0

    throughput = total_images / wall if wall > 0 else 0.0
    top1 = correct_top1 / total_images if total_images > 0 else 0.0
    return {
        "images": int(total_images),
        "correct": int(correct_top1),
        "top1": float(top1),
        "time_sec": float(wall),
        "throughput_ips": float(throughput),
        "batch": int(batches[0][1].shape[0]),
    }

def main():
    ap = argparse.ArgumentParser("TensorRT ViT INT8 vs FP32/TF32 速度+精度对比")
    ap.add_argument("--int8-engine", type=str, default=None, help="INT8 engine .plan 路径")
    ap.add_argument("--fp32-engine", type=str, default=None, help="FP32/TF32 engine .plan 路径")
    ap.add_argument("--data-root", type=str, required=True, help="验证集根目录（val/；按子目录=类别组织）")
    ap.add_argument("--val-list", type=str, default=None, help="可选：val_list.txt（path label），若给定则优先生效")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)  # 兼容参数，不实际使用
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--max-samples", type=int, default=50000, help="最多使用多少样本（默认 50k）")
    args = ap.parse_args()

    assert args.int8_engine or args.fp32_engine, "至少指定一个 engine"

    # 构建样本列表
    if args.val_list and os.path.isfile(args.val_list):
        log(f"[Info] 使用 val_list.txt: {args.val_list}")
        items = load_val_list(args.val_list, args.data_root)
        classes = None
    else:
        log(f"[Info] 自动扫描 ImageNet 目录: {args.data_root}")
        items, class_to_idx, classes = auto_scan_imagenet_folder(args.data_root)
        log(f"[Info] 类别数: {len(classes)}，样本数: {len(items)}")
        # 保存个 mapping 方便复现
        mapping_path = os.path.join(os.path.dirname(__file__), "imagenet_classes.json")
        try:
            with open(mapping_path, "w") as f:
                json.dump({"classes": classes, "class_to_idx": class_to_idx}, f, indent=2)
            log(f"[Info] 已保存类别映射到 {mapping_path}")
        except Exception:
            pass

    if len(items) == 0:
        raise RuntimeError("没有找到任何图片。请检查 --data-root 是否正确。")

    # 限制样本数以控制评测时长
    items = items[:args.max_samples]

    # 打 batch（最后不足 batch 的一组将被跳过）
    batches = build_batches(items, args.batch)
    if len(batches) == 0:
        raise RuntimeError("有效 batch 数为 0。请减小 --batch 或检查数据。")
    log(f"[Info] 可用整批数量: {len(batches)}（将跳过最后不足 {args.batch} 的批次）")

    results = {}
    if args.int8_engine:
        log("\n===== 评测 INT8 Engine =====")
        r = infer_batches(args.int8_engine, batches, args.warmup, args.iters)
        results["INT8"] = r
        log(json.dumps({"INT8": r}, indent=2, ensure_ascii=False))

    if args.fp32_engine:
        log("\n===== 评测 FP32/TF32 Engine =====")
        r = infer_batches(args.fp32_engine, batches, args.warmup, args.iters)
        results["FP32/TF32"] = r
        log(json.dumps({"FP32/TF32": r}, indent=2, ensure_ascii=False))

    # 汇总对比
    log("\n===== 汇总对比 =====")
    def brief(r):
        return f"top1={r['top1']*100:.2f}% | throughput={r['throughput_ips']:.1f} img/s | images={r['images']} | time={r['time_sec']:.2f}s"
    if "INT8" in results:
        log(f"INT8      : {brief(results['INT8'])}")
    if "FP32/TF32" in results:
        log(f"FP32/TF32 : {brief(results['FP32/TF32'])}")

if __name__ == "__main__":
    main()
