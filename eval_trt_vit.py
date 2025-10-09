#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time, argparse, json, math
import numpy as np
from PIL import Image
import tensorrt as trt
import cuda  # from cuda-python
import ctypes

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_labels(label_file, data_root):
    # 返回 [(abs_path, class_id), ...]
    items = []
    with open(label_file, 'r') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            rel, lab = line.split()
            p = rel
            if not os.path.isabs(rel):
                p = os.path.join(data_root, rel)
            items.append((p, int(lab)))
    return items

def preprocess_imagenet(path, size=224):
    # torchvision: Resize(256) + CenterCrop(224) + toTensor + norm
    img = Image.open(path).convert('RGB')
    short = min(img.size)
    scale = 256.0 / short
    new_w = int(round(img.width * scale))
    new_h = int(round(img.height * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (img.width - size) // 2
    top  = (img.height - size) // 2
    img = img.crop((left, top, left+size, top+size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2,0,1)  # CHW
    return arr

def build_runtime(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

class TRTExecutor:
    def __init__(self, engine_path):
        self.engine = build_runtime(engine_path)
        assert self.engine is not None
        self.context = self.engine.create_execution_context()
        assert self.context is not None
        self.stream = cuda.Stream()
        # 绑定
        self.bindings = [None] * self.engine.num_bindings
        self.idx_in = None
        self.idx_out = None
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.idx_in = i
            else:
                self.idx_out = i
        in_shape = self.engine.get_binding_shape(self.idx_in)
        self.nC, self.nH, self.nW = in_shape[1], in_shape[2], in_shape[3]
        self.dtype_in  = trt.nptype(self.engine.get_binding_dtype(self.idx_in))
        self.dtype_out = trt.nptype(self.engine.get_binding_dtype(self.idx_out))
        self.batch = in_shape[0]
        # device mem (lazy)
        self.d_in = None
        self.d_out = None
        self.out_shape = self.engine.get_binding_shape(self.idx_out)

    def ensure_device(self, batch_size):
        # 重新设定动态 batch（如果 engine 是显式 batch）
        shape = self.engine.get_binding_shape(self.idx_in)
        if shape[0] != batch_size:
            ok = self.context.set_binding_shape(self.idx_in, (batch_size, 3, self.nH, self.nW))
            if not ok:
                raise RuntimeError("set_binding_shape failed")
        # 分配
        vol_in  = batch_size * 3 * self.nH * self.nW
        vol_out = batch_size * self.out_shape[1]
        bytes_in  = vol_in  * np.dtype(self.dtype_in).itemsize
        bytes_out = vol_out * np.dtype(self.dtype_out).itemsize
        if (self.d_in is None) or (self.d_in.size_bytes < bytes_in):
            if self.d_in is not None: self.d_in.free()
            self.d_in = cuda.mem_alloc(bytes_in)
        if (self.d_out is None) or (self.d_out.size_bytes < bytes_out):
            if self.d_out is not None: self.d_out.free()
            self.d_out = cuda.mem_alloc(bytes_out)
        self.bindings[self.idx_in]  = int(self.d_in)
        self.bindings[self.idx_out] = int(self.d_out)

    def infer(self, batch_np):
        bs = batch_np.shape[0]
        self.ensure_device(bs)
        # H2D
        cuda.memcpy_htod_async(self.d_in, batch_np.ravel(), self.stream)
        # 执行
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        # D2H
        out_np = np.empty((bs, self.out_shape[1]), dtype=self.dtype_out)
        cuda.memcpy_dtoh_async(out_np, self.d_out, self.stream)
        self.stream.synchronize()
        return out_np

def iterate_batches(items, batch_size):
    batch = []
    for p, lab in items:
        batch.append((p, lab))
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def topk_acc(logits, labels, k=1):
    # logits: [B, C], labels: [B]
    idx = np.argsort(-logits, axis=1)[:, :k]
    hits = 0
    for i in range(labels.shape[0]):
        if labels[i] in idx[i]: hits += 1
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', required=True, help='TensorRT .plan path')
    ap.add_argument('--data-root', required=True, help='dataset root (images base dir)')
    ap.add_argument('--label-file', required=True, help='txt: "relative_path label_id" per line')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--max-samples', type=int, default=-1, help='limit samples for quick test')
    ap.add_argument('--num-workers', type=int, default=0)  # 占位
    args = ap.parse_args()

    items = load_labels(args.label_file, args.data_root)
    if args.max-samples != -1:  # 兼容写法（若有人把 --max-samples 写成 --max_samples）
        pass
    if args.max_samples and args.max_samples > 0:
        items = items[:args.max_samples]

    exe = TRTExecutor(args.engine)

    n_total = 0
    n_top1  = 0
    n_top5  = 0

    # 预热
    warm = min(args.warmup, max(1, args.batch))
    dummy = np.zeros((warm, 3, exe.nH, exe.nW), dtype=np.float32)
    exe.infer(dummy)

    # 正式计时与评测
    t0 = time.time()
    for batch in iterate_batches(items, args.batch):
        xs = np.zeros((len(batch), 3, exe.nH, exe.nW), dtype=np.float32)
        ys = np.zeros((len(batch),), dtype=np.int64)
        for i, (p, lab) in enumerate(batch):
            xs[i] = preprocess_imagenet(p, size=exe.nH)
            ys[i] = lab
        logits = exe.infer(xs)  # [B, C]
        n_total += len(batch)
        n_top1  += topk_acc(logits, ys, k=1)
        n_top5  += topk_acc(logits, ys, k=5)
    t1 = time.time()

    dt = t1 - t0
    ips = n_total / dt if dt > 0 else 0.0
    top1 = 100.0 * n_top1 / n_total
    top5 = 100.0 * n_top5 / n_total

    print(json.dumps({
        "engine": args.engine,
        "samples": n_total,
        "elapsed_sec": round(dt, 4),
        "throughput_ips": round(ips, 2),
        "top1_acc_%": round(top1, 3),
        "top5_acc_%": round(top5, 3),
    }, indent=2))

if __name__ == "__main__":
    main()
