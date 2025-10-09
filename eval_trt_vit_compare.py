#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import math
from collections import deque

import numpy as np
from PIL import Image

import tensorrt as trt
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_pil(path, out_h=224, out_w=224):
    # Resize shorter side to 256, then center crop 224x224, normalize to ImageNet, NCHW float32
    img = Image.open(path).convert("RGB")
    w, h = img.size
    short = 256
    if h < w:
        new_h = short
        new_w = int(round(w * short / h))
    else:
        new_w = short
        new_h = int(round(h * short / w))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    left = (new_w - out_w) // 2
    top  = (new_h - out_h) // 2
    img = img.crop((left, top, left + out_w, top + out_h))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD   # HWC
    arr = np.transpose(arr, (2, 0, 1))           # CHW
    return arr

def load_val_list(list_path, data_root):
    items = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # support "path label" or "path\tlabel"
            parts = line.split()
            if len(parts) < 2:
                continue
            rel_path = parts[0]
            label = int(parts[1])
            full_path = os.path.join(data_root, rel_path.lstrip("./"))
            items.append((full_path, label))
    return items

def topk_correct(logits, labels, ks=(1,5)):
    # logits: [B, C], labels: [B]
    pred = np.argsort(-logits, axis=1)  # descending
    res = {}
    for k in ks:
        topk = pred[:, :k]
        correct = (topk == labels[:, None]).any(axis=1).sum()
        res[k] = int(correct)
    return res

class TrtEngineRunner:
    def __init__(self, engine_path, batch, input_hw=(224,224)):
        self.engine_path = engine_path
        self.batch = batch
        self.h, self.w = input_hw

        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine is not None, f"Failed to load engine: {engine_path}"
        self.context = self.engine.create_execution_context()
        assert self.context is not None, "Failed to create execution context"

        # Find bindings
        self.input_indices = []
        self.output_indices = []
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_indices.append(i)
            else:
                self.output_indices.append(i)
        assert len(self.input_indices) == 1, "Expect exactly 1 input"
        assert len(self.output_indices) == 1, "Expect exactly 1 output"
        self.in_idx = self.input_indices[0]
        self.out_idx = self.output_indices[0]

        # Dynamic shapes?
        in_shape = self.engine.get_binding_shape(self.in_idx)
        if -1 in in_shape:
            # assume NCHW with variable N
            self.context.set_binding_shape(self.in_idx, (self.batch, 3, self.h, self.w))
        self.input_shape = tuple(self.context.get_binding_shape(self.in_idx))
        self.output_shape = tuple(self.context.get_binding_shape(self.out_idx))
        assert self.input_shape[0] == self.batch, f"Engine batch {self.input_shape[0]} != {self.batch}"

        # Allocate device buffers
        self.d_input = cuda.mem_alloc(np.prod(self.input_shape) * np.float32().nbytes)
        self.d_output = cuda.mem_alloc(np.prod(self.output_shape) * np.float32().nbytes)
        self.bindings = [None] * self.engine.num_bindings
        self.bindings[self.in_idx] = int(self.d_input)
        self.bindings[self.out_idx] = int(self.d_output)

        self.stream = cuda.Stream()

        # Host buffers
        self.h_input = np.empty(self.input_shape, dtype=np.float32)
        self.h_output = np.empty(self.output_shape, dtype=np.float32)

    def infer(self, batch_np):
        # batch_np: [N,3,224,224] float32
        assert batch_np.shape == self.input_shape, f"bad input {batch_np.shape} vs {self.input_shape}"
        cuda.memcpy_htod_async(self.d_input, batch_np, self.stream)

        start_evt = cuda.Event()
        end_evt = cuda.Event()
        start_evt.record(self.stream)

        self.context.execute_async_v3(self.stream.handle, self.bindings)

        end_evt.record(self.stream)
        end_evt.synchronize()
        gpu_ms = start_evt.time_till(end_evt)  # ms

        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.h_output.copy(), gpu_ms

def run_eval(engine_path, data_root, val_list, batch=64, workers=4, warmup=50, iters=200, max_images=None, tag=""):
    items = load_val_list(val_list, data_root)
    if max_images:
        items = items[:max_images]
    n = len(items)
    if n == 0:
        raise RuntimeError("val_list is empty.")

    runner = TrtEngineRunner(engine_path, batch=batch, input_hw=(224,224))

    # Dataloader（简易、CPU 线程数用 workers 控制预取）
    # 这里用一个简单的预取队列，避免引入 heavy 依赖
    def batch_iter():
        buf = []
        for path, label in items:
            arr = preprocess_pil(path, 224, 224)
            buf.append((arr, label))
            if len(buf) == batch:
                xs = np.stack([x for x, _ in buf], axis=0).astype(np.float32)
                ys = np.array([y for _, y in buf], dtype=np.int64)
                yield xs, ys
                buf.clear()
        if buf:
            # last partial batch -> pad to full
            pad = batch - len(buf)
            xs = np.stack([x for x, _ in buf] + [buf[-1][0]]*pad, axis=0).astype(np.float32)
            ys = np.array([y for _, y in buf] + [buf[-1][1]]*pad, dtype=np.int64)
            yield xs, ys

    # Warmup
    it = 0
    for xs, _ in batch_iter():
        _, gpu_ms = runner.infer(xs)
        it += 1
        if it >= warmup:
            break

    # Measure
    lat_ms = []
    images_done = 0
    top1 = 0
    top5 = 0
    t0 = time.time()

    measured = 0
    for xs, ys in batch_iter():
        if measured >= iters:
            break
        logits, gpu_ms = runner.infer(xs)
        # real size for this batch (avoid counting padded samples)
        real_b = min(batch, n - images_done) if (images_done + batch) <= n else (n - images_done)
        if real_b <= 0:
            break
        # clip logits/labels to real size
        logits = logits[:real_b]
        ys = ys[:real_b]

        correct = topk_correct(logits, ys, ks=(1,5))
        top1 += correct[1]
        top5 += correct[5]
        images_done += real_b

        lat_ms.append(gpu_ms)
        measured += 1

    t1 = time.time()
    wall_s = t1 - t0
    imgs = images_done
    top1_acc = top1 / imgs * 100.0
    top5_acc = top5 / imgs * 100.0

    if len(lat_ms) == 0:
        raise RuntimeError("No measurements collected. Increase --iters or check val_list.")

    lat_arr = np.array(lat_ms, dtype=np.float64)
    mean_ms = float(lat_arr.mean())
    p50_ms = float(np.percentile(lat_arr, 50))
    p90_ms = float(np.percentile(lat_arr, 90))
    p95_ms = float(np.percentile(lat_arr, 95))
    p99_ms = float(np.percentile(lat_arr, 99))
    # Per-batch GPU ms -> per-image
    mean_ms_per_img = mean_ms / batch
    throughput = imgs / wall_s

    result = {
        "tag": tag,
        "engine": engine_path,
        "images": imgs,
        "batches_measured": measured,
        "batch": batch,
        "gpu_ms_per_batch_mean": mean_ms,
        "gpu_ms_per_img_mean": mean_ms_per_img,
        "gpu_ms_p50": p50_ms,
        "gpu_ms_p90": p90_ms,
        "gpu_ms_p95": p95_ms,
        "gpu_ms_p99": p99_ms,
        "throughput_ips_wall": throughput,
        "top1": top1_acc,
        "top5": top5_acc,
        "wall_seconds": wall_s,
    }
    return result

def print_result_table(res_list):
    # Pretty print
    def fmt(x, n=3):
        return f"{x:.{n}f}"
    print("\n=== Results ===")
    header = [
        "Tag", "Top1(%)", "Top5(%)",
        "GPU ms/b(mean)", "GPU ms/img", "p50", "p90", "p95", "p99",
        "Throughput(img/s)", "Images", "Batch", "Wall(s)"
    ]
    print("\t".join(header))
    for r in res_list:
        row = [
            r["tag"],
            fmt(r["top1"], 2), fmt(r["top5"], 2),
            fmt(r["gpu_ms_per_batch_mean"]), fmt(r["gpu_ms_per_img_mean"]),
            fmt(r["gpu_ms_p50"]), fmt(r["gpu_ms_p90"]), fmt(r["gpu_ms_p95"]), fmt(r["gpu_ms_p99"]),
            fmt(r["throughput_ips_wall"]), str(r["images"]), str(r["batch"]), fmt(r["wall_seconds"])
        ]
        print("\t".join(row))
    # If have two results, print speedup
    if len(res_list) == 2:
        a, b = res_list
        # define speedup: per-image GPU latency or wall throughput
        sp_lat = a["gpu_ms_per_img_mean"] / b["gpu_ms_per_img_mean"]
        sp_thr = b["throughput_ips_wall"] / a["throughput_ips_wall"]
        print("\nSpeedup ({} -> {}):".format(a["tag"], b["tag"]))
        print(f"- GPU latency per image: x{sp_lat:.2f} (smaller is better)")
        print(f"- Wall throughput (img/s): x{sp_thr:.2f} (larger is better)")

def main():
    ap = argparse.ArgumentParser("TRT ViT INT8 vs FP32/TF32 - Speed & Accuracy")
    ap.add_argument("--int8-engine", type=str, default=None, help="INT8 engine path (*.plan)")
    ap.add_argument("--fp32-engine", type=str, default=None, help="FP32/TF32 engine path (*.plan)")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--val-list", type=str, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4, help="(placeholder) kept for interface, not used heavily")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200, help="measured batches")
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    res = []
    if args.int8_engine is None and args.fp32_engine is None:
        raise SystemExit("Please provide at least one engine: --int8-engine or --fp32-engine")

    if args.fp32_engine:
        r = run_eval(
            args.fp32_engine, args.data_root, args.val_list,
            batch=args.batch, workers=args.workers,
            warmup=args.warmup, iters=args.iters, max_images=args.max_images, tag="FP32/TF32"
        )
        res.append(r)
        print_result_table([r])

    if args.int8_engine:
        r = run_eval(
            args.int8_engine, args.data_root, args.val_list,
            batch=args.batch, workers=args.workers,
            warmup=args.warmup, iters=args.iters, max_images=args.max_images, tag="INT8"
        )
        res.append(r)
        if len(res) == 2:
            # print both and speedup
            # ensure order: FP32 first then INT8 for speedup display
            res_sorted = sorted(res, key=lambda x: 0 if x["tag"] == "FP32/TF32" else 1)
            print_result_table(res_sorted)
        else:
            print_result_table([r])

if __name__ == "__main__":
    main()
