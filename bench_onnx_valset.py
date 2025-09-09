import argparse, os, time, statistics, glob
import numpy as np
from PIL import Image
import onnx, onnxruntime as ort
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--val", required=True, help="ImageNet val 目录（子文件夹为类名）")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-n", type=int, default=5000, help="最多评估多少张；0 表示全部")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--provider", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--input-name", default=None, help="默认读取模型第一个输入名")
    ap.add_argument("--preload", action="store_true", help="先将图像预处理并载入内存，避免I/O影响计时")
    return ap.parse_args()

def build_filelist(root):
    paths=[]
    classes=[]
    for d in sorted(os.listdir(root)):
        dd=os.path.join(root,d)
        if not os.path.isdir(dd): continue
        for p in glob.glob(os.path.join(dd, "*")):
            paths.append((p,d))
            classes.append(d)
    classes=sorted(list(set(classes)))
    cls2id={c:i for i,c in enumerate(classes)}
    return paths, cls2id

def load_img(path):
    img = Image.open(path).convert("RGB")
    tr = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tr(img).unsqueeze(0).numpy().astype(np.float32)  # [1,3,224,224]
    return x

def main():
    args = parse()
    model = onnx.load(args.onnx)
    input_name = args.input_name or model.graph.input[0].name

    providers = (["CUDAExecutionProvider","CPUExecutionProvider"]
                 if args.provider=="cuda" else ["CPUExecutionProvider"])
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.onnx, so, providers=providers)

    files, cls2id = build_filelist(args.val)
    if args.max_n and args.max_n > 0:
        files = files[:args.max_n]
    N = len(files)
    print(f"Dataset: {args.val}  N={N}  batch={args.batch}")
    print(f"Model: {args.onnx}  Input={input_name}  Provider={args.provider.upper()}")

    # 预加载（可选）
    preload_buf = []
    labels = []
    if args.preload:
        for p, c in files:
            preload_buf.append(load_img(p))
            labels.append(cls2id[c])
        print(f"[preload] loaded {len(preload_buf)} tensors into RAM")

    # Warmup（用前几个batch）
    dummy = np.random.randn(args.batch, 3, 224, 224).astype(np.float32)
    for _ in range(args.warmup):
        sess.run(None, {input_name: dummy})

    lat_ms = []
    top1 = 0
    seen = 0
    buf = []
    ybuf = []

    def run_batch(X, Y):
        nonlocal top1, seen
        t0 = time.perf_counter()
        logits = sess.run(None, {input_name: X})[0]
        t1 = time.perf_counter()
        lat_ms.append((t1 - t0) * 1000.0)
        pred = logits.argmax(axis=1)
        top1 += (pred == np.array(Y)).sum()
        seen += len(Y)

    for i, (p,c) in enumerate(files):
        if args.preload:
            x = preload_buf[i]
        else:
            x = load_img(p)
        buf.append(x)
        ybuf.append(cls2id[c])
        if len(buf) == args.batch:
            X = np.concatenate(buf, axis=0)
            run_batch(X, ybuf)
            buf, ybuf = [], []

    if buf:
        X = np.concatenate(buf, axis=0)
        run_batch(X, ybuf)

    avg = statistics.mean(lat_ms)
    p50 = statistics.median(lat_ms)
    p90 = np.percentile(lat_ms, 90)
    p99 = np.percentile(lat_ms, 99)
    thr = seen * 1000.0 / sum(lat_ms)  # imgs/s（纯推理）
    acc = top1 / seen

    print(f"\nImages seen: {seen}  Top-1: {acc:.4f}")
    print(f"Latency per batch (ms): avg={avg:.2f}  p50={p50:.2f}  p90={p90:.2f}  p99={p99:.2f}")
    print(f"Throughput (pure inference) ≈ {thr:.2f} img/s")
    print("(注) 统计只计 session.run 时间，不含图像加载；若 --preload 则几乎不受I/O影响。")

if __name__ == "__main__":
    main()
