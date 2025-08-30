# make_onnx_perchannel_int8.py  (v2: prune unused FP32 weights & scales)
import argparse, numpy as np
import onnx
from onnx import helper, numpy_helper

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--scales", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()

def init_map(graph):
    return {init.name: init for init in graph.initializer}

def node_consumers(graph):
    cons = {}
    for node in graph.node:
        for inp in node.input:
            cons.setdefault(inp, []).append(node)
    return cons

def remove_node(graph, n):
    graph.node.remove(n)

def add_or_replace_init(graph, name, arr, dtype):
    t = numpy_helper.from_array(arr.astype(dtype), name)
    imap = init_map(graph)
    if name in imap:
        idx = list(graph.initializer).index(imap[name])
        graph.initializer.remove(imap[name])
        graph.initializer.insert(idx, t)
    else:
        graph.initializer.append(t)

def referenced_tensor_names(graph):
    names = set()
    for n in graph.node:
        for x in n.input:
            if x: names.add(x)
        for y in n.output:
            if y: names.add(y)
    for x in graph.input:
        names.add(x.name)
    for y in graph.output:
        names.add(y.name)
    return names

def prune_unused_initializers(graph):
    names = referenced_tensor_names(graph)
    to_drop = [init for init in list(graph.initializer) if init.name not in names]
    for it in to_drop:
        graph.initializer.remove(it)
    return len(to_drop)

def run():
    args = parse_args()
    model = onnx.load(args.onnx)
    g = model.graph
    imap = init_map(g)
    cons = node_consumers(g)
    scales_npz = np.load(args.scales)

    nodes_to_remove = []
    inits_to_remove = set()  # 待删除的旧 FP32 权重 / 旧 scale / 旧 zp

    # 匹配:  (fp32 W) --QuantizeLinear--> (w_q) --DequantizeLinear--> (w_dq) --> MatMul/Gemm
    for q in list(g.node):
        if q.op_type != "QuantizeLinear":
            continue
        w_fp = q.input[0]
        if w_fp not in imap:  # 有的图用 Constant 节点承载权重，这里先只处理 initializer 情形
            continue
        if not (w_fp.endswith(".inner.weight") or w_fp.endswith("weight")):
            continue

        q_out = q.output[0]
        dq_candidates = [n for n in cons.get(q_out, []) if n.op_type == "DequantizeLinear"]
        if len(dq_candidates) != 1:
            continue
        dq = dq_candidates[0]

        # 读 FP32 权重
        Wfp = numpy_helper.to_array(imap[w_fp])
        OC = Wfp.shape[0]

        # 取 per-channel scale（来自导出时保存的 npz）
        if w_fp in scales_npz.files:
            w_scales = scales_npz[w_fp].astype(np.float32).reshape(-1)
        else:
            # 兜底：按 |W|_max/127
            w_scales = np.maximum(np.abs(Wfp).max(axis=1), 1e-12) / 127.0
        assert w_scales.size == OC, f"scale size mismatch for {w_fp}"

        # 量化为 int8，并写入新的 initializer
        Wq = np.clip(np.round(Wfp / w_scales[:, None]), -128, 127).astype(np.int8)
        wq_name = w_fp + "_int8"
        add_or_replace_init(g, wq_name, Wq, np.int8)

        # 更新 DequantizeLinear：输入换成 int8，scale/zp 换成向量（axis=0）
        new_scale = dq.input[1] + "_perC"
        new_zp    = dq.input[2] + "_perC"
        add_or_replace_init(g, new_scale, w_scales, np.float32)
        add_or_replace_init(g, new_zp, np.zeros_like(w_scales, dtype=np.int8), np.int8)

        # 记录旧的 per-tensor scale/zp（来自 Q 或旧的 DQ）以便清理
        old_q_scale, old_q_zp = (q.input[1] if len(q.input)>1 else None), (q.input[2] if len(q.input)>2 else None)
        old_dq_scale, old_dq_zp = dq.input[1], dq.input[2]
        for name in [old_q_scale, old_q_zp, old_dq_scale, old_dq_zp]:
            if name is not None and name in imap:
                inits_to_remove.add(name)

        # 应用修改
        dq.input[0] = wq_name
        dq.input[1] = new_scale
        dq.input[2] = new_zp

        # 设置 axis=0
        has_axis = False
        for a in dq.attribute:
            if a.name == "axis":
                a.i = 0
                has_axis = True
                break
        if not has_axis:
            dq.attribute.extend([helper.make_attribute("axis", 0)])

        # 标记删除 Q 节点、以及旧 FP32 权重
        nodes_to_remove.append(q)
        inits_to_remove.add(w_fp)

    # 删除 Q 节点
    for n in nodes_to_remove:
        remove_node(g, n)

    # 先删除我们明确标记的旧 initializer
    imap = init_map(g)
    for name in list(inits_to_remove):
        if name in imap:
            g.initializer.remove(imap[name])

    # 再做一遍“全图清理”：把任何未被引用的 initializer 通通移除
    dropped = prune_unused_initializers(g)
    print(f"[Clean] removed unused initializers: {dropped}")

    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print(f"[OK] per-channel INT8 weights ONNX saved to: {args.out}")

if __name__ == "__main__":
    run()


