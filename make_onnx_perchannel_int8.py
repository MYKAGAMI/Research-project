# make_onnx_perchannel_int8.py  (v3: handle Constant-weights & prune)
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
    return {I.name: I for I in graph.initializer}

def producers(graph):
    prod = {}
    for n in graph.node:
        for o in n.output:
            prod[o] = n
    return prod

def consumers(graph):
    cons = {}
    for n in graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    return cons

def add_or_replace_init(graph, name, arr, dtype):
    t = numpy_helper.from_array(arr.astype(dtype), name)
    m = init_map(graph)
    if name in m:
        idx = list(graph.initializer).index(m[name])
        graph.initializer.remove(m[name])
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
    drop = [I for I in list(graph.initializer) if I.name not in names]
    for I in drop:
        graph.initializer.remove(I)
    return len(drop)

def prune_dead_constants(graph):
    # 移除输出未被引用的 Constant 节点
    names = referenced_tensor_names(graph)
    removed = 0
    for n in list(graph.node):
        if n.op_type == "Constant":
            dead = True
            for o in n.output:
                if o in names:
                    dead = False
                    break
            if dead:
                graph.node.remove(n)
                removed += 1
    return removed

def run():
    args = parse_args()
    model = onnx.load(args.onnx)
    g = model.graph
    imap = init_map(g)
    prod = producers(g)
    cons = consumers(g)
    scales_npz = np.load(args.scales)

    nodes_to_remove = []
    inits_to_remove = set()
    constants_to_remove = set()

    # 匹配权重分支： (W fp32 [initializer or Constant]) --Q--> (w_q) --DQ(axis?)-> ... -> MatMul/Gemm
    for q in list(g.node):
        if q.op_type != "QuantizeLinear":
            continue

        w_src = q.input[0]
        Wfp = None
        weight_name_for_npz = None
        is_constant_node = False

        if w_src in imap:
            # initializer 情况
            Wfp = numpy_helper.to_array(imap[w_src])
            weight_name_for_npz = w_src
        elif w_src in prod and prod[w_src].op_type == "Constant":
            # Constant 情况
            cnode = prod[w_src]
            for attr in cnode.attribute:
                if attr.name == "value" and attr.t.data_type == onnx.TensorProto.FLOAT:
                    Wfp = numpy_helper.to_array(attr.t)
                    is_constant_node = True
                    weight_name_for_npz = w_src  # 用输出名索引 npz
                    break
        else:
            continue  # 不是我们要的权重路径

        # 找到与之相连的 DQ
        dq_list = [n for n in cons.get(q.output[0], []) if n.op_type == "DequantizeLinear"]
        if len(dq_list) != 1:  # 形状不同就跳过
            continue
        dq = dq_list[0]
        OC = Wfp.shape[0]

        # 取 per-channel scales
        if weight_name_for_npz in scales_npz.files:
            w_scales = scales_npz[weight_name_for_npz].astype(np.float32).reshape(-1)
        else:
            # fallback：用 |W|_max/127
            w_scales = np.maximum(np.abs(Wfp).max(axis=1), 1e-12) / 127.0
        assert w_scales.size == OC, f"scale size mismatch for {weight_name_for_npz}"

        # 量化 & 写入新的 int8 initializer
        Wq = np.clip(np.round(Wfp / w_scales[:, None]), -128, 127).astype(np.int8)
        wq_name = weight_name_for_npz + "_int8"
        add_or_replace_init(g, wq_name, Wq, np.int8)

        # 更新 DQ：输入换 INT8，scale/zp 改为向量（axis=0）
        new_scale = dq.input[1] + "_perC"
        new_zp    = dq.input[2] + "_perC"
        add_or_replace_init(g, new_scale, w_scales, np.float32)
        add_or_replace_init(g, new_zp, np.zeros_like(w_scales, dtype=np.int8), np.int8)

        dq.input[0] = wq_name
        dq.input[1] = new_scale
        dq.input[2] = new_zp

        has_axis = False
        for a in dq.attribute:
            if a.name == "axis":
                a.i = 0; has_axis = True; break
        if not has_axis:
            dq.attribute.extend([helper.make_attribute("axis", 0)])

        # 标记删除 Q
        nodes_to_remove.append(q)
        # 标记删除旧 FP32 权重
        if is_constant_node:
            constants_to_remove.add(prod[w_src])  # 删除 Constant 节点
        else:
            inits_to_remove.add(w_src)  # 删除 initializer

        # 旧 per-tensor scale/zp（来自 Q 或 DQ）
        for name in [q.input[1] if len(q.input)>1 else None,
                     q.input[2] if len(q.input)>2 else None,
                     dq.input[1], dq.input[2]]:
            if name is not None and name in imap:
                inits_to_remove.add(name)

    # 应用删除
    for n in nodes_to_remove:
        g.node.remove(n)
    # 旧 initializer
    imap = init_map(g)
    for name in list(inits_to_remove):
        if name in imap:
            g.initializer.remove(imap[name])
    # 旧 Constant
    for cn in list(constants_to_remove):
        if cn in g.node:
            g.node.remove(cn)

    # 进一步清理未引用的 initializer / Constant
    dropped_inits = prune_unused_initializers(g)
    dropped_consts = prune_dead_constants(g)
    print(f"[Clean] removed unused initializers: {dropped_inits}, dead constants: {dropped_consts}")

    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print(f"[OK] per-channel INT8 weights ONNX saved to: {args.out}")

if __name__ == "__main__":
    run()


