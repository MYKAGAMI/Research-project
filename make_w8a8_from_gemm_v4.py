# make_w8a8_from_gemm_v4.py
# Convert all Gemm/MatMul to explicit QDQ W8A8 (weights: per-channel symmetric along columns; activations: per-tensor symmetric)
# Works well with TensorRT explicit quantization (Q/DQ) on GPU (no DLA/asym).
# Python 3.8+, onnx >= 1.12 recommended.

import argparse, numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

ALLOW_OPS = {
    "DequantizeLinear","QuantizeLinear","Transpose","Reshape","Constant",
    "Cast","Identity","Squeeze","Unsqueeze"
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--out", required=True)

    # 可选：权重 scale 来源（列方向 N 个数）
    p.add_argument("--weight-scales", default=None, help="npz，key=权重基准名（追溯到的 init/const 名），value 形状=(N,) 列方向的 per-channel scale；优先命中")
    # 可选：激活 per-tensor scale 来源（以张量名为 key）
    p.add_argument("--act-scales", default=None, help="npz，key=某算子输入0张量名，value=标量（或形如 (1,)）")

    # 没有 act-scales 时的默认常数 scale
    p.add_argument("--act-scale-default", type=float, default=0.02,
                   help="当没有提供 act-scales.npz 时，激活 per-tensor scale 的默认值（对称）")

    # 仅改写参数量>=该阈值的 2D 权重
    p.add_argument("--min-chunk", type=int, default=2048, dest="min_chunk")

    # 回溯最大步数
    p.add_argument("--max-hops", type=int, default=5)

    # 是否把算子输出也 Q（便于后续算子继续命中 INT8）
    p.add_argument("--quantize-outputs", action="store_true")

    # 冲突规避：已有 Q/DQ 的输入0 是否强行重插（默认不强行，检测到已有 Q->DQ 就跳过）
    p.add_argument("--force-rewrite-activation", action="store_true")

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def init_map(g): return {I.name: I for I in g.initializer}

def prod_map(g):
    m={}
    for n in g.node:
        for o in n.output: m[o]=n
    return m

def consumers_map(g):
    cm = {}
    for n in g.node:
        for i in n.input:
            if not i: continue
            cm.setdefault(i, []).append(n)
    return cm

def used_tensor_names(g):
    """只统计：所有节点的输入 + 图的输出"""
    names=set()
    for n in g.node:
        for i in n.input:
            if i: names.add(i)
    for o in g.output:
        names.add(o.name)
    return names

def as_const_arr(node):
    for a in node.attribute:
        if a.name=="value":
            arr = numpy_helper.to_array(a.t)
            if arr.dtype==np.float16: arr = arr.astype(np.float32)
            return arr.astype(np.float32)
    return None

def add_or_replace_init(g, name, arr):
    ten = numpy_helper.from_array(arr, name)
    im = init_map(g)
    if name in im:
        idx = list(g.initializer).index(im[name])
        g.initializer.remove(im[name]); g.initializer.insert(idx, ten)
    else:
        g.initializer.append(ten)

def prune_unused_inits(g):
    used = used_tensor_names(g)
    drop=[I for I in list(g.initializer) if I.name not in used]
    for I in drop: g.initializer.remove(I)
    return len(drop)

def prune_dead_nodes_iter(g, extra_ops=("Constant","Reshape","Transpose","Cast",
                                        "Identity","Squeeze","Unsqueeze",
                                        "QuantizeLinear","DequantizeLinear")):
    """迭代清理：若某节点的输出都不在任何节点输入/图输出中，则移除"""
    removed = 0
    while True:
        used = used_tensor_names(g)
        to_remove=[]
        for n in g.node:
            if n.op_type in extra_ops:
                outs_alive = any((o in used) for o in n.output)
                if not outs_alive:
                    to_remove.append(n)
        if not to_remove: break
        for n in to_remove:
            g.node.remove(n); removed += 1
    return removed

def insert_node_before(g, ref_node, new_node):
    for i in range(len(g.node)):
        if g.node[i] is ref_node:
            g.node.insert(i, new_node); return
    # 兜底：按输出名定位
    key = ref_node.output[0] if len(ref_node.output)>0 else None
    if key:
        for i in range(len(g.node)):
            if len(g.node[i].output)>0 and g.node[i].output[0]==key:
                g.node.insert(i, new_node); return
    g.node.extend([new_node])

def insert_node_after(g, ref_node, new_node):
    for i in range(len(g.node)):
        if g.node[i] is ref_node:
            g.node.insert(i+1, new_node); return
    g.node.extend([new_node])

def follow_back(g, start_name, im, prod, max_hops=5, verbose=False):
    """
    反向追踪权重输入，返回：
      ok, W_eff(np.float32), nodes_to_remove(list[node]), inits_to_remove(set[str]), base_key(str), note(str)
    允许穿越：Q/DQ/Transpose(perm=[1,0])/Reshape(const shape)/Cast/Identity/Squeeze/Unsqueeze/Constant
    """
    nodes_to_remove = []
    inits_to_remove = set()
    cur = start_name
    note = []
    hops = 0

    while hops <= max_hops:
        hops += 1
        if cur in im:
            W = numpy_helper.to_array(im[cur]).astype(np.float32)
            note.append(f"init:{cur}")
            return True, W, nodes_to_remove, inits_to_remove, cur, ";".join(note)
        if cur not in prod:
            return False, None, [], set(), None, "no_producer"
        n = prod[cur]
        if n.op_type not in ALLOW_OPS:
            return False, None, [], set(), None, f"producer={n.op_type}"

        if n.op_type == "Constant":
            arr = as_const_arr(n)
            if arr is None:
                return False, None, [], set(), None, "const_unsupported"
            nodes_to_remove.append(n)
            note.append(f"const:{n.name or cur}")
            return True, arr, nodes_to_remove, inits_to_remove, cur, ";".join(note)

        if n.op_type == "DequantizeLinear":
            nodes_to_remove.append(n); note.append("DQ")
            cur = n.input[0]; continue

        if n.op_type == "QuantizeLinear":
            # 记录其 scale/zero-point 初始值（若是 initializer，后续有机会清理）
            if len(n.input) > 1 and n.input[1] in im: inits_to_remove.add(n.input[1])
            if len(n.input) > 2 and n.input[2] in im: inits_to_remove.add(n.input[2])
            nodes_to_remove.append(n); note.append("Q")
            cur = n.input[0]; continue

        if n.op_type == "Transpose":
            perm = None
            for a in n.attribute:
                if a.name=="perm": perm = list(a.ints)
            if perm != [1,0]:
                return False, None, [], set(), None, f"transpose_perm={perm}"
            ok, W, rm_nodes, rm_inits, key, nt = follow_back(g, n.input[0], im, prod, max_hops-hops, verbose)
            if not ok: return False, None, [], set(), None, nt
            nodes_to_remove += rm_nodes + [n]
            inits_to_remove |= rm_inits
            W = W.transpose(1,0)
            note.append(f"T({nt})")
            return True, W, nodes_to_remove, inits_to_remove, key, ";".join(note)

        if n.op_type == "Reshape":
            if len(n.input) < 2:
                return False, None, [], set(), None, "reshape_no_shape"
            shp_t = n.input[1]
            if shp_t in im:
                shp = numpy_helper.to_array(im[shp_t]).astype(np.int64).tolist()
                inits_to_remove.add(shp_t)
            elif shp_t in prod and prod[shp_t].op_type == "Constant":
                arr = as_const_arr(prod[shp_t]).astype(np.int64)
                shp = arr.tolist(); nodes_to_remove.append(prod[shp_t])
            else:
                return False, None, [], set(), None, "reshape_shape_not_const"
            ok, W, rm_nodes, rm_inits, key, nt = follow_back(g, n.input[0], im, prod, max_hops-hops, verbose)
            if not ok: return False, None, [], set(), None, nt
            nodes_to_remove += rm_nodes + [n]
            inits_to_remove |= rm_inits
            try:
                W = W.reshape(shp)
            except Exception:
                return False, None, [], set(), None, "reshape_mismatch"
            note.append(f"R({nt})->{shp}")
            return True, W, nodes_to_remove, inits_to_remove, key, ";".join(note)

        if n.op_type in ("Cast","Identity","Squeeze","Unsqueeze"):
            prev = n.input[0] if n.input else None
            if prev is None:
                return False, None, [], set(), None, f"{n.op_type}_no_input"
            nodes_to_remove.append(n); note.append(n.op_type)
            cur = prev; continue

    return False, None, [], set(), None, "max_hops_exceeded"


def main():
    args = parse_args()
    model = onnx.load(args.onnx)
    g = model.graph
    im = init_map(g)
    prod = prod_map(g)
    cons = consumers_map(g)

    npz_w = np.load(args.weight_scales) if args.weight_scales else None
    npz_a = np.load(args.act_scales) if args.act_scales else None

    changed = 0
    removed_inits = set()
    removed_nodes = []

    def add_node_unique(lst, node):
        for x in lst:
            if x is node: return
        lst.append(node)

    def make_w_scales_cols(key, W_eff):
        """
        为 W.shape==(K,N) 的列（N 个输出通道）生成 per-channel 对称 scale。
        计算使用 axis=0（沿着行聚合，得到 (N,)），DQ.axis 用 1。
        """
        if npz_w is not None and key in npz_w.files:
            s = np.array(npz_w[key], dtype=np.float32).reshape(-1)
            s = np.maximum(s, 1e-12)
        else:
            s = np.max(np.abs(W_eff), axis=0).astype(np.float32)  # -> (N,)
            s = np.maximum(s/127.0, 1e-12)
        return s

    def has_qdq_already_on_tensor(tensor_name):
        """检测给定张量是否已经是 Q->DQ 的输出（简单启发式）：tensor 是某个 DQ 的输出，且该 DQ 的输入0来自一个 Q 的输出。"""
        if tensor_name not in prod: return False
        dq = prod[tensor_name]
        if dq.op_type != "DequantizeLinear": return False
        if len(dq.input) < 1: return False
        xq = dq.input[0]
        if xq not in prod: return False
        return prod[xq].op_type == "QuantizeLinear"

    def ensure_activation_qdq(input_name, node_for_insert, force=False):
        """
        对节点的 input_name 插入 per-tensor 对称 Q/DQ（若已存在 Q->DQ 则跳过，除非 force=True）。
        返回（新输入张量名，是否添加了 QDQ）
        """
        # 如果已是 Q->DQ 输出，直接复用
        if not force and has_qdq_already_on_tensor(input_name):
            return input_name, False

        # 构造常量 scale/zp
        if npz_a is not None and input_name in npz_a.files:
            a_scale = float(np.maximum(np.array(npz_a[input_name]).reshape(-1)[0], 1e-12))
        else:
            a_scale = float(max(abs(args.act_scale_default), 1e-12))

        sc_name = input_name + "_scale_pt"
        zp_name = input_name + "_zp_pt"
        xq_name = input_name + "_int8_q"
        x_dq_name = input_name + "_int8_dq"

        add_or_replace_init(g, sc_name, np.array([a_scale], dtype=np.float32))
        add_or_replace_init(g, zp_name, np.array([0], dtype=np.int8))

        q = helper.make_node(
            "QuantizeLinear",
            inputs=[input_name, sc_name, zp_name],
            outputs=[xq_name],
            name=input_name + "_Q_pt"
        )
        dq = helper.make_node(
            "DequantizeLinear",
            inputs=[xq_name, sc_name, zp_name],
            outputs=[x_dq_name],
            name=input_name + "_DQ_pt"
        )
        # 插到 node_for_insert 前（尽量靠近该算子）
        insert_node_before(g, node_for_insert, q)
        insert_node_before(g, node_for_insert, dq)
        return x_dq_name, True

    def optional_quantize_output(n):
        """可选：把算子输出也 Q（per-tensor，对称），便于下游继续命中 INT8。"""
        if not args.quantize_outputs: return
        y = n.output[0]
        y_scale_name = y + "_scale_pt"
        y_zp_name = y + "_zp_pt"
        y_q = y + "_int8_q"

        add_or_replace_init(g, y_scale_name, np.array([max(abs(args.act_scale_default), 1e-12)], dtype=np.float32))
        add_or_replace_init(g, y_zp_name, np.array([0], dtype=np.int8))

        q = helper.make_node(
            "QuantizeLinear",
            inputs=[y, y_scale_name, y_zp_name],
            outputs=[y_q],
            name=y+"_Q_pt"
        )
        insert_node_after(g, n, q)

    for n in list(g.node):
        if n.op_type not in ("Gemm","MatMul"): continue
        if len(n.input) < 2:  # 需要 A、W 两个输入
            continue

        # 回溯权重端（input[1]）
        Wn = n.input[1]
        ok, W_eff, rm_nodes, rm_inits, key, note = follow_back(
            g, Wn, im, prod, max_hops=args.max_hops, verbose=args.verbose
        )
        if not ok:
            if args.verbose: print(f"[skip] {n.op_type} {n.name or ''}: {note}")
            continue

        # ---- 规范 W_eff 到 (K,N) 并按列量化（N 为输出通道）----
        if W_eff.ndim != 2 or W_eff.size < args.min_chunk:
            if args.verbose: print(f"[skip] {n.op_type} {n.name or ''}: weight ndim/size not match")
            continue

        # Gemm 的 W 来自 input[1]，其 transB=1 时，原权重是 (N,K)，需要转成 (K,N)
        if n.op_type == "Gemm":
            transB = 0
            for a in n.attribute:
                if a.name=="transB": transB=a.i
            if transB == 1:
                W_eff = W_eff.transpose(1,0)

        K, N = W_eff.shape
        s = make_w_scales_cols(key, W_eff)  # (N,)
        if s.shape[0] != N:
            if args.verbose: print(f"[skip] {n.op_type} {n.name or ''}: scale len {s.shape[0]} != N {N}")
            continue
        Wq = np.clip(np.round(W_eff / s[None, :]), -128, 127).astype(np.int8)

        # 写入新 init + DQ(axis=1)，并替换权重输入
        wq = Wn + "_int8_eff"; sc = Wn + "_scale_perC"; zp = Wn + "_zp_perC"
        dq_out = Wn + "_dq_eff"
        add_or_replace_init(g, wq, Wq)
        add_or_replace_init(g, sc, s.astype(np.float32))
        add_or_replace_init(g, zp, np.zeros_like(s, dtype=np.int8))
        dq_w = helper.make_node("DequantizeLinear", inputs=[wq, sc, zp], outputs=[dq_out], name=Wn+"_DQ_w")
        dq_w.attribute.extend([helper.make_attribute("axis", 1)])  # 沿第2维（列）广播
        insert_node_before(g, n, dq_w)
        n.input[1] = dq_out

        # 若是 Gemm，清理 transB -> 0（我们已做过必要转置）
        if n.op_type == "Gemm":
            kept=[]
            for a in n.attribute:
                if a.name=="transB": kept.append(helper.make_attribute("transB", 0))
                else: kept.append(a)
            del n.attribute[:]; n.attribute.extend(kept)

        # 标记回溯路径上的节点/初始化器，稍后若确实未被使用则清理
        for t in rm_nodes: add_node_unique(removed_nodes, t)
        removed_inits |= set(rm_inits)
        if Wn in im: removed_inits.add(Wn)

        # ---- 激活端：对 input[0] 插入 per-tensor 对称 Q/DQ（A8）----
        a_name = n.input[0]
        a_new, added = ensure_activation_qdq(a_name, n, force=args.force_rewrite_activation)
        n.input[0] = a_new

        # ----（可选）对输出也 Q，便于链式 INT8 ----
        optional_quantize_output(n)

        changed += 1
        if args.verbose:
            print(f"[ok] {n.op_type} {n.name or ''}: W perC int8 (axis=1), A perT int8, key={key}, note={note}, shape={K}x{N}")

    # ---- 清理阶段（用“只看输入的 used 集合”）----
    # 1) 尝试删除我们标记的 initializer（若确实未被任何输入使用）
    used = used_tensor_names(g)
    im = init_map(g)
    for name in list(removed_inits):
        if name in im and name not in used:
            g.initializer.remove(im[name])

    # 2) 尝试删除我们标记的 nodes（若其输出不再被任何输入/图输出使用）
    used = used_tensor_names(g)
    for node in list(removed_nodes):
        alive = any((o in used) for o in node.output)
        if (not alive) and (node in g.node):
            g.node.remove(node)

    # 3) 迭代清理：删除一切“输出无人使用”的算子；再删未用 initializer
    dead_nodes_removed = prune_dead_nodes_iter(g)
    dropped_inits = prune_unused_inits(g)

    print(f"[Rewrite] converted W8A8 layers: {changed}")
    print(f"[Clean] removed unused initializers: {dropped_inits}, dead nodes: {dead_nodes_removed}")

    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print(f"[OK] Saved: {args.out}")

if __name__ == "__main__":
    main()

