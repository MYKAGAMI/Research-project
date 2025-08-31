# make_onnx_int8_from_gemm_v3.py  (v3.4: fixed cleanup; only inputs count as "used")
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
    p.add_argument("--scales", default=None, help="可选：per-channel scales 的 npz（命中优先）")
    p.add_argument("--min-chunk", type=int, default=2048, dest="min_chunk",
                   help="仅改写参数量>=该阈值的 2D 权重")
    p.add_argument("--max-hops", type=int, default=5)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def init_map(g): return {I.name: I for I in g.initializer}
def prod_map(g):
    m={}
    for n in g.node:
        for o in n.output: m[o]=n
    return m

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
    key = ref_node.output[0] if len(ref_node.output)>0 else None
    if key:
        for i in range(len(g.node)):
            if len(g.node[i].output)>0 and g.node[i].output[0]==key:
                g.node.insert(i, new_node); return
    g.node.extend([new_node])

def follow_back(g, start_name, im, prod, max_hops=5, verbose=False):
    """
    反向追踪权重输入，返回：
      ok, W_eff(np.float32), nodes_to_remove(list[node]), inits_to_remove(set[str]), base_key(str), note(str)
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
    npz = np.load(args.scales) if args.scales else None

    changed = 0
    removed_inits = set()
    removed_nodes = []

    def add_node_unique(lst, node):
        for x in lst:
            if x is node: return
        lst.append(node)

    def make_scales(key, W_eff, axis_out):
        if npz is not None and key in npz.files:
            s = np.array(npz[key], dtype=np.float32).reshape(-1)
            s = np.maximum(s, 1e-12)
        else:
            s = np.max(np.abs(W_eff), axis=axis_out).astype(np.float32)
            s = np.maximum(s/127.0, 1e-12)
        return s

    for n in list(g.node):
        if n.op_type not in ("Gemm","MatMul"): continue

        if n.op_type == "Gemm":
            if len(n.input) < 2: continue
            B = n.input[1]
            transB = 0
            for a in n.attribute:
                if a.name=="transB": transB=a.i
            ok, W_eff, rm_nodes, rm_inits, key, note = follow_back(g, B, im, prod, max_hops=args.max_hops, verbose=args.verbose)
            if not ok:
                if args.verbose: print(f"[skip] Gemm {n.name or ''}: {note}")
                continue
            if W_eff.ndim != 2 or W_eff.size < args.min_chunk: continue
            if transB == 1: W_eff = W_eff.transpose(1,0)
            axis_out = 1
            s = make_scales(key, W_eff, axis_out)
            Wq = np.clip(np.round(W_eff / s[None,:]), -128, 127).astype(np.int8)

            wq = B + "_int8_eff"; sc = B + "_scale_perC"; zp = B + "_zp_perC"
            dq_out = B + "_dq_eff"
            add_or_replace_init(g, wq, Wq)
            add_or_replace_init(g, sc, s.astype(np.float32))
            add_or_replace_init(g, zp, np.zeros_like(s, dtype=np.int8))
            dq = helper.make_node("DequantizeLinear", inputs=[wq, sc, zp], outputs=[dq_out], name=B+"_DQ_w")
            dq.attribute.extend([helper.make_attribute("axis", axis_out)])
            insert_node_before(g, n, dq)
            n.input[1] = dq_out
            # transB 归零
            kept=[]
            for a in n.attribute:
                if a.name=="transB": kept.append(helper.make_attribute("transB", 0))
                else: kept.append(a)
            del n.attribute[:]; n.attribute.extend(kept)

            for t in rm_nodes: add_node_unique(removed_nodes, t)
            removed_inits |= set(rm_inits)
            if B in im: removed_inits.add(B)
            changed += 1
            if args.verbose: print(f"[ok] Gemm {n.name or ''} using {note} -> INT8 perC")

        else:  # MatMul
            if len(n.input) < 2: continue
            Wn = n.input[1]
            ok, W_eff, rm_nodes, rm_inits, key, note = follow_back(g, Wn, im, prod, max_hops=args.max_hops, verbose=args.verbose)
            if not ok:
                if args.verbose: print(f"[skip] MatMul {n.name or ''}: {note}")
                continue
            if W_eff.ndim != 2 or W_eff.size < args.min_chunk: continue
            axis_out = 1
            s = make_scales(key, W_eff, axis_out)
            Wq = np.clip(np.round(W_eff / s[None,:]), -128, 127).astype(np.int8)

            wq = Wn + "_int8_eff"; sc = Wn + "_scale_perC"; zp = Wn + "_zp_perC"
            dq_out = Wn + "_dq_eff"
            add_or_replace_init(g, wq, Wq)
            add_or_replace_init(g, sc, s.astype(np.float32))
            add_or_replace_init(g, zp, np.zeros_like(s, dtype=np.int8))
            dq = helper.make_node("DequantizeLinear", inputs=[wq, sc, zp], outputs=[dq_out], name=Wn+"_DQ_w")
            dq.attribute.extend([helper.make_attribute("axis", axis_out)])
            insert_node_before(g, n, dq)
            n.input[1] = dq_out

            for t in rm_nodes: add_node_unique(removed_nodes, t)
            removed_inits |= set(rm_inits)
            if Wn in im: removed_inits.add(Wn)
            changed += 1
            if args.verbose: print(f"[ok] MatMul {n.name or ''} using {note} -> INT8 perC")

    # ---- 清理阶段（用“只看输入的 used 集合”）----
    # 先尝试删除我们标记的 initializer（若确实未被任何输入使用）
    used = used_tensor_names(g)
    im = init_map(g)
    for name in list(removed_inits):
        if name in im and name not in used:
            g.initializer.remove(im[name])

    # 尝试删除我们标记的 nodes（若其输出不再被任何输入/图输出使用）
    used = used_tensor_names(g)
    for node in list(removed_nodes):
        alive = any((o in used) for o in node.output)
        if (not alive) and (node in g.node):
            g.node.remove(node)

    # 迭代清理：删除一切“输出无人使用”的算子；再删未用 initializer
    dead_nodes_removed = prune_dead_nodes_iter(g)
    dropped_inits = prune_unused_inits(g)

    print(f"[Rewrite] converted weights: {changed}")
    print(f"[Clean] removed unused initializers: {dropped_inits}, dead nodes: {dead_nodes_removed}")

    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print(f"[OK] Saved: {args.out}")

if __name__ == "__main__":
    main()







