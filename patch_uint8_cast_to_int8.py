import onnx
from onnx import TensorProto

def main(in_path, out_path, verbose=True):
    m = onnx.load(in_path)
    g = m.graph
    fixed = 0
    for n in g.node:
        if n.op_type == "Cast":
            for a in n.attribute:
                if a.name == "to" and a.i == TensorProto.UINT8:
                    a.i = TensorProto.INT8
                    fixed += 1
                    if verbose:
                        print(f"[fix] Cast -> INT8 : {n.name or '(no-name)'}")
    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    print(f"[OK] Saved: {out_path}, patched Cast count: {fixed}")

if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 3, "Usage: python patch_uint8_cast_to_int8.py <in.onnx> <out.onnx>"
    main(sys.argv[1], sys.argv[2])
