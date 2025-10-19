# patch_asym_to_sym_int8.py
import onnx
import numpy as np
from onnx import numpy_helper, TensorProto

def to_array(init):
    return numpy_helper.to_array(init)

def make_int8_zeros_like(init):
    # 生成与 init 相同形状的 int8 全 0 initializer
    shape = list(to_array(init).shape)
    arr = np.zeros(shape, dtype=np.int8)
    return numpy_helper.from_array(arr, name=init.name + "_i8_zeros")

def replace_initializer(model, name, new_init):
    # 用 new_init 替换名为 name 的 initializer；找不到则追加
    g = model.graph
    for i, init in enumerate(g.initializer):
        if init.name == name:
            g.initializer.remove(init)
            break
    g.initializer.extend([new_init])

def get_initializer(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return init
    return None

def ensure_int8_zero_point(model, zp_name):
    """
    确保名为 zp_name 的 zero-point 是 int8 且全 0；如果是 initializer 就替换；
    如果不是 initializer（来自算子输出），则新建一个 Constant initializer 并返回其名字。
    """
    zp_init = get_initializer(model, zp_name)
    if zp_init is None:
        # 不是 initializer：新建 1 标量 int8 零点并返回新名字
        new_name = zp_name + "_i8_zeros_const"
        arr = np.zeros((), dtype=np.int8)
        new_init = numpy_helper.from_array(arr, name=new_name)
        model.graph.initializer.extend([new_init])
        return new_name

    arr = to_array(zp_init)
    need_replace_dtype = arr.dtype != np.int8
    need_replace_value = np.any(arr != 0)

    if need_replace_dtype or need_replace_value:
        # 用同形状 int8 全 0 替换
        new_init = make_int8_zeros_like(zp_init)
        new_init.name = zp_init.name  # 直接沿用原名字，避免改 node 输入名
        replace_initializer(model, zp_init.name, new_init)
    else:
        # 已经是 int8 全 0，无需改
        pass
    return zp_name

def patch_qdq_zero_points_to_int8_zero(model):
    g = model.graph
    changed = 0
    for node in g.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            # Q/DQ 输入约定：x, scale, zero_point
            # 有些图里可能缺少第三个输入（默认为 0），这里仅在存在时处理
            if len(node.input) >= 3:
                zp_name = node.input[2]
                new_name = ensure_int8_zero_point(model, zp_name)
                if new_name != zp_name:
                    node.input[2] = new_name
                    changed += 1
    return changed

def patch_all_uint8_cast_to_int8(model):
    # 额外保险：把任何 Cast(to=UINT8) 改为 Cast(to=INT8)
    CHANGED = 0
    for node in model.graph.node:
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == TensorProto.UINT8:
                    a.i = TensorProto.INT8
                    CHANGED += 1
    return CHANGED

def main(in_path, out_path):
    print(f"Loading: {in_path}")
    model = onnx.load(in_path)

    c1 = patch_all_uint8_cast_to_int8(model)
    c2 = patch_qdq_zero_points_to_int8_zero(model)

    # 简单一致性检查：是否仍存在 uint8 / 非零 zp
    remain_uint8_cast = 0
    for node in model.graph.node:
        if node.op_type == "Cast":
            for a in node.attribute:
                if a.name == "to" and a.i == TensorProto.UINT8:
                    remain_uint8_cast += 1

    remain_bad_zp = 0
    for node in model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear") and len(node.input) >= 3:
            init = get_initializer(model, node.input[2])
            if init is not None:
                arr = to_array(init)
                if arr.dtype != np.int8 or np.any(arr != 0):
                    remain_bad_zp += 1

    print(f"Patched UINT8 Cast -> INT8: {c1}")
    print(f"Patched Q/DQ zero-point to int8 zeros: {c2}")
    print(f"Remaining UINT8 Cast: {remain_uint8_cast}")
    print(f"Remaining non-symmetric zero-points: {remain_bad_zp}")

    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python patch_asym_to_sym_int8.py in.onnx out.onnx")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
