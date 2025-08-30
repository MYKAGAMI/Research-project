# check_int8_onnx.py
import sys, onnx, numpy as np
from onnx import numpy_helper

m = onnx.load(sys.argv[1])
inits = {i.name: i for i in m.graph.initializer}

n_int8 = 0; n_fp32 = 0; bytes_int8 = 0; bytes_fp32 = 0
for it in inits.values():
    arr = numpy_helper.to_array(it)
    if arr.dtype == np.int8:
        n_int8 += 1; bytes_int8 += arr.nbytes
    if arr.dtype == np.float32:
        n_fp32 += 1; bytes_fp32 += arr.nbytes

print("Model:", sys.argv[1])
print("INT8 initializers:", n_int8, "bytes:", bytes_int8)
print("FP32 initializers:", n_fp32, "bytes:", bytes_fp32)
