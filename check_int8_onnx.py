# check_int8_onnx.py
import sys, onnx, numpy as np
from onnx import numpy_helper
m = onnx.load(sys.argv[1])
n_i8=n_f32=0; b_i8=b_f32=0
for it in m.graph.initializer:
    arr = numpy_helper.to_array(it)
    if arr.dtype==np.int8: n_i8+=1; b_i8+=arr.nbytes
    if arr.dtype==np.float32: n_f32+=1; b_f32+=arr.nbytes
print("Model:", sys.argv[1])
print("INT8 initializers:", n_i8, "bytes:", b_i8)
print("FP32 initializers:", n_f32, "bytes:", b_f32)
