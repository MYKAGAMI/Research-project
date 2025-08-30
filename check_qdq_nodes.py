# check_qdq_nodes.py
import sys, onnx
m = onnx.load(sys.argv[1])
qs = sum(1 for n in m.graph.node if n.op_type=="QuantizeLinear")
dqs = sum(1 for n in m.graph.node if n.op_type=="DequantizeLinear")
print(sys.argv[1], "QuantizeLinear:", qs, "DequantizeLinear:", dqs)
