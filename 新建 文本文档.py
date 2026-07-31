# from ultralytics import YOLO
#
# # 加载pt权重
# model = YOLO("best.pt")
#
# # 导出onnx
# # opset建议选12/16，部署兼容性好
# success = model.export(format="onnx", opset=16, simplify=True)
import onnx

onnx_model = onnx.load("best.onnx")

print("=====输入节点=====")
for inp in onnx_model.graph.input:
    print(inp.name)

print("\n=====输出节点=====")
for out in onnx_model.graph.output:
    print(out.name)