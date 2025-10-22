import torch
from ultralytics import YOLO

def export_model_to_onnx(pt_model_path, onnx_model_path, imgsz=640, opset_version=12):
    """
    将模型从.pt格式导出为ONNX格式。
    
    参数:
        pt_model_path (str): 输入的.pt模型文件路径。
        onnx_model_path (str): 输出的ONNX模型文件路径。
        imgsz (int): 输入图像的尺寸，默认为640。
        opset_version (int): ONNX opset版本，默认为12。
    """
    # 加载模型
    model = YOLO(pt_model_path)
    model = model.model  # 获取模型的PyTorch模块
    model.eval()  # 设置为评估模式

    # 创建虚拟输入张量
    dummy_input = torch.randn(1, 3, imgsz, imgsz)  # 输入图像大小为imgsz x imgsz

    # 导出为ONNX格式
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"ONNX模型已成功导出到 {onnx_model_path}")

pt_model_path = '/root/autodl-tmp/yolov8/ultralytics/runs/multi/yolopm0325/weights/best.pt'
onnx_model_path = 'best.onnx'
export_model_to_onnx(pt_model_path, onnx_model_path)