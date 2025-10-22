import sys
sys.path.insert(0, "/root/autodl-tmp/yolov8/ultralytics")  # 将指定路径添加到Python sys路径中，告诉Python解释器在哪里可以找到你自定义的模块或包。
# 现在就可以导入Yolo类了，从Ultralytics库中导入YOLO类。
from ultralytics import YOLO

# Load a model：加载一个YOLO模型，可以根据.yaml文件构建新模型，加载预训练模型或者加载已有的权重。
model = YOLO(r'/root/autodl-tmp/yolov8/ultralytics/models/v8/CSPHet+ACmix.yaml', task='multi')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model；训练模型，包括提供数据集路径、批处理大小、迭代次数、图像尺寸、设备信息、模型名称、验证、指定任务类型、定义模型需要识别的类别列表、combine_class将类别组合成一个类别、单类别训练模式等设置。
#model.train(data=r'/root/yolov8/ultralytics/datasets/bdd-multi.yaml', batch=12, epochs=300, imgsz=(640,640), device=[0], name='yolopm', val=True, task='multi',classes=[2,3,4,9,10,11],combine_class=[2,3,4,9],single_cls=True)
model.train(data=r'/root/autodl-tmp/yolov8/datasets/bdd-multi.yaml', batch=16, epochs=100, imgsz=(640,640), device=[0], name='yolopm0503', val=True, task='multi',classes=[2,3,4,9,10,11],combine_class=[2,3,4,9],single_cls=True)

