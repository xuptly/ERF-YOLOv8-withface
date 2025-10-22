import sys
sys.path.insert(0, r'/root/autodl-tmp/yolov8/ultralytics')
# The same with train, you should change the path to yours.
# 现在就可以导入Yolo类了
from ultralytics import YOLO

# 验证模型
# model = YOLO('yolov8s-seg.pt')
#################### liuya open
number = 3 #input how many tasks in your work                   # 打开注释了
####################

# model = YOLO('/home/jiayuan/ultralytics-main/ultralytics/runs/best.pt')  # 加载自己训练的模型# Validate the model
# model = YOLO('/home/jiayuan/ultralytics-main/ultralytics/runs/best.pt')  # 加载自己训练的模型# Validate the model
model = YOLO('/root/autodl-tmp/yolov8/ultralytics/runs/multi/yolopm0407/weights/best.pt')

# metrics = model.val(data='/home/jiayuan/ultralytics-main/ultralytics/datasets/bdd-multi.yaml',device=[4],task='multi',name='v3-model-val',iou=0.6,conf=0.001, imgsz=(640,640),classes=[2,3,4,9,10,11],combine_class=[2,3,4,9],single_cls=True)  # no arguments needed, dataset and settings remembered

# metrics = model.val(data='/home/jiayuan/ultralytics-main/ultralytics/datasets/bdd-multi.yaml',device=[3],task='multi',name='val',iou=0.6,conf=0.001, imgsz=(640,640),classes=[2,3,4,9,10,11],combine_class=[2,3,4,9],single_cls=True)  # no arguments needed, dataset and settings remembered
metrics = model.val(data='/root/autodl-tmp/yolov8/datasets/bdd-multi.yaml',device=[0],task='multi',name='val04272',iou=0.6,conf=0.001, imgsz=(640,640),classes=[2,3,4,9,10,11],combine_class=[2,3,4,9],single_cls=True,speed=True)  # no arguments needed, dataset and settings remembered


#################### liuya open
for i in range(number):                     # 打开注释了
    print(f'This is for {i} work')          # 打开注释了
    print(metrics[i].box.map)    # map50-95 # 打开注释了
    print(metrics[i].box.map50)  # map50    # 打开注释了
    print(metrics[i].box.map75)  # map75    # 打开注释了
    print(metrics[i].box.maps)   # a list contains map50-95 of each category    # 打开注释了
####################