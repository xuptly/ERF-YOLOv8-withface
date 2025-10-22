import sys

sys.path.insert(0, r'D:\YOLOPv2\ERF-YOLOv8\ultralytics')

from ultralytics import YOLO


number = 3 #input how many tasks in your work
model = YOLO(r'D:\YOLOPv2\ERF-YOLOv8\ultralytics\runs\multi\yolopm0407\weights\best.pt')  # Validate the model,best.pt
model.predict(source=r'D:\YOLOPv2\ERF-YOLOv8-withface\ultralytics\assets', imgsz=(640, 640), device=[0], name='predict1011', save=True, conf=0.25, iou=0.45, show_labels=False)
