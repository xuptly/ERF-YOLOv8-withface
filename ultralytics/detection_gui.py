import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
from ultralytics import YOLO

# 设置环境变量解决 Qt platform plugin 问题
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/opt/conda/lib/python3.8/site-packages/cv2/qt/plugins"
os.environ["DISPLAY"] = ":99"  # 设置虚拟显示

class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = YOLO('/root/autodl-tmp/yolov8/ultralytics/runs/multi/yolopm0407/weights/best.pt')
        self.image_path = None
        self.is_paused = False
        self.initUI()

    def initUI(self):
        self.setWindowTitle('自动驾驶多任务环境感知系统')
        self.setGeometry(100, 100, 1200, 800)

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 标题
        title = QLabel('自动驾驶多任务环境感知系统')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        layout.addWidget(title)

        # 图片显示区域
        content = QHBoxLayout()
        
        # 左侧图片选择区域
        self.left_panel = QGroupBox("输入图片")
        left_layout = QVBoxLayout()
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: white;")
        self.image_label.setMinimumSize(600, 600)
        
        btn_load = QPushButton("选择图片")
        btn_load.clicked.connect(self.load_image)
        
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(btn_load)
        self.left_panel.setLayout(left_layout)
        
        # 右侧结果区域
        self.right_panel = QGroupBox("检测结果")
        right_layout = QVBoxLayout()
        
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("background-color: white;")
        self.result_label.setMinimumSize(600, 600)
        
        right_layout.addWidget(self.result_label)
        self.right_panel.setLayout(right_layout)
        
        content.addWidget(self.left_panel)
        content.addWidget(self.right_panel)
        layout.addLayout(content)

        # 控制按钮
        control_layout = QHBoxLayout()
        self.btn_start = QPushButton("开始检测")
        self.btn_pause = QPushButton("暂停")
        btn_exit = QPushButton("退出")
        
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_pause.clicked.connect(self.toggle_pause)
        btn_exit.clicked.connect(self.close)
        
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_pause)
        control_layout.addWidget(btn_exit)
        layout.addLayout(control_layout)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg)")
        if path:
            self.image_path = path
            pixmap = QPixmap(path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), 
                                                   Qt.KeepAspectRatio,
                                                   Qt.SmoothTransformation))

    def start_detection(self):
        if not self.image_path:
            QMessageBox.warning(self, "警告", "请先选择图片！")
            return

        self.btn_start.setEnabled(False)
        
        # 创建线程和工作对象
        self.thread = QThread()
        self.worker = DetectionWorker(self.model, self.image_path)
        
        # 连接信号和槽
        self.worker.update_signal.connect(self.show_result)
        self.worker.finished.connect(self.detection_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.started.connect(self.worker.run)
        
        # 启动线程
        self.worker.moveToThread(self.thread)
        self.thread.start()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.btn_pause.setText("继续" if self.is_paused else "暂停")

    def show_result(self, result_path):
        if not self.is_paused:
            pixmap = QPixmap(result_path)
            self.result_label.setPixmap(pixmap.scaled(self.result_label.size(),
                                                    Qt.KeepAspectRatio,
                                                    Qt.SmoothTransformation))

    def detection_finished(self):
        self.btn_start.setEnabled(True)
        if hasattr(self, 'thread'):
            self.thread.quit()
            self.thread.wait()

class DetectionWorker(QObject):
    update_signal = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, model, image_path, parent=None):
        super().__init__(parent)
        self.model = model
        self.image_path = image_path

    def run(self):
        results = self.model.predict(source=self.image_path, 
                                   imgsz=(640, 600),
                                   device=[0],
                                   save=True,
                                   conf=0.25,
                                   iou=0.45,
                                   show_labels=False)
        
        result_path = results[0].save_dir + "/" + self.image_path.split("/")[-1]
        self.update_signal.emit(result_path)
        self.finished.emit()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DetectionApp()
    ex.show()
    sys.exit(app.exec_())