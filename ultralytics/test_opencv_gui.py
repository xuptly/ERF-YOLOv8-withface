import cv2
import numpy as np

# 创建一个简单的窗口
cv2.namedWindow('Test Window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Test Window', 600, 600)

# 显示一个黑色图像
cv2.imshow('Test Window', np.zeros((600, 600, 3), np.uint8))

# 等待按键事件，按下 'q' 键退出
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()