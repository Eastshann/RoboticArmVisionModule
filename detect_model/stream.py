import cv2

from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO(r"./detect_model/weights/best.pt")

# 打开视频文件
# cap = cv2.VideoCapture("path/to/your/video/file.mp4")
# 或使用设备“0”打开视频捕获设备读取帧
cap = cv2.VideoCapture(0)

# 设置视频帧大小
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

title = "YOLOv8 Inference"
# 设置窗口位置
cv2.namedWindow(title, cv2.WINDOW_NORMAL)
cv2.moveWindow(title, 640, 640)

# 循环播放视频帧
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()
    if success:
        # 在框架上运行 YOLOv11 推理
        results = model(frame)
        # 在框架上可视化结果
        annotated_frame = results[0].plot()
        # 显示带标注的框架
        cv2.imshow(title, annotated_frame)
        # 如果按下“q”，则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果到达视频末尾，则中断循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()
