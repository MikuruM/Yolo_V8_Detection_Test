# 导入必要的库
from ultralytics import YOLO  # 导入 YOLO 模型
import torch
import cv2  # 导入 OpenCV 库

# 指定设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'使用的设备：{device}')  # 打印使用的设备

# 加载预训练的 YOLOv8 模型并移动到指定设备
model = YOLO('yolov8s.pt').to(device)  # 加载 YOLOv8 nano 模型

# 设置置信度阈值
model.conf = 0.5  # 只保留置信度大于 0.5 的检测结果

# 设置要检测的类别（可选）
# 如果只想检测特定的类别，例如 'person' 和 'car'
# 请将类别的索引添加到 classes 列表中，类别索引可参考 COCO 数据集的类别索引
model.classes = list(range(0, 70))  # 0: 'person', 2: 'car'

# 打开摄像头，并设置摄像头参数
cap = cv2.VideoCapture(0)  # 使用 0 打开默认摄像头

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 设置摄像头分辨率（可选）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 设置高度

# 设置识别帧率（通过跳帧实现）
frame_skip = 1  # 每隔多少帧进行一次识别，1 表示每帧都识别

frame_count = 0  # 帧计数器

while True:
    ret, frame = cap.read()  # 读取摄像头帧
    if not ret:
        print("无法读取摄像头帧")
        break

    # 仅在指定的帧进行识别
    if frame_count % frame_skip == 0:
        # 对帧进行物体检测，指定设备
        results = model(frame, device=device)
        # 获取检测结果
        detections = results[0]
    else:
        # 如果未识别，保持之前的检测结果
        pass

    frame_count += 1

    # 自定义绘制检测结果
    annotated_frame = frame.copy()  # 复制原始帧
    for box in detections.boxes:
        # 提取边界框坐标和类别信息
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf.cpu().numpy()[0]
        cls = int(box.cls.cpu().numpy()[0])
        label = model.names[cls]

        # 仅绘制置信度大于阈值的检测结果
        if conf < model.conf:
            continue

        # 自定义颜色和粗细
        color = (255, 0, 0) #颜色
        thickness = 2  # 线条粗细

        # 绘制边界框
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

        # 显示类别和置信度
        #label_text = f"{label} {conf:.2f}"
        label_text = f"{label}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
        text_x = x1
        text_y = y1 - 5 if y1 - 5 > 15 else y1 + 15

        # 绘制背景框
        cv2.rectangle(annotated_frame, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), color, -1)
        # 绘制文字
        cv2.putText(annotated_frame, label_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    # 显示结果
    cv2.imshow('YOLOv8 实时检测', annotated_frame)  # 显示带有检测结果的帧

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()  # 释放摄像头资源
cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
