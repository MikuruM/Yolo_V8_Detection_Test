# 导入必要的库
from ultralytics import YOLO  # 导入 YOLO 模型
import torch
import cv2  # 导入 OpenCV 库
import numpy as np
import mss  # 导入 mss 库用于屏幕截图
import time

# 指定设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'使用的设备：{device}')  # 打印使用的设备

# 加载预训练的 YOLOv8 模型并移动到指定设备
model = YOLO('yolov8s.pt').to(device)  # 加载 YOLOv8 nano 模型

# 设置置信度阈值
model.conf = 0.5  # 只保留置信度大于 0.5 的检测结果

# 设置要检测的类别（可选）
# model.classes = list(range(0, 80))  # 检测所有类别

# 初始化 mss
sct = mss.mss()

# 获取所有显示器的信息
monitors = sct.monitors
print("可用的显示器：")
for i, m in enumerate(monitors):
    if i == 0:
        print(f"虚拟显示器（所有显示器的组合）：{i}: {m}")
    else:
        print(f"显示器 {i}: {m}")

# 选择要截取的显示器
monitor_number = 1  # 更改此值以选择不同的显示器，1 表示第一个物理显示器

if monitor_number in range(len(monitors)):
    monitor = monitors[monitor_number]
else:
    print(f"显示器 {monitor_number} 不存在，默认使用第一个显示器")
    monitor = monitors[1] #0为两个显示器，1或2为单个显示器

print(f"正在截取显示器 {monitor_number}: {monitor}")

# 您还可以自定义截取区域
# 例如，只截取指定显示器的一部分
monitor = {
    "top": monitor["top"] + 240,
    "left": monitor["left"] + 160,
    "width": 1100,
    "height": 700
}

# 设置识别帧率（通过控制循环间隔实现）
fps = 30  # 设置每秒捕获和处理的帧数
interval = 1 / fps

while True:
    start_time = time.time()

    # 捕获屏幕截图
    screenshot = sct.grab(monitor)  # 截取指定显示器的屏幕
    img = np.array(screenshot)  # 将截图转换为 NumPy 数组
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 转换颜色格式（去除透明度通道）

    # 对截图进行物体检测，指定设备
    results = model(img, device=device)

    # 获取检测结果
    detections = results[0]

    # 自定义绘制检测结果
    annotated_frame = img.copy()  # 复制原始截图
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
        color = (255, 0, 0)  # 颜色
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
    cv2.imshow(f'YOLOv8 屏幕实时检测 - 显示器 {monitor_number}', annotated_frame)  # 显示带有检测结果的截图

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 控制帧率
    elapsed_time = time.time() - start_time
    if elapsed_time < interval:
        time.sleep(interval - elapsed_time)

# 释放资源
cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
