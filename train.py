from ultralytics import YOLO

if __name__ == '__main__':
    # 加载 YOLOv8 模型
    model = YOLO('yolov8s.pt')

    # 使用原始字符串指定数据路径，避免路径解析问题
    model.train(data=r'D:\PyCharm\Detect\train\data.yaml', epochs=100, imgsz=320, batch=32, amp=True, workers=12, cache=True)


