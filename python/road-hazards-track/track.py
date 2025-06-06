from ultralytics import YOLO
import cv2

model = YOLO('E:\\repos\\playground\\python_playground\\road-hazards-track\\runs\\detect\\train8\\weights\\best.pt')
results = model.track('c.mp4', tracker='bytetrack.yaml')

# 打开输入视频
cap = cv2.VideoCapture('c.mp4')

# 获取视频帧宽度和高度
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 定义输出视频编解码器和创建 VideoWriter 对象
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))

# 遍历视频帧，进行处理并保存结果
for result in results:
    frame = result.orig_img

    # 获取识别框、识别类型和追踪 ID
    boxes = result.boxes
    labels = result.names

    # 在帧上绘制识别框、识别类型和追踪 ID
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取边界框坐标
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制边界框
        cv2.putText(frame, f'label: {label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                    2)  # 绘制标签和 ID

    out.write(frame)

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
