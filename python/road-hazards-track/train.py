from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO()
    model.train(data='datasets/landslide-full-fixed/data.yaml', epochs=200, batch=16, imgsz=640)