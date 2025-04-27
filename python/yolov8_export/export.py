from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

# Use the model
path = model.export(format="onnx")  # export the model to ONNX format
print(path)