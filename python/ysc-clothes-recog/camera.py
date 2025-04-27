import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import annoy
from auto_encoder import load_all_encoded_from_db, CAE, open_db


def load_image_from_cam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("can't open camera")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("can't read from camera")
        return None
    # Convert the image from BGR to RGB
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 显示捕获的图片
    cv2.imshow('Captured Image', frame)

    return frame


def preprocess_image_from_cam(image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.fromarray(image)
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def predict_by_cam(device, model, db_conn):
    model.eval()
    while True:
        frame = load_image_from_cam()
        if frame is None:
            continue
        image = preprocess_image_from_cam(frame).to(device)
        with torch.no_grad():
            encoded, _ = model(image)
        encoded = encoded.cpu().numpy()
        # Load data from the database
        data = load_all_encoded_from_db(db_conn)
        index = annoy.AnnoyIndex(encoded.shape[1], 'euclidean')  # Create Annoy index
        for i, record in enumerate(data):
            id, name, encoded_blob = record
            encoded_db = np.frombuffer(encoded_blob, dtype=np.float32)
            index.add_item(i, encoded_db)
        index.build(10)
        nearest = index.get_nns_by_vector(encoded.flatten(), 1)[0]
        closest_record = data[nearest]
        print(f"Closest match ID: {closest_record[0]}, Name: {closest_record[1]}")
        # To exit the loop, you can add a condition, e.g., pressing 'q'
        if input("Press 'q' to quit or any other key to continue: ") == 'q':
            break


if __name__ == "__main__":
    img_size = 32
    device = torch.device("cpu")
    model = CAE(initial_channels=4, hidden_dim=64, bottleneck_dim=32).to(device)
    db_conn = open_db("encoded_data.db")
    predict_by_cam(device, model, db_conn)
