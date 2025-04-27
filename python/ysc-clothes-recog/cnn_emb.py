import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import sqlite3
import numpy as np
import annoy

def load_image(path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图像
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(path)
    image = transform(image).unsqueeze(0)  # 增加batch维度
    return image

class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def encode(device, model, image_path):
    model.eval()
    image = load_image(image_path).to(device)
    with torch.no_grad():
        encoded, _ = model(image)
    return encoded

def open_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS EncodedData (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            encoded BLOB NOT NULL
        )
    ''')
    conn.commit()
    return conn

def save_encoded_to_db(db_conn, encoded, id, name):
    cursor = db_conn.cursor()
    encoded_blob = np.array(encoded).tobytes()
    cursor.execute("INSERT INTO EncodedData (id, name, encoded) VALUES (?, ?, ?)", (id, name, encoded_blob))
    db_conn.commit()

def load_all_encoded_from_db(db_conn):
    cursor = db_conn.cursor()
    cursor.execute("SELECT id, name, encoded FROM EncodedData")
    data = cursor.fetchall()
    return data

def load_ysc_test_data_to_db():
    db_conn = open_db("encoded_data.db")
    model.load_state_dict(torch.load("cae_epoch_20.pth"))
    for i in range(1,5):
        encoded = encode(device, model, f"data/a{i}.jpg")
        save_encoded_to_db(db_conn, encoded, i, f"汉服{i}")
    db_conn.close()

def find_closest_encoded(device, model, image_path, db_conn):
    model.eval()
    image = load_image(image_path).to(device)
    with torch.no_grad():
        encoded, _ = model(image)
    encoded = encoded.cpu().numpy()
    data = load_all_encoded_from_db(db_conn)
    index = annoy.AnnoyIndex(encoded.shape[1], 'euclidean')
    distances = []
    for i, record in enumerate(data):
        id, name, encoded_blob = record
        encoded_db = np.frombuffer(encoded_blob, dtype=np.float32)
        index.add_item(i, encoded_db)
        distance = np.linalg.norm(encoded.flatten() - encoded_db)
        distances.append((id, name, distance))
    index.build(10)
    nearest = index.get_nns_by_vector(encoded.flatten(), 1)[0]
    closest_record = data[nearest]
    distances.sort(key=lambda x: x[2])
    return closest_record[0], closest_record[1], distances

def predict():
    db_conn = open_db("encoded_data.db")
    model.load_state_dict(torch.load("cae_epoch_20.pth"))
    while True:
        img_path = input("img_path: ")
        a, b, distances = find_closest_encoded(device, model, img_path, db_conn)
        print(f"a={a}\nb={b}\ndistances={distances}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型权重
model = FashionMNISTModel()
model.load_state_dict(torch.load('fashion_mnist_model.pth'))
model.eval()

# 去掉输出层
model = nn.Sequential(*list(model.children())[:-1])

# 读取图像并进行预处理
image = load_image("data/1.jpg")

# 获取嵌入
with torch.no_grad():
    embedding = model(image)
    print('Embedding:', embedding.numpy())
