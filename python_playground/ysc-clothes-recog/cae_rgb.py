import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import sqlite3
import numpy as np
import annoy
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

class CAE(nn.Module):
    def __init__(self, initial_channels=32, hidden_dim=128, bottleneck_dim=64, img_size=32):  # 修改图像大小为256
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, initial_channels, kernel_size=3, stride=2, padding=1),  # 输入通道从1改为3
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(),
            nn.Conv2d(initial_channels, initial_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(initial_channels * 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((initial_channels * 2) * (img_size // 4) * (img_size // 4), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (initial_channels * 2) * (img_size // 4) * (img_size // 4)),
            nn.ReLU(),
            nn.Unflatten(1, (initial_channels * 2, img_size // 4, img_size // 4)),
            nn.ConvTranspose2d(initial_channels * 2, initial_channels, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(initial_channels, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出通道从1改为3
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def prepare_data(batch_size, img_size=32):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = ImageFolder(root='data/DeepFashion2/train', transform=transform)
    test_set = ImageFolder(root='data/DeepFashion2/test', transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_image(image_path,img_size=32):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 调整图片大小
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # 增加一个批次维度
    return image

def prepare_data(batch_size, img_size=32):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = ImageFolder(root='data/train', transform=transform)
    test_set = ImageFolder(root='data/test', transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train(batch_size, learning_rate, num_epochs):
    train_loader, test_loader = prepare_data(batch_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data, _ in tqdm(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            encoded, decoded = model(data)
            loss = criterion(decoded, data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        # Save model after each epoch
        torch.save(model.state_dict(), f"cae_epoch_{epoch + 1}.pth")
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.show()
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            encoded, decoded = model(data)
            test_loss += criterion(decoded, data).item()
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

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

def load_ysc_test_data_to_db():
    db_conn = open_db("encoded_data.db")
    model.load_state_dict(torch.load("cae_epoch_20.pth"))
    for i in range(1,5):
        encoded = encode(device, model, f"data/a{i}.jpg")
        save_encoded_to_db(db_conn, encoded, i, f"汉服{i}")
    db_conn.close()

def predict():
    db_conn = open_db("encoded_data.db")
    model.load_state_dict(torch.load("cae_epoch_20.pth"))
    while True:
        img_path = input("img_path: ")
        a, b, distances = find_closest_encoded(device, model, img_path, db_conn)
        print(f"a={a}\nb={b}\ndistances={distances}")

if __name__ == "__main__":
    # img_size = 32
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = CAE(initial_channels=256, hidden_dim=1024, bottleneck_dim=512).to(device)

    #train(32,0.01,20)
    #load_ysc_test_data_to_db()
    predict()