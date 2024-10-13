import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import sqlite3
import numpy as np
import annoy


class CAE(nn.Module):
    def __init__(self, initial_channels=32, hidden_dim=128, bottleneck_dim=64):
        super(CAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, initial_channels, kernel_size=3, stride=2, padding=1),
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
            nn.ConvTranspose2d(initial_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def prepare_data(batch_size):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_set = FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),  # 将图片转换为单通道
        transforms.Resize((img_size, img_size)),  # 调整图片大小
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # 增加一个批次维度
    return image


def train(batch_size,learning_rate,num_epochs):
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
        torch.save(model.state_dict(), f"resae_epoch_{epoch + 1}.pth")
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
    #print("Encoded feature vector:", encoded.cpu().numpy())
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


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def find_closest_encoded(device, model, image_path, db_conn):
    model.eval()
    image = load_image(image_path).to(device)
    with torch.no_grad():
        encoded, _ = model(image)
    encoded = encoded.cpu().numpy()

    # Load data from the database
    data = load_all_encoded_from_db(db_conn)
    index = annoy.AnnoyIndex(encoded.shape[1], 'euclidean')  # Create Annoy index

    # Add data to Annoy index
    for i, record in enumerate(data):
        id, name, encoded_blob = record
        encoded_db = np.frombuffer(encoded_blob, dtype=np.float32)
        index.add_item(i, encoded_db)

    index.build(10)  # Build the tree with 10 trees

    # Query the index for the nearest neighbor
    nearest = index.get_nns_by_vector(encoded.flatten(), 1)[0]
    closest_record = data[nearest]
    return closest_record[0], closest_record[1]


def load_test_data_to_db():
    db_conn = open_db("encoded_data.db")
    model.load_state_dict(torch.load("resae_epoch_10.pth"))
    for i,s in enumerate(["a","b"]):
        encoded = encode(device, model, f"data/{s}1.jpg")
        save_encoded_to_db(db_conn, encoded, i, s)
    db_conn.close()


def predict():
    db_conn = open_db("encoded_data.db")
    model.load_state_dict(torch.load("resae_epoch_10.pth"))
    while(True):
        img_path = input("img_path: ")
        print(find_closest_encoded(device, model, img_path, db_conn))
    #db_conn.close()


if __name__ == "__main__":
    img_size = 32
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = CAE(initial_channels=4, hidden_dim=64, bottleneck_dim=32).to(device)

    #train(16,0.001,10)
    #load_test_data_to_db()
    predict()