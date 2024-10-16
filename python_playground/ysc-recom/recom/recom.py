import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import mysql.connector
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import numpy as np
import sqlite3
from flask import Flask, jsonify, request, render_template


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, fc_hidden_dim, dropout_rate):
        super(Autoencoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(GATConv(input_dim, hidden_dim, heads=num_heads, concat=True))
        for _ in range(num_layers - 1):
            self.encoder.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim * num_heads, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, hidden_dim * num_heads)
        self.decoder = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.decoder.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True))
        self.decoder.append(GATConv(hidden_dim * num_heads, input_dim, heads=num_heads, concat=False))

    def forward(self, x, edge_index):
        for layer in self.encoder:
            x = layer(x, edge_index)
            x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        for layer in self.decoder:
            x = layer(x, edge_index)
        return x

db_config = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_DATABASE')
}


def get_data_from_db():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT hanfu_id, image, CONCAT(shop_name, ' ' , label , ' ', name, ' ', price) AS info FROM hanfu")
    items = cursor.fetchall()
    cursor.close()
    connection.close()
    item_ids = [item['hanfu_id'] for item in items]
    item_pic_urls = [item['image'] for item in items]
    item_titles = [item['info'] for item in items]
    item_ids_dict = {item_id: index for index, item_id in enumerate(item_ids)}
    num_items = len(item_ids)
    row = []
    col = []
    for i in range(num_items):
        for j in range(i + 1, num_items):
            row.append(i)
            col.append(j)
            row.append(j)
            col.append(i)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    return item_ids,item_pic_urls, item_titles, item_ids_dict, edge_index


def embed_text(bert_tokenizer,bert_model,text):
    inputs = bert_tokenizer(text, return_tensors='pt')
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def embed_image(resnet_model,image_path):
    #input_image = Image.open(image_path).convert('RGB')
    # for test only
    input_image = Image.open("C:\\Users\\lain\\Pictures\\wallhaven-x6m7dl_2560x1440.png").convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image).unsqueeze(0)  # create a mini-batch as expected by the model
    with torch.no_grad():
        embedding = resnet_model(input_tensor)
    return embedding


def pre_embedding(item_pic_urls, item_titles):
    # Load BERT and ResNet50 models
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    resnet_model = models.resnet50(pretrained=True)
    resnet_model.eval()
    resnet_model.fc = nn.Identity()  # Remove the classification layer
    text_embeddings = [embed_text(bert_tokenizer,bert_model,text) for text in item_titles]
    image_embeddings = [embed_image(resnet_model,img_path) for img_path in item_pic_urls]
    combined_embeddings = [torch.cat((text_emb, img_emb), dim=1) for text_emb, img_emb in
                           zip(text_embeddings, image_embeddings)]
    return combined_embeddings


def get_subgraph(edge_index, batch_size):
    num_edges = edge_index.size(1)
    for i in range(0, num_edges, batch_size):
        sub_edge_index = edge_index[:, i:i + batch_size]
        yield sub_edge_index


def train():
    _, item_pic_urls, item_titles, item_ids_dict, edge_index = get_data_from_db()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    combined_embeddings = pre_embedding(item_pic_urls, item_titles)
    combined_embeddings_tensor = torch.cat(combined_embeddings).to(device)
    edge_index = edge_index.to(device)

    data_list = []
    for sub_edge_index in get_subgraph(edge_index, batch_size):
        data = Data(x=combined_embeddings_tensor, edge_index=sub_edge_index)
        data_list.append(data)
    dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    # input_dim = combined_embeddings_tensor.size(1)
    model = Autoencoder(input_dim=input_dim,hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers,
                        fc_hidden_dim=fc_hidden_dim, dropout_rate=dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.to(device)
    training_losses = []
    for epoch in range(num_epoch):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            x, edge_index = batch.x, batch.edge_index
            x, edge_index = x.to(device), edge_index.to(device)
            output = model(x, edge_index)
            loss = criterion(output, x)
            # loss.backward()
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        training_losses.append(avg_loss)
        print(f'Epoch {epoch + 1} ({math.floor((epoch + 1) / num_epoch)}%), Loss: {avg_loss}')
    torch.save(model.state_dict(), 'autoencoder.pth')
    plt.plot(range(num_epoch), training_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('train.jpg')


def encode(item_title, item_pic_url):
    combined_embed = pre_embedding([item_pic_url], [item_title])
    combined_embed_tensor = torch.cat(combined_embed).to(device)

    with torch.no_grad():
        x = combined_embed_tensor
        for layer in model.encoder:
            x = layer(x, torch.tensor([[0], [0]]).to(device))
            x = model.dropout(x)
        x = torch.relu(model.fc1(x))
        x = model.dropout(x)
        encoded = torch.relu(model.fc2(x))

    return encoded


def dbscan(encoded_items, item_ids, eps=0.5, min_samples=2):
    # Convert encoded items to numpy array for DBSCAN
    X = np.vstack([item.cpu().detach().numpy() for item in encoded_items])
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    # Create a dictionary to hold the item IDs and their cluster labels
    cluster_results = {item_id: cluster_label for item_id, cluster_label in zip(item_ids, clustering.labels_)}
    return cluster_results


def dbscan_all():
    # Pull all data from the database
    item_ids, item_pic_urls, item_titles, item_ids_dict, edge_index = get_data_from_db()
    # Encode each item using the encode function
    encoded_results = [encode(title, img_url) for title, img_url in zip(item_titles, item_pic_urls)]
    # Perform DBSCAN on the encoded items
    cluster_results = dbscan(encoded_results, item_ids)
    return cluster_results


if __name__ == '__main__':
    batch_size = 32
    num_heads = 8
    num_epoch = 5
    lr = 0.00001
    hidden_dim = 256
    num_layers = 2
    fc_hidden_dim = 128
    dropout_rate = 0.2
    input_dim = 2816

    dbscan_eps = 0.9
    dbscan_min_samples = 2

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\
    device = torch.device('cpu')
    model = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers,
                        fc_hidden_dim=fc_hidden_dim, dropout_rate=dropout_rate).to(device)
    model.load_state_dict(torch.load("autoencoder.pth"))
    model.eval()

    dbscan_all()

    #train()

    #a = encode("autoencoder.pth","某种衣服","a.jpg")
    #print(a.shape)
    #print(a)





