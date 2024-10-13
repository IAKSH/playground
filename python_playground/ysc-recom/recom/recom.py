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
    return item_pic_urls, item_titles, item_ids_dict, edge_index

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

def create_embeddings(item_pic_urls, item_titles):
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

def train_model(model, dataloader, optimizer, criterion, device, num_epoch):
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
            #loss.backward()
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


def train():
    item_pic_urls, item_titles, item_ids_dict, edge_index = get_data_from_db()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    combined_embeddings = create_embeddings(item_pic_urls, item_titles)
    combined_embeddings_tensor = torch.cat(combined_embeddings).to(device)
    edge_index = edge_index.to(device)
    batch_size = 32
    num_heads = 8
    num_epoch = 5
    lr = 0.00001
    data_list = []
    for sub_edge_index in get_subgraph(edge_index, batch_size):
        data = Data(x=combined_embeddings_tensor, edge_index=sub_edge_index)
        data_list.append(data)
    dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    model = Autoencoder(input_dim=combined_embeddings_tensor.size(1),
                        hidden_dim=256, num_heads=num_heads, num_layers=2, fc_hidden_dim=128, dropout_rate=0.2).to(
        device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_model(model, dataloader, optimizer, criterion, device, num_epoch)


if __name__ == '__main__':
    train()
