import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from torch_geometric.nn import GATConv
from sklearn.cluster import DBSCAN
from flask import Flask, jsonify, request, render_template
import mysql.connector
import threading
import time


class BiGAT(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, num_heads):
        super(BiGAT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.gat = GATConv(self.bert.config.hidden_size, hidden_dim, heads=num_heads, concat=False)
        self.decoder = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, sub_item_titles, edge_index):
        inputs = self.tokenizer(sub_item_titles, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = self.bert(**inputs)
        node_features = outputs.last_hidden_state[:, 0, :]
        node_embeddings = self.gat(node_features, edge_index)
        reconstructed_features = self.decoder(node_embeddings)
        return node_embeddings, reconstructed_features


db_config = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_DATABASE')
}


def get_data_from_db():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT hanfu_id, CONCAT(shop_name, ' ', name, ' ', price) AS info FROM hanfu")
    items = cursor.fetchall()
    cursor.close()
    connection.close()

    item_ids = [item['hanfu_id'] for item in items]
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
    return item_titles, item_ids_dict, edge_index


def get_subgraph(edge_index, batch_size):
    num_edges = edge_index.size(1)
    for i in range(0, num_edges, batch_size):
        sub_edge_index = edge_index[:, i:i+batch_size]
        yield sub_edge_index


def train_model_periodically(interval, model, dataloader, optimizer, criterion, device):
    while True:
        epoch_loss = train(model, dataloader, optimizer, criterion, device)
        print(f'Training Loss: {epoch_loss}')
        time.sleep(interval)


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for sub_edge_index, sub_item_titles in dataloader:
        sub_edge_index = sub_edge_index.to(device)
        sub_item_titles = [title for title in sub_item_titles]
        optimizer.zero_grad()
        node_embeddings, reconstructed_features = model(sub_item_titles, sub_edge_index)
        loss = criterion(reconstructed_features, node_embeddings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/recom', methods=['POST'])
def get_recom():
    data = request.json
    input_info = data['info']

    inputs = model.tokenizer(input_info, return_tensors='pt', padding=True, truncation=True).to(device)
    outputs = model.bert(**inputs)
    input_node_features = outputs.last_hidden_state[:, 0, :].to(device)

    all_inputs = model.tokenizer(item_titles, return_tensors='pt', padding=True, truncation=True).to(device)
    all_outputs = model.bert(**all_inputs)
    all_node_features = all_outputs.last_hidden_state[:, 0, :].to(device)

    combined_node_features = torch.cat((all_node_features, input_node_features), dim=0).to(device)

    node_embeddings = model.gat(combined_node_features, edge_index.to(device))

    clustering = DBSCAN(eps=0.05, min_samples=2).fit(node_embeddings.cpu().detach().numpy())
    labels = clustering.labels_

    input_label = labels[-1]
    similar_items = [{"item_id": i + 1, "info": item_titles[i]} for i, label in enumerate(labels[:-1]) if label == input_label]

    return jsonify(similar_items)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    item_titles, item_ids_dict, edge_index = get_data_from_db()
    batch_size = 32
    hidden_dim = 128
    num_heads = 8
    num_epoch = 3
    lr = 0.00001
    bert_model_name = 'bert-base-chinese'

    model = BiGAT(bert_model_name, hidden_dim, num_heads).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataloader = []
    for sub_edge_index in get_subgraph(edge_index, batch_size):
        nodes = torch.unique(sub_edge_index)
        node_mapping = {node.item(): idx for idx, node in enumerate(nodes)}
        sub_edge_index = torch.tensor([[node_mapping[node.item()] for node in edge] for edge in sub_edge_index.t()], dtype=torch.long).t()
        sub_item_titles = [item_titles[node.item()] for node in nodes]
        dataloader.append((sub_edge_index, sub_item_titles))

    # train every hour
    interval = 3600
    threading.Thread(target=train_model_periodically, args=(interval, model, dataloader, optimizer, criterion, device)).start()

    app.run(debug=False, host='0.0.0.0')
