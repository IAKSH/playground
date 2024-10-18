import os
import torch
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data
from gcn_ae import GAE, get_bert_embedding
from hyperparameters import *
import mysql.connector


def encode(item_titles, edge_index):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(item_titles, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        embeddings = get_bert_embedding(inputs).to(device)
        data = Data(x=embeddings, edge_index=edge_index.to(device))
        encoded_features = model.encode(data.x, data.edge_index)
    return encoded_features


def generate_full_edge_index(n):
    row = []
    col = []
    for i in range(n):
        for j in range(n):
            if i != j:
                row.append(i)
                col.append(j)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    return edge_index


def get_data_from_db():
    db_config = {
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_DATABASE')
    }
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT hanfu_id, CONCAT(shop_name, ' ' , label , ' ', name, ' ', price) AS info FROM hanfu")
    items = cursor.fetchall()
    cursor.close()
    connection.close()
    return items


def demo(item_title):
    items_from_db = get_data_from_db()
    item_titles = [item_title] + [item['info'] for item in items_from_db]
    item_ids = [None] + [item['hanfu_id'] for item in items_from_db]
    edge_index = generate_full_edge_index(len(item_titles))

    encoded_features = encode(item_titles, edge_index)

    encoded_item = (encoded_features[0], item_ids[0])
    encoded_others = [(encoded_features[i+1], item_ids[i+1]) for i in range(len(items_from_db))]

    return encoded_item, encoded_others


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model = GAE(input_dim, hidden_dim, latent_dim).to(device)
    model.load_state_dict(torch.load("gae_model.pth"))

    item_title = "可能是某种衣服"
    encoded_item, encoded_others = demo(item_title)
    print(f"Encoded item: {encoded_item}")
    print(f"Encoded others: {encoded_others}")
