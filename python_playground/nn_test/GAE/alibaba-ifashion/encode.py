import os
import torch
from torch_geometric.data import Data
from bert_utils import get_bert_embedding
import mysql.connector
from model_loader import ModelLoader


def encode(model_loader, item_titles, edge_index):
    model = model_loader.model
    tokenizer = model_loader.tokenizer
    device = model_loader.device
    bert_model = model_loader.bert_model
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(item_titles, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        embeddings = get_bert_embedding(bert_model, inputs).to(device)
        data = Data(x=embeddings, edge_index=edge_index.to(device))
        encoded_features, _, _ = model.encode(data.x, data.edge_index)
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


def demo(model_loader, item_title):
    items_from_db = get_data_from_db()
    item_titles = [item_title] + [item['info'] for item in items_from_db]
    item_ids = [None] + [item['hanfu_id'] for item in items_from_db]
    edge_index = generate_full_edge_index(len(item_titles))
    encoded_features = encode(model_loader, item_titles, edge_index)
    encoded_item = (encoded_features[0], item_ids[0])
    encoded_others = [(encoded_features[i+1], item_ids[i+1]) for i in range(len(items_from_db))]
    return encoded_item, encoded_others


def main():
    model_loader = ModelLoader('gae_model.pth')
    item_title = '可能是某种衣服'
    encoded_item, encoded_others = demo(model_loader, item_title)
    print(f"Encoded item: {encoded_item}")
    print(f"Encoded others: {encoded_others}")


if __name__ == "__main__":
    main()
