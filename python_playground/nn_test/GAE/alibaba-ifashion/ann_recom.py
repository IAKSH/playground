import os
import torch
import mysql.connector
import faiss
from encode import encode
from model_loader import ModelLoader


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


def ann_recom(model_loader, item_title, n):
    items_from_db = get_data_from_db()
    item_titles = [item_title] + [item['info'] for item in items_from_db]
    item_ids = [None] + [item['hanfu_id'] for item in items_from_db]
    edge_index = generate_full_edge_index(len(item_titles))
    encoded_features = encode(model_loader, item_titles, edge_index)
    encoded_item = (encoded_features[0], item_ids[0])
    encoded_others = [(encoded_features[i + 1], item_ids[i + 1]) for i in range(len(items_from_db))]

    # 从encoded_others中提取特征向量和item_id
    other_features = torch.stack([item[0] for item in encoded_others]).cpu().numpy()
    other_ids = [item[1] for item in encoded_others]

    # 使用FAISS进行ANN搜索
    index = faiss.IndexFlatL2(other_features.shape[1])  # 建立索引
    index.add(other_features)  # 添加向量到索引中
    D, I = index.search(encoded_item[0].cpu().numpy().reshape(1, -1), n)  # 搜索最接近的n个向量

    # 提取最相近的item_id
    similar_items = [other_ids[i] for i in I[0]]

    return similar_items


def test():
    model_loader = ModelLoader('gae_model.pth')

    while True:
        n = 5
        item_title = input("请输入item_title: ")
        similar_items = ann_recom(model_loader, item_title, n)
        print(f"最相近的 {n} 个 item_id: {similar_items}")


if __name__ == "__main__":
    test()
