import torch
import faiss
import encode
from model_loader import ModelLoader


def ann_recom(model_loader, n):
    item_title = input("请输入item_title: ")
    encoded_item, encoded_others = encode.demo(model_loader, item_title)

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


def main():
    model_loader = ModelLoader('gae_model.pth')

    n = 5
    similar_items = ann_recom(model_loader, n)
    print(f"最相近的 {n} 个 item_id: {similar_items}")


if __name__ == "__main__":
    main()
