import torch
from transformers import BertTokenizer, BertModel
from gcn_ae import GAE
from hyperparameters import *
import faiss
import encode


def ann_recom(n):
    item_title = input("请输入item_title: ")
    encoded_item, encoded_others = encode.demo(item_title)

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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model = GAE(input_dim, hidden_dim, latent_dim).to(device)
    model.load_state_dict(torch.load("gae_model.pth"))

    # 示例调用
    n = 5  # 设定要找的最近邻数量
    similar_items = ann_recom(n)
    print(f"最相近的 {n} 个 item_id: {similar_items}")
