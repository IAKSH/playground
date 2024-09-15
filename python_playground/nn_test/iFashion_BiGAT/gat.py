import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from torch_geometric.nn import GATConv
from sklearn.cluster import DBSCAN
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

class BiGAT(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, num_heads):
        super(BiGAT, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.gat = GATConv(self.bert.config.hidden_size, hidden_dim, heads=num_heads, concat=False)
        self.decoder = nn.Linear(hidden_dim, hidden_dim)  # 调整 decoder 的输出维度

    def forward(self, sub_item_titles, edge_index):
        # BERT 层
        inputs = self.tokenizer(sub_item_titles, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = self.bert(**inputs)
        node_features = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] token 的输出

        # 双向图注意力层
        node_embeddings = self.gat(node_features, edge_index)

        # 重构
        reconstructed_features = self.decoder(node_embeddings)
        return node_embeddings, reconstructed_features

def get_data(max_rows=None):
    items = pd.read_csv('dataset/Alibaba-iFashion.item', sep='\t')
    items_unique = items.drop_duplicates(subset=['item_id:token', 'title:token_seq'])
    item_ids = items_unique['item_id:token'].tolist()
    item_ids_dict = {item_id: index for index, item_id in enumerate(item_ids)}

    item_index_list = []
    last_user_id = 0
    row = []
    col = []

    # 获取文件的总行数
    total_lines = 191394393

    # 计算需要读取的总行数
    if max_rows is not None:
        total_lines = min(total_lines, max_rows)

    # 使用 tqdm 添加进度条
    for chunk in tqdm(pd.read_csv('dataset/Alibaba-iFashion.inter', sep='\t', chunksize=1000, nrows=total_lines), total=total_lines//1000):
        inter_unique = chunk.drop_duplicates(subset=['user_id:token', 'item_id:token'])
        user_ids = inter_unique['user_id:token'].tolist()
        item_ids = inter_unique['item_id:token'].tolist()

        for i, user_id in enumerate(user_ids):
            if user_id == last_user_id:                
                item_index_list.append(item_ids_dict[item_ids[i]])
            else:
                # 为item_index_list中的所有item创建两两相连的边
                for j in range(len(item_index_list)):
                    for k in range(j + 1, len(item_index_list)):
                        row.append(item_index_list[j])
                        col.append(item_index_list[k])
                        row.append(item_index_list[k])
                        col.append(item_index_list[j])

                last_user_id = user_id
                item_index_list = [item_ids_dict[item_ids[i]]]
            i += 1

    edge_index = torch.tensor([row, col], dtype=torch.long)
    item_titles = items_unique['title:token_seq'].tolist()
    return item_titles, item_ids_dict, edge_index

def get_subgraph(edge_index, batch_size):
    num_edges = edge_index.size(1)
    for i in range(0, num_edges, batch_size):
        sub_edge_index = edge_index[:, i:i+batch_size]
        yield sub_edge_index

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

def main():
    # 获取全局图数据
    #item_titles, item_ids_dict, edge_index = get_data(1000000)
    item_titles, item_ids_dict, edge_index = get_data(100)

    batch_size = 32
    hidden_dim = 128
    num_heads = 8
    num_epoch = 3
    lr = 0.00001
    bert_model_name = 'bert-base-chinese'

    model = BiGAT(bert_model_name, hidden_dim, num_heads).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history = []

    for epoch in range(num_epoch):
        dataloader = []
        for sub_edge_index in get_subgraph(edge_index, batch_size):
            nodes = torch.unique(sub_edge_index)
            node_mapping = {node.item(): idx for idx, node in enumerate(nodes)}
            sub_edge_index = torch.tensor([[node_mapping[node.item()] for node in edge] for edge in sub_edge_index.t()], dtype=torch.long).t()
            sub_item_titles = [item_titles[node.item()] for node in nodes]
            dataloader.append((sub_edge_index, sub_item_titles))

        epoch_loss = train(model, dataloader, optimizer, criterion, device)
        loss_history.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epoch}, Loss: {epoch_loss}')

    # 保存 loss 变化图表
    plt.plot(range(1, num_epoch + 1), loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('loss_plot.png')

    # 保存模型参数
    torch.save(model.state_dict(), 'bigat_model.pth')

    # 聚类
    model.eval()
    all_embeddings = []
    for sub_edge_index, sub_item_titles in dataloader:
        sub_edge_index = sub_edge_index.to(device)
        sub_item_titles = [title for title in sub_item_titles]
        with torch.no_grad():
            node_embeddings, _ = model(sub_item_titles, sub_edge_index)
            all_embeddings.append(node_embeddings)
    all_embeddings = torch.cat(all_embeddings, dim=0).cpu().numpy()
    clustering = DBSCAN(eps=0.075, min_samples=2).fit(all_embeddings)
    print(clustering.labels_)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
