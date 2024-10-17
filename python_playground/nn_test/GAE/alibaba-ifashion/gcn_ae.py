import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


class AlibabaDataset(Dataset):
    def __init__(self, inter_file, item_file, tokenizer):
        self.inter_file = inter_file
        self.item_file = item_file
        self.tokenizer = tokenizer
        self.item_data = pd.read_csv(self.item_file, sep='\t', header=0)
        self.item_ids = set(self.item_data['item_id:token'].unique())
        self.inter_data = pd.read_csv(self.inter_file, sep='\t', header=0)
        self.edge_index = torch.tensor(self.inter_data.values.T, dtype=torch.long)

    def __len__(self):
        return len(self.item_data)

    def __getitem__(self, idx):
        item_info = self.item_data.iloc[idx]
        item_id = item_info['item_id:token']
        title = item_info['title:token_seq']
        inputs = self.tokenizer(title, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return item_id, inputs


def custom_collate(batch):
    item_ids, inputs_list = zip(*batch)
    inputs = {}
    for key in inputs_list[0].keys():
        inputs[key] = torch.stack([inputs[key] for inputs in inputs_list], dim=0)
    return item_ids, inputs


def get_bert_embedding(inputs):
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


class GAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GAE, self).__init__()
        self.encoder1 = GATConv(input_dim, hidden_dim, heads=8, concat=True)
        self.encoder2 = GATConv(hidden_dim * 8, latent_dim, heads=1, concat=True)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x, edge_index):
        x = F.relu(self.encoder1(x, edge_index))
        return self.encoder2(x, edge_index)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z)


def filter_edges(edge_index, batch_node_idx):
    mask = (torch.isin(edge_index[0], batch_node_idx) & torch.isin(edge_index[1], batch_node_idx))
    return edge_index[:, mask]


def train():
    model.train()
    epoch_losses = []  # 记录每个epoch的损失
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            item_ids, inputs = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embeddings = get_bert_embedding(inputs).to(device)
            batch_node_idx = torch.tensor([int(item) for item in item_ids], dtype=torch.long)
            batch_edge_index = filter_edges(edge_index, batch_node_idx).to(device)
            batch_edge_index = batch_edge_index.type(torch.long)
            batch_node_idx_map = {idx: i for i, idx in enumerate(batch_node_idx.tolist())}
            reindexed_edge_index = torch.stack([
                torch.tensor([batch_node_idx_map[idx.item()] for idx in batch_edge_index[0]], dtype=torch.long),
                torch.tensor([batch_node_idx_map[idx.item()] for idx in batch_edge_index[1]], dtype=torch.long)
            ], dim=0)
            data = Data(x=embeddings, edge_index=reindexed_edge_index).to(device)
            optimizer.zero_grad()
            recon = model(data.x, data.edge_index)
            loss = F.mse_loss(recon, data.x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        end_time = time.time()
        epoch_duration = end_time - start_time
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss}, Duration: {epoch_duration:.2f} seconds')
    return epoch_losses


if __name__ == "__main__":
    batch_size = 32
    input_dim = 768
    hidden_dim = 32
    latent_dim = 16
    epochs = 5
    lr = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    inter_file = '../data/Alibaba-iFashion/Alibaba-iFashion-pairs.inter'
    item_file = '../data/Alibaba-iFashion/Alibaba-iFashion-trimmed.item'
    dataset = AlibabaDataset(inter_file, item_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    edge_index = dataset.edge_index
    model = GAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_losses = train()
    torch.save(model.state_dict(), 'gae_model.pth')
    plt.figure()
    plt.plot(range(1, epochs + 1), epoch_losses)  # 绘制epoch损失
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.show()
    plt.savefig("train.png")