import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from model_loader import ModelLoader
from bert_utils import get_bert_embedding


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


def filter_edges(edge_index, batch_node_idx):
    mask = (torch.isin(edge_index[0], batch_node_idx) & torch.isin(edge_index[1], batch_node_idx))
    return edge_index[:, mask]


def train(model_loader, optimizer, dataloader, edge_index):
    model = model_loader.model
    device = model_loader.device
    bert_model = model_loader.bert_model
    model.train()
    epoch_losses = []  # 记录每个epoch的损失
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            item_ids, inputs = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embeddings = get_bert_embedding(bert_model, inputs).to(device)
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
            recon, mu, logvar = model(data.x, data.edge_index)
            # 重建损失
            recon_loss = F.mse_loss(recon, data.x)
            # KL 散度损失
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            # 总损失
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        end_time = time.time()
        epoch_duration = end_time - start_time
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss}, Duration: {epoch_duration:.2f} seconds')
    return epoch_losses


def validate(model_loader, dataloader, edge_index):
    model = model_loader.model
    device = model_loader.device
    bert_model = model_loader.bert_model
    model.eval()
    recon_losses = []
    kl_losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            item_ids, inputs = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embeddings = get_bert_embedding(bert_model, inputs).to(device)
            batch_node_idx = torch.tensor([int(item) for item in item_ids], dtype=torch.long)
            batch_edge_index = filter_edges(edge_index, batch_node_idx).to(device)
            batch_edge_index = batch_edge_index.type(torch.long)
            batch_node_idx_map = {idx: i for i, idx in enumerate(batch_node_idx.tolist())}
            reindexed_edge_index = torch.stack([
                torch.tensor([batch_node_idx_map[idx.item()] for idx in batch_edge_index[0]], dtype=torch.long),
                torch.tensor([batch_node_idx_map[idx.item()] for idx in batch_edge_index[1]], dtype=torch.long)
            ], dim=0)
            data = Data(x=embeddings, edge_index=reindexed_edge_index).to(device)
            recon, mu, logvar = model(data.x, data.edge_index)
            # 重建损失
            recon_loss = F.mse_loss(recon, data.x)
            # KL 散度损失
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())
    avg_recon_loss = sum(recon_losses) / len(recon_losses)
    avg_kl_loss = sum(kl_losses) / len(kl_losses)
    print(f'Validation, Reconstruction Loss: {avg_recon_loss}, KL Loss: {avg_kl_loss}')
    return avg_recon_loss, avg_kl_loss


def main():
    model_loader = ModelLoader()
    inter_file = '../data/Alibaba-iFashion/Alibaba-iFashion-pairs.inter'
    item_file = '../data/Alibaba-iFashion/Alibaba-iFashion-trimmed.item'
    val_inter_file = '../data/Alibaba-iFashion/Alibaba-iFashion-pairs-val.inter'
    val_item_file = '../data/Alibaba-iFashion/Alibaba-iFashion-trimmed-val.item'

    dataset = AlibabaDataset(inter_file, item_file, model_loader.tokenizer)
    val_dataset = AlibabaDataset(val_inter_file, val_item_file, model_loader.tokenizer)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    edge_index = dataset.edge_index
    val_edge_index = val_dataset.edge_index

    optimizer = torch.optim.Adam(model_loader.model.parameters(), lr=lr)

    epoch_losses = train(model_loader, optimizer, dataloader, edge_index)

    avg_recon_loss, avg_kl_loss = validate(model_loader, val_dataloader, val_edge_index)

    torch.save(model_loader.model.state_dict(), 'train/gae_model.pth')

    plt.figure()
    plt.plot(range(1, epochs + 1), epoch_losses)  # 绘制epoch损失
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.savefig("train/train.png")
    plt.show()

    # 保存验证指标
    with open("train/validation_metrics.txt", "w") as f:
        f.write(f'Reconstruction Loss: {avg_recon_loss}\n')
        f.write(f'KL Loss: {avg_kl_loss}\n')


if __name__ == "__main__":
    batch_size = 32
    lr = 0.001
    epochs = 10
    main()
