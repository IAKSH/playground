import torch
from torch_geometric.data import Data
from bert_utils import get_bert_embedding


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
