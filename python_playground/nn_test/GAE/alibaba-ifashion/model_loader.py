import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F


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


class ModelLoader:
    def __init__(self, model_path=None):
        self.input_dim = 768
        self.hidden_dim = 512
        self.latent_dim = 256

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.model = self.__load_model(model_path)

    def __load_model(self, model_path):
        model = GAE(self.input_dim, self.hidden_dim, self.latent_dim).to(self.device)
        if model_path:
            model.load_state_dict(torch.load(model_path))
        return model

# Usage example:
# model_loader = ModelLoader(model_path='path_to_trained_model.pt')
# model = model_loader.model
