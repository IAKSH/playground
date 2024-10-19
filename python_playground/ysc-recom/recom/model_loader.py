import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch_geometric.nn import GATConv


def reparameterize(mu, logvar):
    std = torch.exp(logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


class GATModelVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GATModelVAE, self).__init__()
        self.encoder1 = GATConv(input_dim, hidden_dim, heads=8, concat=True)
        self.encoder2_mu = GATConv(hidden_dim * 8, latent_dim, heads=1, concat=True)
        self.encoder2_logvar = GATConv(hidden_dim * 8, latent_dim, heads=1, concat=True)
        self.decoder1 = GATConv(latent_dim, hidden_dim * 8, heads=1, concat=True)
        self.decoder2 = GATConv(hidden_dim * 8, input_dim // 8, heads=8, concat=True)
        self.relu = nn.ReLU()

    def encode(self, x, edge_index):
        x = self.relu(self.encoder1(x, edge_index))
        mu = self.encoder2_mu(x, edge_index)
        logvar = self.encoder2_logvar(x, edge_index)
        z = reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z, edge_index):
        x = self.relu(self.decoder1(z, edge_index))
        return self.decoder2(x, edge_index)

    def forward(self, x, edge_index):
        z, mu, logvar = self.encode(x,edge_index)
        return self.decode(z, edge_index), mu, logvar


class ModelLoader:
    def __init__(self, model_path=None, use_gpu=True):
        self.input_dim = 768
        self.hidden_dim = 512
        self.latent_dim = 256
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.model = self.__load_model(model_path,use_gpu)

    def __load_model(self, model_path,use_gpu):
        model = GATModelVAE(self.input_dim, self.hidden_dim, self.latent_dim).to(self.device)
        if model_path:
            if not use_gpu:
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            else:
                model.load_state_dict(torch.load(model_path))
        return model
