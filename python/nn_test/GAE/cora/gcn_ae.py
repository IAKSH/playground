import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt


class GAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GAE, self).__init__()
        self.encoder1 = GCNConv(input_dim, hidden_dim)
        self.encoder2 = GCNConv(hidden_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x, edge_index):
        x = F.relu(self.encoder1(x, edge_index))
        return self.encoder2(x, edge_index)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z)


def train(model, optimizer, data, epochs, device):
    model.train()
    data = data.to(device)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon = model(data.x, data.edge_index)
        loss = F.mse_loss(recon, data.x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    return losses


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid(root='data/Cora', name='Cora')
    data = dataset[0]
    input_dim = dataset.num_node_features
    hidden_dim = 32
    latent_dim = 16
    model = GAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 50

    losses = train(model, optimizer, data, epochs, device)
    torch.save(model.state_dict(), 'gcn_ae_model.pth')

    plt.figure()
    plt.plot(range(1, epochs+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()
    plt.savefig("gcn_ae_train.jpg")
