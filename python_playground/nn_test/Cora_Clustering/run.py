import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import os


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def train():
    dataset = Planetoid(root="dataset", name="Cora")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN(dataset.num_node_features, dataset.num_classes, 16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    losses = []
    validate_accuracies = []

    train_mask = dataset[0].train_mask
    val_mask = dataset[0].val_mask
    test_mask = dataset[0].test_mask

    for epoch in range(400):
        total_loss = 0
        for data in loader:
            data = data.to(device)
            model.train()
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # validate
        val_data = dataset[0].to(device)
        model.eval()
        val_out = model(val_data)
        val_pred = val_out.argmax(dim=1)
        val_correct = (val_pred[val_mask] == val_data.y[val_mask]).sum()
        val_acc = int(val_correct) / int(val_mask.sum())
        validate_accuracies.append(val_acc)

        epoch_loss = total_loss / len(loader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss}, accuracy: {val_acc}")\

    # 测试
    test_data = dataset[0].to(device)
    model.eval()
    test_out = model(test_data)
    test_pred = test_out.argmax(dim=1)
    test_correct = (test_pred[test_mask] == test_data.y[test_mask]).sum()
    test_acc = int(test_correct) / int(test_mask.sum())
    print(f"Test Accuracy: {test_acc}")

    os.makedirs("out", exist_ok=True)

    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.plot(validate_accuracies, label="Validation Accuracy")
    plt.legend()
    plt.savefig("out/training.png")

    torch.save(model.state_dict(), "out/last.pt")


if __name__ == "__main__":
    train()
