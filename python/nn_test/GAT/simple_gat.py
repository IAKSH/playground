import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class SimpleGAT(nn.Module):
    def __init__(self):
        super(SimpleGAT, self).__init__()
        self.conv1 = GATConv(3,16)
        self.conv2 = GATConv(16,1)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x,edge_index))
        x = self.conv2(x, edge_index)
        return x
