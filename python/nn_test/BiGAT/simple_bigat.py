import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class SimpleBiGAT(nn.Module):
    def __init__(self):
        super(SimpleBiGAT, self).__init__()
        self.conv1 = GATConv(3, 16)
        self.conv2 = GATConv(16, 1)
        self.conv1_reverse = GATConv(3, 16)
        self.conv2_reverse = GATConv(16, 1)

    def forward(self, x, edge_index):
        # 正向传播
        x_forward = torch.relu(self.conv1(x, edge_index))
        x_forward = self.conv2(x_forward, edge_index)

        # 反向传播
        edge_index_reverse = torch.stack([edge_index[1], edge_index[0]], dim=0)
        x_reverse = torch.relu(self.conv1_reverse(x, edge_index_reverse))
        x_reverse = self.conv2_reverse(x_reverse, edge_index_reverse)

        # 合并正向和反向的结果
        x = (x_forward + x_reverse) / 2
        return x