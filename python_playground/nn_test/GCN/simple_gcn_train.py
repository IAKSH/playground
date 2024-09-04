import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
from simple_gcn import SimpleGCN
import matplotlib.pyplot as plt

# 载入数据
user_features = torch.tensor(np.loadtxt('user_features.txt'), dtype=torch.float32)
edges = torch.tensor(np.loadtxt('edges.txt', dtype=int), dtype=torch.long).t().contiguous()
labels = torch.tensor(np.loadtxt('labels.txt'), dtype=torch.float32).unsqueeze(1)

# 构建图数据
data = Data(x=user_features, edge_index=edges, y=labels)

# 创建模型实例
model = SimpleGCN()

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
losses = []
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(data.x, data.edge_index)
    edge_outputs = outputs[data.edge_index[0]]
    loss = criterion(edge_outputs, data.y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.savefig('loss_curve.jpg')

torch.save(model.state_dict(), 'simple_gcn.pt')