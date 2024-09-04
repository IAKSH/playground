import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
from simple_gat import SimpleGAT

user_features = torch.tensor(np.loadtxt('user_features.txt'), dtype=torch.float32)
edges = torch.tensor(np.loadtxt('edges.txt', dtype=int), dtype=torch.long).t().contiguous()
labels = torch.tensor(np.loadtxt('labels.txt'), dtype=torch.float32).unsqueeze(1)

data = Data(x=user_features, edge_index=edges, y=labels)
model = SimpleGAT()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

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

torch.save(model.state_dict(), 'simple_gat.pt')