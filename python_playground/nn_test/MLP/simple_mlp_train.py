import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from simple_mlp import SimpleMLP

model = SimpleMLP()
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01)

data = torch.tensor([[i] for i in range(-10, 11)], dtype=torch.float32)
labels = torch.tensor([[1] if i > 0 else [0] for i in range(-10, 11)], dtype=torch.float32)

losses = []

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

#test_data = torch.tensor([[5.0], [-3.0], [-0.514]], dtype=torch.float32)
#with torch.no_grad():
#    test_outputs = model(test_data)
#    print(test_outputs)

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.savefig('loss_curve.jpg')

torch.save(model.state_dict(), 'simple_mlp.pt')