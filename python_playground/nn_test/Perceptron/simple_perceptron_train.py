import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from simple_perceptron import SimplePerceptron

model = SimplePerceptron()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

data = torch.tensor([[i] for i in range(-10,11)], dtype=torch.float32)
labels = torch.tensor([[1] if i > 0 else [0] for i in range(-10,11)], dtype=torch.float32)

losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.savefig('loss_curve.jpg')

torch.save(model.state_dict(), 'simple_perceptron.pt')