import torch
import torch.nn as nn


class SimplePerceptron(nn.Module):
    def __init__(self):
        super(SimplePerceptron, self).__init__()
        self.fc = nn.Linear(1,1)

    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        return x