import torch
from simple_perceptron import SimplePerceptron

model = SimplePerceptron()
model.load_state_dict(torch.load('simple_perceptron.pt'))
model.eval()

test_data = torch.tensor([[114.514],[-0.0001]], dtype=torch.float32)
with torch.no_grad():
    test_output = model(test_data)
    print(test_output)