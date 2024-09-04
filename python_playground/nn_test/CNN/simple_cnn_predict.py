import torch
from simple_cnn import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load('simple_cnn.pt'))
model.eval()

test_data = torch.tensor([[114.514],[-0.0001]], dtype=torch.float32)
with torch.no_grad():
    test_outputs = model(test_data)
    print(test_outputs)