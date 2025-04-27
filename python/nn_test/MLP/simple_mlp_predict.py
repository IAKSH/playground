import torch
from simple_mlp import SimpleMLP

model = SimpleMLP()
model.load_state_dict(torch.load('simple_mlp.pt'))
model.eval()  # 设置模型为评估模式

test_data = torch.tensor([[114.514],[-0.0001]], dtype=torch.float32)
with torch.no_grad():
    test_outputs = model(test_data)
    print(test_outputs)
