import torch
import numpy as np
from simple_bigat import SimpleBiGAT

# 加载训练好的模型
model = SimpleBiGAT()
model.load_state_dict(torch.load('bigat.pt'))
model.eval()

# 载入数据
user_features = torch.tensor(np.loadtxt('user_features.txt'), dtype=torch.float32)
edges = torch.tensor(np.loadtxt('edges.txt'), dtype=torch.long).t().contiguous()
labels = torch.tensor(np.loadtxt('labels.txt'), dtype=torch.float32).unsqueeze(1)

# 随机选择一个测试例
random_index = np.random.randint(0, edges.size(1))
test_edge_index = edges[:, random_index].unsqueeze(1)
test_data = user_features[test_edge_index.flatten()]

with torch.no_grad():
    # 进行预测
    test_outputs = model(test_data, torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t().contiguous())
    print("output:", test_outputs)

    # 比对结果的正确性
    actual_label = labels[random_index]
    print("actually:", actual_label)