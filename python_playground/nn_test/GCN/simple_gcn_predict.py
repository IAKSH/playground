import torch
import numpy as np
from simple_gcn import SimpleGCN

model = SimpleGCN()
test_data = torch.tensor(np.random.rand(2, 3), dtype=torch.float32)  # 生成两个测试用户
test_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t().contiguous()
with torch.no_grad():
    test_outputs = model(test_data, test_edge_index)
    print(test_outputs)