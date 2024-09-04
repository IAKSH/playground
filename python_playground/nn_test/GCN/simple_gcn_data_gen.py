import numpy as np
import torch

# 生成用户节点和特征
num_users = 100
user_features = torch.tensor(np.random.rand(num_users, 3), dtype=torch.float32)  # 每个用户有3个特征

# 生成朋友关系边
edges = []
for i in range(num_users):
    for j in range(i + 1, num_users):
        if np.random.rand() > 0.9:  # 10%的概率生成朋友关系
            edges.append([i, j])
            edges.append([j, i])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
labels = torch.tensor([1 if np.random.rand() > 0.5 else 0 for _ in range(len(edges))], dtype=torch.float32).unsqueeze(1)

np.savetxt('user_features.txt', user_features)
np.savetxt('edges.txt', edges, fmt='%d')
np.savetxt('labels.txt', labels)