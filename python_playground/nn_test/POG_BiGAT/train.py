from io import BytesIO
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torchvision import models, transforms
import os
from PIL import Image
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import numpy as np
import random

# 仅仅为测试用
test_img = Image.open('img/1.jpg')

embedding_dim = 1024
gat_out_features = 2048
heads = 4
lr = 0.00001
num_epoch = 30
# TODO: 不知道为什么，用batch_size=4就会炸
batch_size = 2
num_users = 4

user_data_path = 'dataset/user_data.txt'
item_data_path = 'dataset/item_data.txt'

def convert_to_rgb(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    return image

class BiGAT(nn.Module):
    def __init__(self, num_users, embedding_dim, bert_output_dim=768, resnet_output_dim=32, gat_out_features=32, heads=4):
        super(BiGAT, self).__init__()
        
        # BERT for text processing (product description)
        self.bert = AutoModel.from_pretrained("bert-base-chinese",
                                              cache_dir="models/bert-base-chinese/", local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", cache_dir="models/bert-base-chinese/", local_files_only=True)
        
        # CNN for image processing
        os.environ['TORCH_HOME'] = 'models/resnet50/'
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, resnet_output_dim)
        
        # Embedding for user ID
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        # GAT layers
        concat_dim = bert_output_dim + resnet_output_dim + embedding_dim
        self.gat1 = GATConv(concat_dim, gat_out_features, heads=heads, concat=True)
        self.gat2 = GATConv(gat_out_features * heads, gat_out_features, heads=heads, concat=False)
        
        self.fc = nn.Linear(gat_out_features, 1)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the images to the size expected by ResNet
            transforms.ToTensor(),  # Convert the images to tensors
        ])

    def forward(self, user_ids, item_titles, item_images, edge_matrix):
        # 将字符串 ID 列表转换为索引列表
        user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
        user_ids_indices = [user_id_to_index[user_id] for user_id in user_ids]
        user_ids_tensor = torch.tensor(user_ids_indices).to(device)

        # 提取item title特征
        item_titles = [" ".join(title) for title in item_titles]
        bert_inputs = self.tokenizer(item_titles, return_tensors='pt', padding=True, truncation=True).to(device)
        item_titles_tensor = self.bert(**bert_inputs).last_hidden_state.mean(dim=1).to(device)
        
        # 提取item picture特征
        item_images = [convert_to_rgb(image) for image in item_images]
        item_images_tensor = torch.stack([self.transform(image) for image in item_images]).to(device)
        item_images_tensor = self.resnet(item_images_tensor)
          
        # 提取user id特征
        user_ids_tensor = self.user_embedding(user_ids_tensor)
        
        # 然后把item和picture和title加起来成一个项
        items_tensor = torch.cat((item_titles_tensor, item_images_tensor), dim=1).to(device)
                                                          
        # 最后把user和item分别填充成同样规格的节点特征向量
        users_len = user_ids_tensor.size(1)
        items_len = items_tensor.size(1)
        users_tensor_padded = torch.nn.functional.pad(user_ids_tensor, (0, items_len)).to(device)
        items_tensor_padded = torch.nn.functional.pad(items_tensor, (users_len, 0)).to(device)
        
        # 然后再拼一起就可以送进GAT了
        x = torch.cat((users_tensor_padded, items_tensor_padded), dim=0).to(device)
        # GAT layers
        edge_matrix = torch.tensor(edge_matrix, dtype=torch.int).to(device)
        x = F.relu(self.gat1(x, edge_matrix))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_matrix)
        
        # Fully connected layer
        x = self.fc(x)

        # 输出购买倾向的概率
        return torch.sigmoid(x)

def download_image(url):
    if url.startswith('//'):
        url = 'http:' + url
    print(f'doawnloading: {url}')
    response = requests.get(url)
    if response.status_code == 200 and 'image' in response.headers['Content-Type']:
        try:
            img = Image.open(BytesIO(response.content))
            return img
        except Exception as e:
            print(f"Error opening image: {e}")
            return test_img
    else:
        print(f"url: {url}")
        return test_img

class DynamicSquareArray:
    def __init__(self, initial_size=1):
        self.array = np.zeros((initial_size, initial_size), dtype=int)
    
    def set_value(self, row, col, value):
        max_dim = max(row + 1, col + 1)
        if max_dim > self.array.shape[0]:
            new_size = max_dim
            new_array = np.zeros((new_size, new_size), dtype=int)
            new_array[:self.array.shape[0], :self.array.shape[1]] = self.array
            self.array = new_array
        self.array[row, col] = value
    
    def get_array(self):
        return self.array
    

def load_items_to_memory(item_data_path):
    items = {}
    with open(item_data_path, encoding='utf-8') as item_file:
        for line in item_file:
            parts = line.strip().split(',')
            item_id = parts[0]
            pic_url = parts[2]
            title = parts[3:]
            items[item_id] = (pic_url, title)
    return items

def load_data():
    user_ids = []
    item_titles = []
    item_images = []
    adj = DynamicSquareArray()

    # 这个文件得有1G+，暂且还能塞进内存吧
    all_loaded_items = load_items_to_memory(item_data_path)

    with open(user_data_path, encoding='utf-8') as user_file:
        load_count = 0
        for i, line in enumerate(user_file):
            if i >= num_users:
                break
            parts = line.strip().split(',')
            user_id = parts[0]
            user_ids.append(user_id)
            item_ids = parts[1].split(';')
            for required_item_id in item_ids:
                if required_item_id not in all_loaded_items:
                    raise Exception(f'Item {required_item_id} not found!!!')
                pic_url, title = all_loaded_items[required_item_id]
                item_images.append(download_image(pic_url))
                item_titles.append(title)
                adj.set_value(len(user_ids) - 1, len(item_images) - 1, 1)
                load_count += 1
                print(f'loaded item: {load_count}')

    return user_ids, item_titles, item_images, adj.get_array()

def batch_data(user_ids, item_titles, item_images, adj, batch_size):
    for i in range(0, len(user_ids), batch_size):
        batch_user_ids = user_ids[i:i + batch_size]
        batch_item_titles = item_titles[i:i + batch_size]
        batch_item_images = item_images[i:i + batch_size]
        
        # 提取子矩阵
        batch_adj = adj[i:i + batch_size, i:i + batch_size]
        
        yield (
            batch_user_ids,
            batch_item_titles,
            batch_item_images,
            batch_adj
        )

# 生成负样本
def generate_negative_samples(user_ids, item_titles, item_images, adj, num_neg_samples):
    neg_user_ids = []
    neg_item_titles = []
    neg_item_images = []
    neg_adj = DynamicSquareArray()

    for _ in range(num_neg_samples):
        user_id = random.choice(user_ids)
        item_title = random.choice(item_titles)
        item_image = random.choice(item_images)
        
        neg_user_ids.append(user_id)
        neg_item_titles.append(item_title)
        neg_item_images.append(item_image)
        neg_adj.set_value(len(neg_user_ids) - 1, len(neg_item_images) - 1, 0)

    return neg_user_ids, neg_item_titles, neg_item_images, neg_adj.get_array()

def pad_matrix(matrix, target_shape):
    padded_matrix = np.zeros(target_shape, dtype=int)
    padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
    return padded_matrix

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiGAT(num_users=num_users, embedding_dim=embedding_dim,
                  gat_out_features=gat_out_features, heads=heads).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []

    user_ids, item_titles, item_images, adj = load_data()
    neg_user_ids, neg_item_titles, neg_item_images, neg_adj = generate_negative_samples(user_ids, item_titles, item_images, adj, num_neg_samples=1000)

    # 合并正样本和负样本
    all_user_ids = user_ids + neg_user_ids
    all_item_titles = item_titles + neg_item_titles
    all_item_images = item_images + neg_item_images

    # 填充邻接矩阵
    max_dim = max(adj.shape[0], neg_adj.shape[0])
    adj_padded = pad_matrix(adj, (max_dim, max_dim))
    neg_adj_padded = pad_matrix(neg_adj, (max_dim, max_dim))
    all_adj = np.vstack((adj_padded, neg_adj_padded))

    # run epoch
    for epoch in range(num_epoch):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0
        batches = batch_data(all_user_ids, all_item_titles, all_item_images, all_adj, batch_size)
        batch_count = 0
        # run batch
        for batch in batches:
            user_id_batch, item_title_batch, item_image_batch, adj_batch = batch
            output = model(user_id_batch, item_title_batch, item_image_batch, adj_batch)
            # 生成标签，正样本为1，负样本为0
            labels = torch.cat((torch.ones(len(user_id_batch)), torch.zeros(len(user_id_batch)))).to(device)
            labels = labels.view(-1, 1)  # 调整标签大小以匹配输出
            loss = criterion(output, labels)
            loss.backward()
            epoch_loss += loss.item()
            batch_count += 1
        
        optimizer.step()
        train_losses.append(epoch_loss / batch_count)
        print(f'Epoch {epoch+1}/{num_epoch}, Loss: {epoch_loss / batch_count}')

    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig('loss_curve.jpg')

    torch.save(model.state_dict(), 'model.pth')