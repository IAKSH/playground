import torch
from PIL import Image
import numpy as np
from train import BiGAT, convert_to_rgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torchvision import models, transforms
import os
from PIL import Image
from torch_geometric.nn import GATConv
import numpy as np

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

def load_model(model_path, num_users, embedding_dim, gat_out_features, heads):
    model = BiGAT(num_users=num_users, embedding_dim=embedding_dim,
                  gat_out_features=gat_out_features, heads=heads)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, user_id, item_title, item_image_path):
    model.to(device)
    
    item_image = Image.open(item_image_path)
    item_image = convert_to_rgb(item_image)
    
    user_ids = [user_id]
    item_titles = [item_title]
    item_images = [item_image]
    
    # 创建一个只有两个节点的邻接矩阵，并连接这两个节点
    edge_matrix = np.array([[0, 1], [1, 0]])
    
    with torch.no_grad():
        output = model(user_ids, item_titles, item_images, edge_matrix)
    
    return output[0][0]

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = 'model.pth'
    num_users = 4
    embedding_dim = 1024
    gat_out_features = 2048
    heads = 4
    
    model = load_model(model_path, num_users, embedding_dim, gat_out_features, heads)

    #user_id = input("user id: ")
    user_id = '00001c0dae6531f6111ad8718a91d534'
    #item_image_path = input("item image path: ")
    item_image_path = 'C:/Users/lain/Pictures/Photo/屏幕截图_20240808_152904.png'
    
    while True:
        item_title = input("item title: ").split()
        prediction = predict(model, user_id, item_title, item_image_path)
        print(f"predict: {prediction}")
