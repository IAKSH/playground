import torch
import pandas as pd
from tqdm import tqdm

def count_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for line in file)


def get_data(max_rows=None):
    items = pd.read_csv('dataset/Alibaba-iFashion.item', sep='\t')
    items_unique = items.drop_duplicates(subset=['item_id:token', 'title:token_seq'])
    item_ids = items_unique['item_id:token'].tolist()
    item_ids_dict = {item_id: index for index, item_id in enumerate(item_ids)}

    item_index_list = []
    last_user_id = 0
    row = []
    col = []

    # 获取文件的总行数
    total_lines = 191394393

    # 计算需要读取的总行数
    if max_rows is not None:
        total_lines = min(total_lines, max_rows)

    # 使用 tqdm 添加进度条
    for chunk in tqdm(pd.read_csv('dataset/Alibaba-iFashion.inter', sep='\t', chunksize=1000, nrows=total_lines), total=total_lines//1000):
        inter_unique = chunk.drop_duplicates(subset=['user_id:token', 'item_id:token'])
        user_ids = inter_unique['user_id:token'].tolist()
        item_ids = inter_unique['item_id:token'].tolist()

        for i, user_id in enumerate(user_ids):
            if user_id == last_user_id:                
                item_index_list.append(item_ids_dict[item_ids[i]])
            else:
                # 为item_index_list中的所有item创建两两相连的边
                for j in range(len(item_index_list)):
                    for k in range(j + 1, len(item_index_list)):
                        row.append(item_index_list[j])
                        col.append(item_index_list[k])
                        row.append(item_index_list[k])
                        col.append(item_index_list[j])

                last_user_id = user_id
                item_index_list = [item_ids_dict[item_ids[i]]]
            i += 1

    edge_index = torch.tensor([row, col], dtype=torch.long)
    item_titles = items_unique['title:token_seq'].tolist()
    return item_titles, item_ids_dict, edge_index


def get_subgraph(edge_index, batch_size):
    num_edges = edge_index.size(1)
    for i in range(0, num_edges, batch_size):
        sub_edge_index = edge_index[:, i:i+batch_size]
        yield sub_edge_index


num_epoch = 10

def main():
    # 获取全局图数据
    item_titles, item_ids_dict, edge_index = get_data(1000000)

    batch_size = 32
    for epoch in range(num_epoch):
        for sub_edge_index in get_subgraph(edge_index, batch_size):
            # 获取子图中的节点
            nodes = torch.unique(sub_edge_index)
            # 重新索引子图中的节点
            node_mapping = {node.item(): idx for idx, node in enumerate(nodes)}
            sub_edge_index = torch.tensor([[node_mapping[node.item()] for node in edge] for edge in sub_edge_index.t()], dtype=torch.long).t()
            # 获取相关的 item_title
            sub_item_titles = [item_titles[node.item()] for node in nodes]
            
            # 处理每个子图
            # TODO
            print(sub_edge_index)
            print(sub_item_titles)

if __name__ == "__main__":
    main()