import pandas as pd
from collections import defaultdict
import argparse


def preprocess_inter_file(inter_file, output_file, n_lines, offset):
    user_items = defaultdict(list)
    # 读取inter文件的偏移后前N行
    with open(inter_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if i >= offset + n_lines:
                break
            user_id, item_id = line.strip().split('\t')
            user_items[user_id].append(item_id.strip())
    # 写入输出文件，包括格式说明的第一行
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        f.write("item_a_id:token\titem_b_id:token\n")  # 写入第一行
        for items in user_items.values():
            unique_pairs = set()
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    if (items[i], items[j]) not in unique_pairs and (items[j], items[i]) not in unique_pairs:
                        f.write(f"{items[i]}\t{items[j]}\n")
                        unique_pairs.add((items[i], items[j]))


def process_item_file(item_file, output_file, inter_file):
    inter_items = set()
    # 读取inter文件，收集所有涉及的item_id
    with open(inter_file, 'r', encoding='utf-8') as f:
        next(f)  # 跳过第一行
        for line in f:
            item1, item2 = line.strip().split('\t')
            inter_items.add(item1.strip())
            inter_items.add(item2.strip())
    # 按块读取item文件，并筛选涉及到的item信息
    chunk_size = 1000
    filtered_chunks = []
    with pd.read_csv(item_file, sep='\t', header=0, chunksize=chunk_size, dtype=str) as reader:
        for chunk in reader:
            chunk['item_id:token'] = chunk['item_id:token'].str.strip()  # 去除空格和换行符
            filtered_chunk = chunk[chunk['item_id:token'].isin(inter_items)]
            filtered_chunks.append(filtered_chunk)
    # 合并所有筛选后的块并写入文件
    result = pd.concat(filtered_chunks).drop_duplicates()
    result.to_csv(output_file, sep='\t', index=False, header=True, mode='w', encoding='utf-8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inter', type=str, required=True)
    parser.add_argument('--item', type=str, required=True)
    parser.add_argument('--inter_out', type=str, required=True)
    parser.add_argument('--item_out', type=str, required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--offset', type=int, required=True)

    args = parser.parse_args()

    preprocess_inter_file(args.inter, args.inter_out, args.n, args.offset)
    process_item_file(args.item, args.item_out, args.inter_out)

    print("数据集裁剪完成！")
