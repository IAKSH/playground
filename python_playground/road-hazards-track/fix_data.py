import os

# 定义你的数据集路径和train.txt文件路径
dataset_dir = 'E:/repos/playground/python_playground/road-hazards-track/datasets/UAV-PDD2023'
train_file_path = os.path.join(dataset_dir, 'ImageSets/Main/train.txt')
output_file_path = os.path.join(dataset_dir, 'ImageSets/Main/train_full_paths.txt')

# 创建一个新的文件来存储带有完整路径的图像文件
with open(train_file_path, 'r') as f:
    lines = f.readlines()

with open(output_file_path, 'w') as f:
    for line in lines:
        line = line.strip()  # 去掉每行末尾的空白字符
        full_path = os.path.join('datasets/UAV-PDD2023/JPEGImages', f"{line}.jpg")
        f.write(f"{full_path}\n")

print("完成：新的 train_full_paths.txt 文件已生成。")
