import os
import shutil
import random
from PIL import Image


def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.width, img.height


def convert_to_yolo_format(input_file, output_file, image_folder):
    instance_name = os.path.splitext(os.path.basename(input_file))[0]
    image_path = os.path.join(image_folder, f"{instance_name}.png")
    image_width, image_height = get_image_dimensions(image_path)

    with open(input_file, 'r') as file:
        lines = file.readlines()

    coordinates = []
    for line in lines[2:]:  # 从第二行开始读取坐标点
        coordinates.extend(map(int, line.strip().split(' ')))

    x_coords = coordinates[::2]
    y_coords = coordinates[1::2]

    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height

    class_id = 0  # 因为只有一个类型
    yolo_format = f"{class_id} {x_center} {y_center} {width} {height}"

    with open(output_file, 'w') as file:
        file.write(yolo_format)


def change_extension_to_png(file_name):
    base = os.path.splitext(file_name)[0]
    return base + '.png'


def split_files(image_folder, label_folder, train_image_folder, train_label_folder, val_image_folder, val_label_folder,
                non_landslide_image_folder, ratio):
    for folder in [train_label_folder, val_label_folder, train_image_folder, val_image_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    files = os.listdir(label_folder)
    random.shuffle(files)
    validate_count = int(len(files) * ratio)
    validate_files = files[:validate_count]
    train_files = files[validate_count:]

    for file_name in validate_files:
        png_path = change_extension_to_png(file_name)
        shutil.copy(os.path.join(label_folder, file_name), os.path.join(val_label_folder, file_name))
        shutil.copy(os.path.join(image_folder, png_path), os.path.join(val_image_folder, png_path))

    for file_name in train_files:
        png_path = change_extension_to_png(file_name)
        shutil.copy(os.path.join(label_folder, file_name), os.path.join(train_label_folder, file_name))
        shutil.copy(os.path.join(image_folder, png_path), os.path.join(train_image_folder, png_path))

    non_landslide_images = os.listdir(non_landslide_image_folder)
    for image_name in non_landslide_images:
        shutil.copy(os.path.join(non_landslide_image_folder, image_name), os.path.join(train_image_folder, image_name))
        label_name = os.path.splitext(image_name)[0] + '.txt'
        with open(os.path.join(train_label_folder, label_name), 'w') as file:
            pass  # 创建一个空的标签文件


def generate_yaml(output_path):
    yaml_content = f"""
path: landslide-full-fixed
train: images/train
val: images/val

nc: 1
names: ['landslide']
"""
    with open(os.path.join(output_path, 'data.yaml'), 'w') as yaml_file:
        yaml_file.write(yaml_content)


def process_landslide():
    output_path = 'datasets/landslide-full-fixed'
    input_path = 'datasets/landslide'
    image_folder = f'{input_path}/landslide/image'
    non_landslide_image_folder = f'{input_path}/non-landslide/image'

    os.makedirs(os.path.join(output_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'temp', 'labels'), exist_ok=True)

    yolo_temp_path = f'{output_path}/temp'
    yolo_temp_label_path = f'{yolo_temp_path}/labels'
    original_label_folder = f'{input_path}/landslide/polygon_coordinate'

    label_file_names = os.listdir(original_label_folder)

    for label_file_name in label_file_names:
        original_label_path = f'{original_label_folder}/{label_file_name}'
        yolo_label_path = f'{yolo_temp_label_path}/{label_file_name}'
        convert_to_yolo_format(original_label_path, yolo_label_path, image_folder)

    split_files(image_folder, yolo_temp_label_path,
                f'{output_path}/images/train',
                f'{output_path}/labels/train',
                f'{output_path}/images/val',
                f'{output_path}/labels/val',
                non_landslide_image_folder,
                0.2)

    generate_yaml(output_path)
    shutil.rmtree(yolo_temp_path)


if __name__ == '__main__':
    process_landslide()
