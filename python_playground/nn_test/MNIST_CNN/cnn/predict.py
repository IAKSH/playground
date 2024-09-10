import time

import torch
from train import CNNModel
from PIL import Image
from torchvision import transforms


def load_image(path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    img = Image.open(path)
    img = transform(img).unsqueeze(0)
    return img


if __name__ == '__main__':
    device = torch.device('cpu')
    model = CNNModel().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pth'))
    model.eval()

    while True:
        img_path = input("image path:")
        img = load_image(img_path).to(device)

        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output.data, 1)
            print(f'predict: {predicted.item()}')

        time.sleep(1)