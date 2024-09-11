import torch
from train import CNNModel
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


def show_img_from_2d_array(img):
    img = img.numpy().transpose(1, 2, 0)
    std = [0.5]
    mean = [0.5]
    img = img * std + mean
    img.resize(28, 28)
    plt.imshow(img)
    plt.show()


def load_image(path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    img = Image.open(path)
    transformed_img = transform(img)
    show_img_from_2d_array(transformed_img)
    img = transformed_img.unsqueeze(0)
    return img


if __name__ == '__main__':
    device = torch.device('cpu')
    model = CNNModel().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pth'))

    model.eval()
    with torch.no_grad():
        while True:
            img_path = input('path:')
            img = load_image(img_path).to(device)
            output = model(img)
            score, predicted = torch.max(output.data, 1)
            print(f'score: {score.item()}\tpredict: {predicted.item()}')