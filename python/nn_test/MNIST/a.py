import os
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image


def get_data_loader(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def save_first_five_images(data_loader, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images_saved = 0
    for images, _ in data_loader:
        for i in range(images.size(0)):
            if images_saved >= 10:
                return
            image = images[i]
            image = transforms.functional.to_pil_image(image)
            image.save(os.path.join(save_dir, f'image_{images_saved}.png'))
            images_saved += 1


batch_size = 64
train_loader, test_loader = get_data_loader(batch_size)
save_first_five_images(train_loader, './saved_images')
