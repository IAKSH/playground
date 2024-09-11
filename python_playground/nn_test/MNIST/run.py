import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import argparse
from model import cnn, linear
from utils.dataset import get_data_loader
from utils.image import load_image


def train(device, model, batch_size, lr, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader, test_loader = get_data_loader(batch_size)

    train_losses = []
    test_accuracies = []

    best_acc = 0
    all_labels = []
    all_predictions = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), "out/best_acc.pt")

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss / len(train_loader)}, Accuracy: {accuracy}%")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.savefig("out/training_loss.png")

    plt.figure(figsize=(10, 5))
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy over Epochs")
    plt.legend()
    plt.savefig("out/test_accuracy.png")

    precision = precision_score(all_labels, all_predictions, average="macro")
    recall = recall_score(all_labels, all_predictions, average="macro")
    f1 = f1_score(all_labels, all_predictions, average="macro")

    metrics = {"Precision": precision, "Recall": recall, "F1 Score": f1}
    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(10, 5))
    plt.bar(names, values)
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.title("Model Performance Metrics")
    plt.savefig("out/model_performance_metrics.png")

    conf_matrix = confusion_matrix(all_labels, all_predictions)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig("out/confusion_matrix.png")

    torch.save(model.state_dict(), "out/last.pt")

    print(f"best accuracy: {best_acc}")


def predict(device, model, para_path):
    model.load_state_dict(torch.load(para_path))
    model.eval()
    with torch.no_grad():
        while True:
            img_path = input("path:")
            img = load_image(img_path).to(device)
            output = model(img)
            score, predicted = torch.max(output.data, 1)
            print(f"score: {score.item()}\tpredict: {predicted.item()}")


def get_model(args):
    if args.cnn:
        return cnn.Model()
    elif args.linear:
        return linear.Model()


def get_args():
    parser = argparse.ArgumentParser(description="nn for MNIST")
    parser.add_argument("--device", type=str, default="cpu", help="which device to run")

    parser.add_argument("--batch_size", type=int, help="size of a single batch", default=64)
    parser.add_argument("--epoch", type=int, help="num of epoch", default=10)
    parser.add_argument("--lr", type=float, help="learning rate", default=0.0001)

    parser.add_argument("--train", action="store_true", help="start a train")
    parser.add_argument("--predict", action="store_true", help="start a predict")
    parser.add_argument("--cnn", action="store_true", help="use cnn model")
    parser.add_argument("--linear", action="store_true", help="use linear model")

    parser.add_argument("--para_path", type=str, help="path of the model to load")

    return parser.parse_args()


def main():
    args = get_args()

    device = torch.device(args.device)
    model = get_model(args).to(device)

    if args.train:
        train(device, model, args.batch_size, args.lr, args.epoch)
    elif args.predict:
        predict(device, model, args.para_path)


if __name__ == "__main__":
    main()