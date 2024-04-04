import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from my_network import MyNetwork


# Constants
PLOT_NUM_MNIST = 9
PLOT_NUM_HANDWRITTEN = 10
MODEL_PATH = "./model/model.pth"
MNIST_TEST_ROOT = './data'
HANDWRITTEN_TEST_ROOT = './data/handwritten'


def test_network_MNIST(model, test_loader, plot_num=PLOT_NUM_MNIST):
    """Test the network on MNIST data and plot predictions."""
    model.eval()
    plt.figure(figsize=(12, (plot_num // 3 + 2) * 3))

    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            if idx == plot_num:
                break

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            print(f"Image {idx + 1} prediction: {pred[0][0].item()}, label: {target.item()}")

            plt.subplot((plot_num + 1) // 3, 3, idx + 1)
            plt.imshow(data[0].squeeze().numpy(), cmap="gray")
            plt.title(f"Pred: {pred[0][0].item()}", y=1.08, fontsize=16)
            plt.xticks([])
            plt.yticks([])

    plt.tight_layout()
    plt.show()


def test_network_handwritten(network, test_loader):
    """Test the network on handwritten digit images and plot predictions."""
    network.eval()
    test_loss, correct = 0, 0

    plt.figure(figsize=(15, 6))

    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            if idx == PLOT_NUM_HANDWRITTEN:
                break

            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            plt.subplot(3, 3, idx + 1)
            plt.imshow(data[0].squeeze().numpy(), cmap='gray')
            plt.title(f'Pred: {pred.item()}', pad=2)
            plt.axis('off')

    plt.tight_layout()
    plt.show()

    test_loss /= idx + 1
    accuracy = 100. * correct / ((idx + 1) * test_loader.batch_size)
    print(f'\nTest Set: Average Loss: {test_loss:.4f}, Accuracy: {correct}/{(idx + 1) * test_loader.batch_size} ({accuracy:.0f}%)')


class HandwrittenTransform:
    """Custom transformation for preprocessing handwritten digit images."""
    def __call__(self, x):
        x = transforms.functional.rgb_to_grayscale(x)
        x = transforms.functional.invert(x)
        return x


def main():
    network = MyNetwork()
    network.load_state_dict(torch.load(MODEL_PATH))

    test_data_MNIST = datasets.MNIST(
        root=MNIST_TEST_ROOT, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    test_loader_MNIST = DataLoader(Subset(test_data_MNIST, range(PLOT_NUM_MNIST)), batch_size=1)

    test_network_MNIST(network, test_loader_MNIST)

    handwritten_transform = transforms.Compose([
        HandwrittenTransform(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    handwritten_data = datasets.ImageFolder(root=HANDWRITTEN_TEST_ROOT, transform=handwritten_transform)
    handwritten_loader = DataLoader(handwritten_data, batch_size=1)

    test_network_handwritten(network, handwritten_loader)


if __name__ == '__main__':
    main()