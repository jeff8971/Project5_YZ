import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from my_network import MyNetwork
import numpy as np


# Constants
PLOT_NUM_MNIST = 9
PLOT_NUM_HANDWRITTEN = 10
MODEL_PATH = "./model/model.pth"
MNIST_TEST_ROOT = './data'
HANDWRITTEN_TEST_ROOT = './data/handwritten'


def test_network_MNIST(model, test_loader):
    """Test the network on MNIST data, print output values, predictions, labels, and plot digits."""
    model.eval()
    plt.figure(figsize=(9, 9))  # Adjust the figure size to fit a 3x3 grid

    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            output_probabilities = F.softmax(output, dim=1).squeeze().tolist()

            # Format output probabilities to scientific notation with two decimal places
            formatted_output = [f"{prob:.2e}" for prob in output_probabilities]

            # Print the required information
            print(f"Image {idx + 1}, output values: [{', '.join(formatted_output)}], max value index: {pred.item()}, ground truth label: {target.item()}")

            # Plot the first 9 digits in a 3x3 grid
            if idx < 9:
                plt.subplot(3, 3, idx + 1)
                plt.imshow(data[0].squeeze().numpy(), cmap="gray")
                plt.title(f"Pred: {pred.item()}")
                plt.xticks([])
                plt.yticks([])

            # Break after 10 images
            if idx == 9:
                break

    plt.tight_layout()
    plt.show()

# The rest of the code remains the same



def test_network_handwritten(network, test_loader):
    """Test the network on handwritten digit images and plot predictions, and show accuracy."""
    network.eval()
    plt.figure(figsize=(15, 8))  # Adjust the figure size for a 3x4 grid

    correct = 0  # Initialize correct prediction counter
    total = 0  # Initialize total prediction counter

    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            if idx == PLOT_NUM_HANDWRITTEN:
                break

            output = network(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions
            total += target.size(0)  # Update total predictions

            plt.subplot(3, 4, idx + 1)  # Adjust for a 3x4 grid
            plt.imshow(data[0].squeeze().numpy(), cmap='gray')
            plt.title(f'Pred: {pred.item()}', pad=2)
            plt.axis('off')

    plt.tight_layout()
    plt.show()

    accuracy = 100. * correct / total  # Calculate accuracy
    print(f'Accuracy: {accuracy:.2f}%')



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