#!/usr/bin/env python3
"""
Filename: test1_test.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: Tests a neural network on MNIST and handwritten digits using
PyTorch,
displaying predictions and images.
It includes custom preprocessing for handwritten images
and dynamically adjusts visualization based on the number of images tested.
Date: 2024-03-30
"""

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


def test_network_MNIST(network, test_loader, plot_num=9):
    """Test the network on MNIST data, print output values, predictions, labels, and plot digits."""
    network.eval()  # Set the network to evaluation mode.
    predicted_labels = []  # Store the predicted labels.

    # Adjust figure size to be slightly smaller; adjust as needed for your display
    plt.figure(figsize=(8, 8))

    with torch.no_grad():  # No need to track gradients for testing.
        for idx, (data, target) in enumerate(test_loader):
            if idx >= plot_num:  # Only process a certain number of images.
                break

            output = network(data)  # Compute the network's output.
            pred = output.argmax(dim=1, keepdim=True)  # Find the prediction.

            # Format raw output values directly, without converting to probabilities.
            formatted_output = [f"{value:.2f}" for value in output[0]]

            # Print formatted information about predictions and actual labels.
            print(f"Image {idx + 1}, output values: [{', '.join(formatted_output)}], prediction: {pred.item()}, true label: {target.item()}")
            predicted_labels.append(pred.item())  # Store the predicted label for later.

            # Plotting: Adjust subplot to accommodate the specified number of images.
            ax = plt.subplot((plot_num + 2) // 3, 3, idx + 1)
            ax.imshow(data[0].squeeze().numpy(), cmap="gray", interpolation="none")  # Display the image.
            ax.set_title(f"Prediction: {predicted_labels[idx]}", pad=3)  # Set
            # the title to show the prediction, adjust padding as needed.
            ax.axis('off')  # Hide the axes.

    # Adjust the layout to ensure the text is not covered
    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.5)
    plt.show()


def test_network_handwritten(network, test_loader, plot_num=12):
    """Test the network on handwritten digit images, print output values, predictions, labels, and plot digits. Show overall accuracy."""
    network.eval()  # Set the network to evaluation mode.
    correct = 0  # Initialize correct prediction counter.
    total = 0  # Initialize total prediction counter.
    plt.figure(figsize=(10, 8))  # Adjust the figure size for a better fit of plot_num images.

    with torch.no_grad():  # No need to track gradients for testing.
        for idx, (data, target) in enumerate(test_loader):
            if idx >= plot_num:  # Only process a certain number of images.
                break

            output = network(data)  # Compute the network's output.
            pred = output.argmax(dim=1, keepdim=True)  # Find the prediction.
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions.
            total += target.size(0)  # Update total predictions.

            # Format raw output values directly, without converting to probabilities.
            formatted_output = [f"{value:.2f}" for value in output[0]]

            # Print formatted information about predictions and actual labels.
            print(f"Image {idx + 1}, output values: [{', '.join(formatted_output)}], prediction: {pred.item()}, true label: {target.item()}")

            # Plotting: Adjust subplot to accommodate the specified number of images.
            ax = plt.subplot((plot_num + 3) // 4, 4, idx + 1)  # Adjust for a dynamic grid based on plot_num.
            ax.imshow(data[0].squeeze().numpy(), cmap="gray", interpolation="none")  # Display the image.
            ax.set_title(f"Pred: {pred.item()}", pad=3)  # Set the title to show the prediction, adjust padding as needed.
            ax.axis('off')  # Hide the axes.

    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.5)
    plt.show()

    accuracy = 100. * correct / total  # Calculate accuracy.
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

    print("=================================================================")
    print("=================================================================")
    print("MNIST digits:\n")

    test_network_MNIST(network, test_loader_MNIST)

    handwritten_transform = transforms.Compose([
        HandwrittenTransform(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    handwritten_data = datasets.ImageFolder(root=HANDWRITTEN_TEST_ROOT, transform=handwritten_transform)
    handwritten_loader = DataLoader(handwritten_data, batch_size=1)
    print("=================================================================")
    print("=================================================================")
    print("\nHandwritten digits:")

    test_network_handwritten(network, handwritten_loader)


if __name__ == '__main__':
    main()