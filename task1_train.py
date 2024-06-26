#!/usr/bin/env python3
"""
Filename: test1_train.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: trains and tests a neural network on the MNIST dataset,
visualizes the training data and model architecture,
and plots training and testing losses.
Date: 2024-03-30
"""

import sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from my_network import MyNetwork
from task1_train_network import train_network, test_network
from torchviz import make_dot

# Constants
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.5
N_EPOCHS = 5
TRAIN_DATA_ROOT = './data'
TEST_DATA_ROOT = './data'
MODEL_SAVE_PATH = './model'
LOSS_PLOT_PATH = 'loss_plot.png'


def plot_image(dataset):
    """Plot the first six images and their labels from the dataset."""
    fig, ax = plt.subplots(figsize=(15, 10))
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(dataset.data[i].numpy(), cmap='gray')
        plt.title(f'Label: {dataset.targets[i]}')
        plt.axis('off')
    plt.show()


def main():
    # Data loading and normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST(root=TRAIN_DATA_ROOT, train=True,
                                download=True, transform=transform)
    test_data = datasets.MNIST(root=TEST_DATA_ROOT, train=False, download=True,
                               transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE_TRAIN,
                              shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE_TEST,
                             shuffle=True)

    # Visualize data samples
    plot_image(train_data)

    # Network and optimizer setup
    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM)

    # Visualize the model architecture
    sample_input = torch.randn(1000, 1, 28, 28)  # Adjust based on your
    # network's expected input shape
    model_vis = make_dot(network(sample_input), params=dict(network.named_parameters()))
    model_vis.render('model_visualization', format='png')  # Saves the visualization

    # Training and testing setup
    train_losses, train_counter = [], []
    test_losses, test_counter = [], [i * len(train_loader.dataset) for i in
                                     range(N_EPOCHS + 1)]

    # Initial test
    test_network(network, test_loader, test_losses)

    # Training and testing loop
    for epoch in range(1, N_EPOCHS + 1):
        train_network(network, train_loader, optimizer, epoch, train_losses,
                      train_counter, BATCH_SIZE_TRAIN, MODEL_SAVE_PATH)
        test_network(network, test_loader, test_losses)

    # Plotting the training and testing losses
    plt.plot(train_counter, train_losses, color='black')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Number of Training Examples Seen')
    plt.ylabel('Negative Log Likelihood Loss')
    plt.savefig(LOSS_PLOT_PATH, format='png', dpi=600)
    plt.show()


if __name__ == "__main__":
    main()