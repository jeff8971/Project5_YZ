import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from time import time

# Configuration settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size_train, batch_size_test = 64, 1000
learning_rate, momentum = 0.01, 0.5
num_epochs = 10

# Dataset Loaders
train_loader = DataLoader(
    datasets.FashionMNIST('./data_fashion', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                          ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = DataLoader(
    datasets.FashionMNIST('./data_fashion', train=False, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                          ])),
    batch_size=batch_size_test, shuffle=True)

# ExperimentNetwork Definition
class UpdatedExperimentNetwork(nn.Module):
    def __init__(self, num_conv_layers=2, conv_filters_per_layer=10, dropout_rate=0.5):
        super(UpdatedExperimentNetwork, self).__init__()
        self.layers = nn.Sequential()
        in_channels = 1

        for i in range(num_conv_layers):
            out_channels = conv_filters_per_layer * (2 ** i)
            self.layers.add_module("conv{}".format(i), nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.layers.add_module("relu{}".format(i), nn.ReLU())
            self.layers.add_module("pool{}".format(i), nn.MaxPool2d(2))
            self.layers.add_module("dropout{}".format(i), nn.Dropout(dropout_rate))
            in_channels = out_channels

        self.fc1 = nn.Linear(in_channels * (28 // (2 ** num_conv_layers)) ** 2, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Training Function
def train(model, device, train_loader, optimizer):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation Function
def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    total_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return total_loss, accuracy

# Main Experiment Function
def main_experiment():
    results = []
    configurations = [
        (num_layers, filters, dropout_rate)
        for num_layers in [1, 2, 3]
        for filters in [10, 40, 80]
        for dropout_rate in [0.25, 0.5, 0.75]
    ]

    for num_layers, filters, dropout_rate in configurations:
        model = UpdatedExperimentNetwork(num_layers, filters, dropout_rate).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        for epoch in range(num_epochs):
            train(model, device, train_loader, optimizer)
        loss, accuracy = evaluate(model, device, test_loader)
        results.append((num_layers, filters, dropout_rate, accuracy, loss))
        print(f"Layers: {num_layers}, Filters: {filters}, Dropout: {dropout_rate}, Accuracy: {accuracy}, Loss: {loss}")

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    accuracies = [acc for _, _, _, acc, _ in results]
    losses = [loss for _, _, _, _, loss in results]
    labels = [f"L:{l} F:{f} D:{d}" for l, f, d, _, _ in results]

    axes[0].bar(labels, accuracies, color='skyblue')
    axes[0].set_title('Accuracy per Configuration')
    axes[0].set_xticklabels(labels, rotation=90)
    axes[0].set_ylabel('Accuracy (%)')

    axes[1].bar(labels, losses, color='lightgreen')
    axes[1].set_title('Loss per Configuration')
    axes[1].set_xticklabels(labels, rotation=90)
    axes[1].set_ylabel('Loss')

    plt.tight_layout()
    plt.savefig('experiment_results.png')
    plt.show()

if __name__ == "__main__":
    main_experiment()
