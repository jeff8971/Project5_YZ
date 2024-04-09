#!/usr/bin/env python3
"""
Filename: task2_examine.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: trains and evaluates a neural network on Greek letter images
Date: 2024-03-30
"""


import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from my_network import MyNetwork  # Ensure this imports your MNIST model



CLASS_NAMES = {0: "alpha", 1: "beta", 2: "gamma"}

class GreekTransform:
    """
    Transforms Greek letter images to match MNIST formatting:
    grayscale, scaled, cropped, and inverted intensities.
    """
    def __init__(self):
        pass

    def __call__(self, x):
        # convert the image to grayscale
        x = transforms.functional.rgb_to_grayscale(x)
        # rotate the image 36 degrees clockwise
        x = transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        # crop the image to 28x28
        x = transforms.functional.center_crop(x, (28, 28))
        # invert the image
        return transforms.functional.invert(x)


def test_network_MNIST(network, test_loader, plot_num=9):
    """Test the network on MNIST data, print output values, predictions, labels, and plot digits."""
    network.eval()  # Set the network to evaluation mode.
    predicted_labels = []  # Store the predicted labels.
    class_names = {0: "alpha", 1: "beta", 2: "gamma"}  # Mapping of numeric predictions to class names.

    plt.figure(figsize=(8, 8))  # Adjust figure size as needed.

    with torch.no_grad():  # No need to track gradients for testing.
        for idx, (data, target) in enumerate(test_loader):
            if idx >= plot_num:  # Only process a certain number of images.
                break

            output = network(data)  # Compute the network's output.
            pred = output.argmax(dim=1, keepdim=True).item()  # Find the prediction as an integer.
            pred_class_name = class_names[pred]  # Map the numeric prediction to its class name.

            # Format raw output values directly, without converting to probabilities.
            formatted_output = [f"{value:.2f}" for value in output[0]]

            # Print formatted information about predictions and actual labels.
            print(f"Image {idx + 1}, output values: [{', '.join(formatted_output)}], prediction: {pred_class_name}, true label: {class_names[target.item()]}")
            predicted_labels.append(pred_class_name)  # Store the predicted label name for later.

            # Plotting: Adjust subplot to accommodate the specified number of images.
            ax = plt.subplot((plot_num + 2) // 3, 3, idx + 1)
            ax.imshow(data[0].squeeze().numpy(), cmap="gray", interpolation="none")  # Display the image.
            ax.set_title(f"Prediction: {predicted_labels[idx]}", pad=3)  # Set the title to show the prediction, adjust padding as needed.
            ax.axis('off')  # Hide the axes.

    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.5)  # Adjust the layout.
    plt.show()



def initialize_model():
    model = MyNetwork()
    model.load_state_dict(torch.load('./model/model.pth'))

    # Freeze all the layers in the pre-trained network
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last layer for 3 class classification (alpha, beta, gamma)
    num_features = model.fc2.in_features  # Assuming 'fc2' is the name of the last layer
    model.fc2 = nn.Linear(num_features, 3)

    return model


def prepare_data_loaders(train_path, test_path, batch_size=5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader


def train_model(model, device, train_loader, optimizer, epochs=3):
    model.to(device)
    model.train()
    train_losses = []
    train_counter = []  # To store the number of training examples seen for each training step

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_counter.append((batch_idx + 1) * len(data))  # Increment the counter by the batch size

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}')

    return train_losses, train_counter


def evaluate_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f'Test set: Accuracy: {correct}/{len(test_loader.dataset)} ({100. * accuracy:.0f}%)')
    return test_loss


def test_model(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).item()
            print(f'Predicted: "{CLASS_NAMES[pred]}", Actual: "{CLASS_NAMES[target.item()]}"')


def plot_metrics(train_losses, train_loader):
    # Calculate the number of training examples processed after each epoch for plotting
    train_counter = [i * len(train_loader.dataset) for i in
                     range(len(train_losses))]

    plt.figure(figsize=(10, 6))
    plt.plot(train_counter, train_losses, color='black', label='Train Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.title('Training Loss')
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = initialize_model()
    print(model)

    train_loader, test_loader = prepare_data_loaders("./data/greek_train",
                                                     "./data/greek_test", batch_size=5)
    optimizer = optim.SGD(model.fc2.parameters(), lr=0.01, momentum=0.5)
    epochs = 30

    train_losses, train_counter = train_model(model, device, train_loader, optimizer, epochs)

    test_losses = [evaluate_model(model, device, test_loader) for _ in range(epochs)]  # Test after each epoch
    test_model(model, device, test_loader)  # Call test_model to print out
    # predictions
    plot_metrics(train_losses, train_loader)
    test_network_MNIST(model, test_loader)

    plt.show()


if __name__ == "__main__":
    main()