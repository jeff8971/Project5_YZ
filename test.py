import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from network import MyNetwork


def test_network_MNIST(model, test_loader, plot_num=9):
    model.eval()  # Set the model to evaluation mode
    predictions = []

    # Set up the figure for plotting predictions
    # Adjust the figure
    plt.figure(figsize=(10, (plot_num//3 + 1) * 3))
    # size dynamically based on the plot_num
    # Disable gradient computation for inference
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            if idx == plot_num:
                # Stop after reaching the desired number of plotted digits
                break
            # Get model output for the current batch
            output = model(data)
            # Determine the predicted class
            pred = output.argmax(dim=1, keepdim=True)

            # Record the prediction for each image for potential future use
            predictions.append(pred.item())

            # Print output values, predicted label, and correct label for 
            # each image
            output_vals = ', '.join([f"{o:.2f}" for o in output[0]])
            print(f"Image {idx + 1} output: [{output_vals}]")
            print(f"Image {idx + 1} prediction: {pred[0][0].item()}")
            print(f"Image {idx + 1} label: {target.item()}")

            # Plot the image and its predicted label
            plt.subplot((plot_num + 1) // 3, 3, idx + 1)
            plt.imshow(data[0].squeeze().numpy(), cmap="gray", interpolation="none")
            plt.title(f"Pred: {pred[0][0].item()}", y=0.95)
            plt.xticks([])
            plt.yticks([])

    plt.tight_layout()
    plt.show()

    return predictions
    # analysis


def test_network_handwritten(network, test_loader):
    network.eval()  # Set the network to evaluation mode
    test_loss = 0  # Initialize the test loss
    correct = 0  # Initialize the count of correct predictions

    # Prepare for plotting the images
    # Adjust the figure size for better visibility
    plt.figure(figsize=(15, 6))
    # No need to track gradients for testing
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            # Assuming we only want to plot and evaluate the first 10 images
            if idx == 10:
                break
            # Get the output from the network
            output = network(data)
            # Sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # Get the index of the max log-probability as prediction
            pred = output.argmax(dim=1, keepdim=True)
            # Check against the correct label
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Plot the image along with the predicted label
            plt.subplot(2, 5, idx + 1)
            plt.imshow(data[0].squeeze().cpu().numpy(), cmap='gray',
                       interpolation='none')
            plt.title(f'Pred: {pred.item()}', pad=2)
            plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Report the final statistics
    # Average the loss over the number of batches processed
    test_loss /= idx + 1
    accuracy = 100. * correct / ((idx + 1) * test_loader.batch_size)
    print(
        f'\nTest Set: \n'
        f'Average Loss: {test_loss:.4f}\n'
        f'Accuracy: {correct}'
        f'/{(idx + 1) * test_loader.batch_size} ({accuracy:.0f}%)')


class HandwrittenTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        # convert to grayscale and invert
        x = transforms.functional.rgb_to_grayscale(x)
        x = transforms.functional.invert(x)
        return x


def main():
    # Load pre-trained network
    network = MyNetwork()
    network.load_state_dict(torch.load("./model/model.pth"))

    # Load MNIST test data
    test_data = datasets.MNIST(
        root='./data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    # Create a subset of the test data with the first 10 images
    subset_test_data = Subset(test_data, range(10))
    # Create a data loader for the test data
    test_loader = DataLoader(subset_test_data, batch_size=1)

    # Test the network on the MNIST dataset
    test_network_MNIST(network, test_loader)

    # Load handwritten digits test data
    handwritten_transform = transforms.Compose([
        HandwrittenTransform(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    handwritten_data = datasets.ImageFolder(root='./data/handwritten',
                                             transform=handwritten_transform)
    handwritten_loader = DataLoader(handwritten_data, batch_size=1)

    # Test the network on the handwritten digits dataset
    test_network_handwritten(network, handwritten_loader)

    plt.show()

    return


if __name__ == '__main__':
    main()