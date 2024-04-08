# Standard library imports
import sys

# Third-party imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from my_network import MyNetwork


class GaborConvNet(MyNetwork):
    """
    A Convolutional Neural Network incorporating Gabor filters in the first layer.
    Inherits from MyNetwork.
    """

    def __init__(self, kernels, first_layer_check=False):
        super(GaborConvNet, self).__init__()
        self.kernels = kernels
        self.first_layer_check = first_layer_check

    def forward(self, x):
        conv = nn.Conv2d(1, 10, kernel_size=5)
        conv.weight = nn.Parameter(torch.tensor(self.kernels))

        if self.first_layer_check:
            return conv(x)

        x = F.relu(F.max_pool2d(conv(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.reshape(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def create_gabor_filters():
    """
    Generate a series of Gabor kernels.

    Returns:
        A numpy array containing 10 Gabor kernels with varying orientations.
    """
    angles = np.linspace(0, np.pi, 10)
    filters = [cv2.getGaborKernel((5, 5), 10, angle, 0.05, 0.05, 0, cv2.CV_32F)
               for angle in angles]
    filters = np.expand_dims(np.array(filters), 1)
    return filters


def evaluate_accuracy(model):
    """
    Calculates and prints the accuracy of the model on the test dataset.
    """
    correct = 0
    total = len(model.test_loader.dataset)
    with torch.no_grad():
        for data, targets in model.test_loader:
            outputs = model(data)
            predictions = outputs.argmax(dim=1, keepdim=True)
            correct += predictions.eq(
                targets.view_as(predictions)).sum().item()

    print(
        f'\nTest set: Accuracy: {correct}/{total} ({100. * correct / total:.0f}%)\n')


def visualize_first_layer(kernels, model, images):
    """
    Visualizes the effect of the Gabor filters in the first convolutional layer.
    """
    model.eval()
    with torch.no_grad():
        output = model(images[:1])

    fig, axes = plt.subplots(5, 4, figsize=(9, 6), xticks=[], yticks=[])
    for idx, ax in enumerate(axes.flat):
        img = kernels[idx // 2, 0] if idx % 2 == 0 else output[
            0, idx // 2].cpu()
        ax.imshow(img, cmap='gray')
    plt.show()


def main(args):
    """
    Main function to execute the model training and evaluation.
    """
    kernels = create_gabor_filters()
    model = GaborConvNet(kernels)
    model.load_state_dict(torch.load('./model/model.pth'))
    model.eval()

    images, _ = next(iter(model.test_loader))
    output = model(images[:1])
    prediction = output.argmax(dim=1, keepdim=True).item()
    print(f'Prediction for the first test image: {prediction}')

    evaluate_accuracy(model)

    first_layer_model = GaborConvNet(kernels, True)
    first_layer_model.load_state_dict(torch.load('./model/model.pth'))
    visualize_first_layer(kernels, first_layer_model, images)


if __name__ == "__main__":
    main(sys.argv)
