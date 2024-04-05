import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from my_network import MyNetwork
import cv2
import numpy as np


# Constants for dataset normalization, image preparation, and layout calculation
DATASET_MEAN = 0.1307
DATASET_STD = 0.3081
NUM_FILTERS_FIRST_LAYER = 10
IMAGE_DIMENSION = 28  # MNIST images are 28x28
FILTER_SIZE = 5  # Assuming the first layer uses 5x5 filters


class SubModel(MyNetwork):
    def __init__(self, layer_num):
        super().__init__()
        assert layer_num in [1, 2], "layer_num should be 1 or 2."
        self.layer_num = layer_num

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        if self.layer_num == 2:
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        return x


def visualize_filters_and_effects(network: torch.nn.Module, train_loader: torch.utils.data.DataLoader):
    network.eval()
    with torch.no_grad():
        filters = network.conv1.weight.data

        # Calculate layout for subplots
        rows = NUM_FILTERS_FIRST_LAYER // 4 + int(NUM_FILTERS_FIRST_LAYER % 4 != 0)
        cols = min(NUM_FILTERS_FIRST_LAYER, 4)

        # Visualize filters
        plt.figure(figsize=(cols * 3, rows * 3))
        for i in range(NUM_FILTERS_FIRST_LAYER):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(filters[i, 0], cmap='viridis', interpolation='none')
            plt.title(f'Filter {i + 1}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Prepare and visualize filter effects on an example image
        image, _ = next(iter(train_loader))
        image_np = image[0].squeeze().numpy()
        image_for_filtering = cv2.normalize(image_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        plt.figure(figsize=(cols * 6, rows * 3))
        for i in range(NUM_FILTERS_FIRST_LAYER):
            filter_np = filters[i, 0].numpy()
            filtered_image = cv2.filter2D(image_for_filtering, -1, filter_np)

            # Adjusting to the referenced subplot format
            plt.subplot(5, 4,
                        2 * i + 1)  # Assuming 5 rows and 4 cols layout as in reference
            plt.imshow(filter_np, cmap='gray')
            plt.axis('off')  # Simplifying axis visibility management

            plt.subplot(5, 4, 2 * i + 2)  # Next subplot for the filtered image
            plt.imshow(filtered_image, cmap='gray')
            plt.axis('off')  # Keeping axes off as in reference

        plt.tight_layout()
        plt.show()


def main(argv):
    network = MyNetwork()
    network.load_state_dict(torch.load("./model/model.pth"))

    mnist_train = datasets.MNIST(root='./data', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((DATASET_MEAN,), (DATASET_STD,))
                                 ]))
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=1, shuffle=True)

    visualize_filters_and_effects(network, train_loader)


if __name__ == "__main__":
    main(sys.argv)