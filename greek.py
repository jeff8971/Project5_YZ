# import statements
import os
import sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from task1_train import Net, train_network
from task1_test import test_network_MNIST


# greek data set transform
class GreekTransform:
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


# main function
def main(argv):
    # read the model from file
    network = Net()
    network.load_state_dict(torch.load("./results/MNIST/model.pth"))

    # freeze the parameters of the network
    for param in network.parameters():
        param.requires_grad = False

    # change the last layer of the network to have 4 neurons
    network.fc2 = nn.Linear(network.fc2.in_features, 4)

    # print the network
    print(network)

    # import greek letters train data
    train_data = datasets.ImageFolder(
        root="./data/greek_train",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                GreekTransform(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    # create a data loader for the train data
    train_loader = DataLoader(train_data, batch_size=5, shuffle=True)

    # import greek letters test data
    test_data = datasets.ImageFolder(
        root="./data/greek_test",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                GreekTransform(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
    # create a data loader for the test data
    test_loader = DataLoader(test_data, batch_size=1)

    # initialize the optimizer
    optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)
    # train the network for 30 epochs
    n_epochs = 30
    # initialize the training and testing losses and counters
    train_losses = []
    train_counter = []

    # train the network for 30 epochs
    for epoch in range(1, n_epochs + 1):
        train_network(
            network,
            optimizer,
            epoch,
            train_loader,
            train_losses,
            train_counter,
            5,
            "./results/greek",
        )

    # plot the training loss
    plt.plot(train_counter, train_losses, color="blue")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")

    # test the network on the greek images
    test_network_MNIST(network, test_loader, 12)
    plt.show()

    return


if __name__ == "__main__":
    main(sys.argv)
