# import statements
import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# define the convolutional neural network
class Net(nn.Module):
    # function to initialize the layers of the network
    def __init__(self, dropout: float = 0.5, dense: int = 60):
        super(Net, self).__init__()
        # conv1 is a convolutional layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # conv2 is a convolutional layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # this is a dropout layer with the given dropout rate
        self.dropout = nn.Dropout2d(dropout)
        # fc1 is a fully connected layer with given number of neurons
        self.fc1 = nn.Linear(320, dense)
        # fc2 is a fully connected layer with 10 neurons
        self.fc2 = nn.Linear(dense, 10)

    # function to compute a forward pass for the network
    def forward(self, x):
        # x is 1x28x28. conv1 is applied here
        x = self.conv1(x)
        # x is 10x24x24
        # a max pooling layer with a 2x2 window and a ReLU function are applied here
        x = F.relu(F.max_pool2d(x, 2))
        # x is 10x12x12. conv2 is applied here
        x = self.conv2(x)
        # x is 20x8x8. a dropout layer is applied here
        x = self.dropout(x)
        # x is 20x8x8
        # a max pooling layer with a 2x2 window and a ReLU function are applied here
        x = F.relu(F.max_pool2d(x, 2))
        # x is 20x4x4. x is flattened here
        x = x.view(-1, 320)
        # x is 320x1. fc1 and a ReLU function are applied here
        x = F.relu(self.fc1(x))
        # x is dense x 1. fc2 and a log softmax function are applied here
        x = F.log_softmax(self.fc2(x), dim=1)
        # x is 10x1
        return x


# function to train the network
def run_experiment(
    network: Net,
    optimizer: optim.SGD,
    train_loader: DataLoader,
    test_loader: DataLoader,
):
    # for each epoch
    for epoch in range(1, 6):
        # train the network
        for batch_idx, (data, target) in enumerate(train_loader):
            # set the gradients to zero
            optimizer.zero_grad()
            # compute the output of the network
            output = network(data)
            # compute the loss
            loss = F.nll_loss(output, target)
            # compute the gradients
            loss.backward()
            # update the weights
            optimizer.step()
            # for every 100 batches
            if batch_idx % 100 == 0:
                # print the loss
                print(
                    "Train epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    # evaluate the network
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = network(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# main function
def main(argv):
    # import MNIST fashion train data
    train_data = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    )

    # import MNIST test data
    test_data = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    )

    # initialize the current params
    # best result {'dropout': 0.05, 'dense_nodes': 160, 'batch_size': 8}
    current_params = {"dropout": 0.5, "dense_nodes": 50, "batch_size": 64}

    # define the parameter ranges
    param_ranges = {
        "dropout": np.arange(0.05, 1, 0.05),
        "dense_nodes": range(16, 176, 8),
        "batch_size": range(4, 80, 4),
    }

    # track the accuracy for each parameter
    accuracy_tracking = {param: [] for param in current_params}

    # experimentation loop
    for param in current_params:
        best_accuracy = 0
        best_value = current_params[param]

        for value in param_ranges[param]:
            # print the current parameter and value
            print("Working on {} as {}".format(param, value))
            # update the current parameter
            current_params[param] = value

            # initialize the network
            network = Net(current_params["dropout"], current_params["dense_nodes"])
            # initialize the optimizer
            optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)
            # create a data loader for the train data
            train_loader = DataLoader(
                train_data, batch_size=current_params["batch_size"], shuffle=True
            )
            # create a data loader for the test data
            test_loader = DataLoader(test_data, batch_size=1000, shuffle=True)

            # run the experiment
            accuracy = run_experiment(
                network,
                optimizer,
                train_loader,
                test_loader,
            )

            # keep track of the accuracy
            accuracy_tracking[param].append((value, accuracy))

            # update the best accuracy and value
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_value = value

            # print the current best accuracy and value
            print(
                "Current best accuracy for {} is {} with value {}".format(
                    param, best_accuracy, best_value
                )
            )

        # print the best accuracy and value
        current_params[param] = best_value
        print(
            "Optimized {} to {} with accuracy {}".format(
                param, best_value, best_accuracy
            )
        )

    # plot the experiment results
    for param, values in accuracy_tracking.items():
        plt.figure()
        plt.plot([v[0] for v in values], [v[1] for v in values], marker="o")
        plt.title(f"Accuracy vs {param}")
        plt.xlabel(param)
        plt.ylabel("Accuracy")
        plt.grid(True)
    plt.show()

    # print the final optimized parameters
    print("Final optimized parameters:", current_params)

    return


if __name__ == "__main__":
    main(sys.argv)