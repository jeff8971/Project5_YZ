
import sys
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from network import MyNetwork
from train_network import train_network, test_network


def plot_image(dataset):
    fig, ax = plt.subplots(figsize=(15, 10))
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(dataset.data[i].numpy(), cmap='gray')
        plt.title(f'Label: {dataset.targets[i]}')
        plt.axis('off')
    plt.show()


def main(argc):
    # import train_data
    train_data = datasets.MNIST(root='./data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    # import test data
    test_data = datasets.MNIST(root='./data', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))

    # create a data loader for the train data
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    # create a data loader for the test data
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=True)

    # show first six example digits from the training set
    plot_image(train_data)

    # initialize the network
    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)
    # train the network for 5 epochs
    n_epochs = 5
    # initialize the training and testing losses and counters
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    # test the network before training
    test_network(network, test_loader, test_losses)
    # train the network for 5 epochs, and test it after each epoch
    for epoch in range(1, n_epochs + 1):
        train_network(
            network, train_loader, optimizer, epoch, train_losses,
            train_counter, 64, './model')
        test_network(network, test_loader, test_losses)

    # plot the training losses
    plt.plot(train_counter, train_losses, color='black')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')

    #save the plot
    plt.savefig('loss_plot.png', format='png', dpi=600)

    plt.show()

    return


if __name__ == "__main__":
    main(sys.argv)
