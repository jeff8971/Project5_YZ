# Your Name Here and a short header describing the purpose of this script

# Import statements
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Class definitions
class MyNetwork(nn.Module):
    """
    A neural network class for digit recognition with the following layers:
    - A convolution layer with 10 5x5 filters
    - A max pooling layer with a 2x2 window and a ReLU function applied.
    - A convolution layer with 20 5x5 filters and a dropout layer with a 0.5 rate
    - A max pooling layer with a 2x2 window and a ReLU function applied
    - A fully connected Linear layer with 50 nodes and a ReLU function on the output
    - A final fully connected Linear layer with 10 nodes and the log_softmax function applied to the output.
    """

    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_dropout = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_dropout(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Useful functions
def train_network(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    return train_loss


def test_network(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy



def visualize_test_images(test_loader):
    """Function to visualize the first six example digits from the test set."""
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f'Label: {example_targets[i]}')
        plt.xticks([])
        plt.yticks([])
    plt.show()


# Main function
def main(argv):
    """
    The main function of the script that sets up data loaders,
    initializes the model, trains, evaluates, and saves it, and
    also exports the training and testing diagrams.
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True,
                                  transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Model, optimizer
    model = MyNetwork().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # Train and evaluate
    epochs = 5
    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(1, epochs + 1):
        train_loss = train_network(model, device, train_loader, optimizer,
                                   epoch)
        train_losses.append(train_loss)

        test_loss, test_accuracy = test_network(model, device, test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(
            f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

    # Plotting and saving the diagram
    plt.figure(figsize=(10, 5))

    # Plot for losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Testing Loss')
    plt.legend()

    # Plot for accuracy
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Testing Accuracy')
    plt.legend()

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(
        'training_testing_metrics.png')  # Specify the path and filename

    # Optionally show the plot
    plt.show()

    # Save the model
    torch.save(model.state_dict(), 'mnist_model.pth')


if __name__ == "__main__":
    main(sys.argv)

