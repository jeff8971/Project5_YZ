import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from my_network import MyNetwork  # Ensure this imports your MNIST model
from task1_test import test_network_MNIST


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

    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
    return train_losses


def evaluate_model(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)

    print(
        f'Test set: Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')

    return accuracy


def plot_metrics(train_losses, accuracy):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Batch number')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.bar(['Test Accuracy'], [accuracy], color='orange')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')

    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = initialize_model()
    print(model)

    train_loader, test_loader = prepare_data_loaders("./data/greek_train",
                                                     "./data/greek_test")
    optimizer = optim.SGD(model.fc2.parameters(), lr=0.01,
                          momentum=0.5)  # Optimize only the last layer

    train_losses = train_model(model, device, train_loader, optimizer, epochs=30)
    accuracy = evaluate_model(model, device, test_loader)

    plot_metrics(train_losses, accuracy)

    test_network_MNIST(model, test_loader, 12)
    plt.show()


if __name__ == "__main__":
    main()
