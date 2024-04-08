import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from my_network import MyNetwork

class GreekTransform:
    def __call__(self, x):
        x = transforms.functional.rgb_to_grayscale(x)
        x = transforms.functional.affine(x, angle=0, translate=(0, 0), scale=36 / 128, shear=0)
        x = transforms.functional.center_crop(x, output_size=(28, 28))
        return transforms.functional.invert(x)

def initialize_model():
    model = MyNetwork()
    model.load_state_dict(torch.load('./model/model.pth'))
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc2.in_features
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Use same batch_size for test
    return train_loader, test_loader

def train_model(model, device, train_loader, optimizer, epochs=3):
    model.to(device)
    model.train()
    train_losses = []
    train_counter = []
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_counter.append((epoch * len(train_loader.dataset)) + (batch_idx * train_loader.batch_size))
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

def plot_metrics(train_losses, train_counter, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_counter, train_losses, color='orange', label='Train Loss')
    plt.scatter([train_counter[-1] for _ in test_losses], test_losses, color='red', label='Test Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.title('Training and Test Loss')
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model()
    print(model)
    batch_size = 64
    train_loader, test_loader = prepare_data_loaders("./data/greek_train", "./data/greek_test", batch_size)
    optimizer = optim.SGD(model.fc2.parameters(), lr=0.01, momentum=0.5)
    epochs = 30

    train_losses, train_counter = train_model(model, device, train_loader, optimizer, epochs)
    test_losses = [evaluate_model(model, device, test_loader) for _ in range(epochs)]  # Collect test loss after each epoch

    plot_metrics(train_losses, train_counter, test_losses)

    # Uncomment the following line if you have the MNIST testing functionality
    # test_network_MNIST(model, test_loader)
    plt.show()

if __name__ == "__main__":
    main()
