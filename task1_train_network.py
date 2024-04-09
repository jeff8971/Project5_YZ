#!/usr/bin/env python3
"""
task1_train_network.py
Author: Yuan Zhao (zhao.yuan2@northeastern.edu)
Description: train and test the neural network.
Date: 2024-03-30
"""


import os
import torch
import torch.nn.functional as F

# Constants for training
LOG_INTERVAL = 10
MODEL_FILENAME = 'model.pth'
OPTIMIZER_FILENAME = 'optimizer.pth'


def train_network(model, train_loader, optimizer, epoch, train_losses, train_counter, batch_size, save_path):
    """
    Trains the neural network model.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for the training data.
        optimizer: Optimizer used for model parameter updates.
        epoch: Current epoch number.
        train_losses: List to store loss values.
        train_counter: List to store the number of examples seen.
        batch_size: The size of each batch of training data.
        save_path: Path to save the model and optimizer states.
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Reset gradients for next step
        output = model(data)  # Compute model output
        loss = F.nll_loss(output, target)  # Calculate loss
        loss.backward()  # Backpropagate loss
        optimizer.step()  # Adjust model weights

        # Logging and saving state at specified interval
        if batch_idx % LOG_INTERVAL == 0:
            print(f'\rTraining Epoch: {epoch}  Loss: {loss.item():.6f}  [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]', end='')
            train_losses.append(loss.item())
            train_counter.append((batch_idx * batch_size) + ((epoch - 1) * len(train_loader.dataset)))
            os.makedirs(save_path, exist_ok=True)  # Ensure save directory exists
            torch.save(model.state_dict(), os.path.join(save_path, MODEL_FILENAME))
            torch.save(optimizer.state_dict(), os.path.join(save_path, OPTIMIZER_FILENAME))


def test_network(model, test_loader, test_losses):
    """
    Tests the neural network model.

    Args:
        model: The neural network model to test.
        test_loader: DataLoader for the test data.
        test_losses: List to store loss values from testing.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # No gradients needed for testing
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get index of max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f'\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')