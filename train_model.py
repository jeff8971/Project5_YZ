#!/usr/bin/env python3
"""
train_model.py
Author: Yuan Zhao (zhao.yuan2@northeastern.edu)
Description: train and test the neural network.
Date: 2024-03-19
"""
# Import statements
import os
import torch
import torch.nn.functional as F


# train network function
def train_network(model, train_loader, optimizer, epoch,
                  train_losses, train_counter, batch_size, save_path):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Zero the gradients
        output = model(data)  # Output of the model
        loss = F.nll_loss(output, target)  # Calculate the loss
        loss.backward()  # Calculate the gradients
        optimizer.step()  # Update the weights

        if batch_idx % 10 == 0:  # Log progress every 10 batches
            print(
                '\rTraining Epoch: {}  Loss: {:.6f}  [{}/{} ({:.0f}%)]'.format(
                    epoch, loss.item(), batch_idx * len(data),
                    len(train_loader.dataset),
                                        100. * batch_idx / len(train_loader)),
                end='')

            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size) + (
                        (epoch - 1) * len(train_loader.dataset)))

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), save_path + '/model.pth')
            torch.save(optimizer.state_dict(), save_path + '/optimizer.pth')


def test_network(model, test_loader, test_losses):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
