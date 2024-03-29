#!/usr/bin/env python3
"""
network.py
Author: Yuan Zhao (zhao.yuan2@northeastern.edu)
Description: module containing the MyNetwork class for digit recognition.
Date: 2024-03-19
"""



import torch.nn.functional as F
import torch.nn as nn


# MyNetwork class
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
