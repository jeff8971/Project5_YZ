#!/usr/bin/env python3
"""
Filename: my_network.py
Author: Yuan Zhao
Email: zhao.yuan2@northeastern.edu
Description: This module implements the MyNetwork class, designed for recognizing digits in images. It constructs a neural network using PyTorch, featuring convolutional and fully connected layers tailored for image classification tasks.
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

    # Constants for network parameters
    KERNEL_SIZE = 5
    DROPOUT_RATE = 0.5
    FC1_OUTPUT_SIZE = 50
    FC2_OUTPUT_SIZE = 10
    INPUT_CHANNELS = 1
    FIRST_CONV_OUTPUT_CHANNELS = 10
    SECOND_CONV_OUTPUT_CHANNELS = 20
    FLATTENED_SIZE = 320
    MAX_POOL_SIZE = 2  # Added constant for the max pooling size

    def __init__(self):
        super(MyNetwork, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(self.INPUT_CHANNELS,
                               self.FIRST_CONV_OUTPUT_CHANNELS,
                               kernel_size=self.KERNEL_SIZE)
        self.conv2 = nn.Conv2d(self.FIRST_CONV_OUTPUT_CHANNELS,
                               self.SECOND_CONV_OUTPUT_CHANNELS,
                               kernel_size=self.KERNEL_SIZE)
        self.conv2_dropout = nn.Dropout2d(p=self.DROPOUT_RATE)

        # Fully connected layers
        self.fc1 = nn.Linear(self.FLATTENED_SIZE, self.FC1_OUTPUT_SIZE)
        self.fc2 = nn.Linear(self.FC1_OUTPUT_SIZE, self.FC2_OUTPUT_SIZE)

    def forward(self, x):
        """
        Defines the forward pass of the neural network with input x.
        """
        # Apply ReLU activation function after max pooling with a 2x2 window size
        x = F.relu(F.max_pool2d(self.conv1(x), self.MAX_POOL_SIZE))
        x = F.relu(F.max_pool2d(self.conv2_dropout(self.conv2(x)),
                                self.MAX_POOL_SIZE))

        # Flatten the tensor for the fully connected layer
        x = x.view(-1, self.FLATTENED_SIZE)

        # Apply ReLU activation function after the first fully connected layer
        x = F.relu(self.fc1(x))

        # Output layer with log_softmax function applied
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
