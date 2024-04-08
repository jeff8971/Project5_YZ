# Project5_YZ: Recognition using Deep Networks

[Project Repository](https://github.com/jeff8971/Project5_YZ)

## Overview
This project, Project 5: Recognition using Deep Networks by Yuan Zhao for CS5330 Pattern Recognition & Computer Vision, focuses on the application of deep learning techniques for image recognition tasks. The project explores the training and testing of neural networks on the MNIST dataset, the examination of network layers, transfer learning for recognizing Greek letters, and experimenting with network design. The code is developed using Python, PyTorch, and OpenCV.

## System Environment
- **IDE**: JetBrains PyCharm 2023.3.2 or any other Python IDE
- **Interpreter**: Python 3.10.2 or any compatible version
- **System Configuration**: Compatible with various operating systems, including macOS, Linux, and Windows.
- **Dependencies**: Requires PyTorch for deep learning functionalities and OpenCV for image processing.

## Project Structure
- `my_network.py`: Defines the neural network architecture.
- `task1_train.py`: Script for training the network on the MNIST dataset.
- `task1_test.py`: Script for testing the network on the MNIST dataset and custom handwritten digits.
- `task1_train_network.py`: Utility functions for training and testing the network.
- `task2_examine.py`: Script for examining the first layer of the neural network.
- `task3_greek.py`: Script for applying transfer learning to recognize Greek letters.
- `task4_design_experiment.py`: Script for experimenting with different network designs.
- `model/`: Contains the saved model and optimizer state.
- `data/`: Contains the MNIST and FashionMNIST datasets, along with custom datasets for Greek letters and handwritten digits.
- `experiment_results.png`, `loss_plot.png`, `training_testing_loss.png`, `training_testing_metrics.png`: Visualization of training and testing results.
- `extension.py`: Implementation of the Gabor filter for the model.
  project.

- [Extra Greek letters](https://github.com/jeff8971/Project5_YZ)

## Features
- Training and testing a neural network on the MNIST dataset.
- Visual examination of network layers.
- Transfer learning for recognizing Greek letters.
- Experimentation with network design parameters.

## Getting Started
### Prerequisites
- Install Python 3.10.2 or a compatible version.
- Install required Python packages: PyTorch, OpenCV, and others as specified in the project requirements.

### Installation
1. Clone the repository:
```bash
git clone https://github.com/jeff8971/Project5_YZ.git
```

2. Navigate to the project directory:
```cd Project5_YZ```


### Running the Applications
- To train the MNIST network:
```python task1_train.py```


- To test the network on MNIST and custom handwritten data:
```python task1_test.py```


- To examine the first layer of the network:
```python task2_examine.py```


- To transfer the network to Greek letters:
```python task3_greek.py```


- To run custom design and experiment network:
```python task4_design_experiment.py```

- To run the extension:
```python extension.py```


## Utilization of Travel Days

2 days