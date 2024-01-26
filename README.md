# Train_Test_Functions_for_Custom_Dataset

This repository contains Python scripts for training and testing machine learning models on a custom dataset using PyTorch. The main focus is on providing modular and reusable functions for training and testing loops.

## Files

### `train.py`

The `train.py` file includes a function for the training step of a PyTorch model. The function, `train_step`, takes as input a PyTorch model, a data loader, a loss function, and an optimizer. It iterates through the provided data loader, performs forward and backward passes, updates the model's parameters, and calculates the training loss and accuracy.

### `test.py`

The `test.py` file contains a function, `test_step`, for evaluating a trained model on a test set. Similar to the training step, it takes a model, a data loader, and a loss function as input. The function iterates through the test data, computes the test loss and accuracy, and returns the results.

## Usage

To use these functions in your project, follow these steps:

1. Import the necessary modules and functions:

```python
import torch
from train import train_step
from test import test_step


# Define your model, data loaders, loss function, and optimizer

# Training
train_loss, train_acc = train_step(model, train_dataloader, loss_function, optimizer)

# Testing
test_loss, test_acc = test_step(model, test_dataloader, loss_function)



