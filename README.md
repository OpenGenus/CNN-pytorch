# CNN-pytorch
CNN model training and inference in PyTorch

## Project Overview
This project contains a simple Convolutional Neural Network (CNN) model implemented using PyTorch. The model is trained on the CIFAR-10 dataset for image classification tasks. This README provides instructions to set up, run, and evaluate the model.

## Prerequisites
The below dependencies are to be installed before running the training model. For the versions used, please refer the requirements.txt file.

- Python 3.7 or higher
- PyTorch
- torchvision
- Numpy

## Setup
1. Clone the repository using git clone <URL>
2. Check all the dependencies to be installed.
3. Train the model on a dataset : python simple_cnn_model.py
4. Model inference and evaluation : Once the best model is saved in this path : `saved_model/net_cifar10.pt` , go ahead and experiment with the model by fine-tuning on new data or different datasets. 
Some examples are given below : 

i. Load the saved model : 
```python
net.load_state_dict(torch.load('net_cifar10.pt'))
net.eval()  # Set the model to evaluation mode
print("Model loaded and set to evaluation mode.")

```
ii. Print the model architecture : 
```python
print(net)
for name, param in net.named_parameters():
    print(name, param.data)
```
iii. Visualize filters :

```python
import matplotlib.pyplot as plt
import torchvision

def visualize_filters(layer, num_filters=6):
    filters = layer.weight.data.clone().cpu().numpy()

    # Normalization
    (filters - filters.mean()) / filters.std()

    # Plot filters
    plt.figure(figsize=(num_filters, num_filters))
    for i in range(num_filters):
        plt.subplot(1, num_filters, i + 1)
        plt.imshow(filters[i, 0, :, :], cmap='plasma')
        plt.axis('on')
    plt.show()

# Example usage with a convolutional layer from the model
visualize_filters(net.conv1[0])  # Visualize filters from the first conv layer
```
## Further Exploration
The current model has a total accuracy of 78.11%. This can be further optimized to meet particular project requirements by modifying the cnn model and the optimizer, as well as the batch size and workers involved in training the model.
