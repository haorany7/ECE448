# submitted.py

"""
This is the module you will submit to the autograder.

There are several function and variable definitions, here, that raise RuntimeErrors.
You should replace each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

If you are not sure how to use PyTorch, you may want to take a look at the tutorial.
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize


from models import resnet18


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


"""
1.  Define and build a PyTorch Dataset
"""
class CIFAR10(Dataset):
    def __init__(self, data_files, transform=None, target_transform=None):
        """
        Initialize your dataset here. Note that transform and target_transform
        correspond to your data transformations for train and test respectively.
        """
        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform
        
        def unpickle(file):
            import pickle
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        for data_file in data_files:
            batch = unpickle(data_file)
            self.data += batch[b'data']
            self.targets += batch[b'labels']

        for i in range(len(self.data)):
            self.data[i] = np.transpose(np.reshape(self.data[i], (3, 32, 32)), (1, 2, 0))

    def __len__(self):
        """
        Return the length of your dataset here.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Obtain a sample from your dataset. 
        Parameters:
            x:      an integer, used to index into your data.
        Outputs:
            y:      a tuple (image, label), although this is arbitrary so you can use whatever you would like.
        """
        image, label = self.data[idx], self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
    

def get_preprocess_transform(mode):
    """
    Parameters:
        mode:           "train" or "test" mode to obtain the corresponding transform
    Outputs:
        transform:      a torchvision transforms object e.g. transforms.Compose([...]) etc.
    """
    if mode == 'train':
        preprocess = transforms.Compose([
            ToPILImage(),  # Add this line to convert numpy ndarray to PIL Image
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif mode == 'test':
        preprocess = transforms.Compose([
            ToPILImage(),  # Add this line to convert numpy ndarray to PIL Image
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        raise ValueError("Invalid mode. Choose either 'train' or 'test'.")
    
    return preprocess


def build_dataset(data_files, transform=None):
    """
    Parameters:
        data_files:      a list of strings e.g. "cifar10_batches/data_batch_1" corresponding to the CIFAR10 files to load data
        transform:       the preprocessing transform to be used when loading a dataset sample
    Outputs:
        dataset:      a PyTorch dataset object to be used in training/testing
    """
    dataset = CIFAR10(data_files=data_files, transform=transform)
    return dataset
"""
2.  Build a PyTorch DataLoader
"""
def build_dataloader(dataset, loader_params):
    """
    Parameters:
        dataset:         a PyTorch dataset to load data
        loader_params:   a dict containing all the parameters for the loader. 
        
    Please ensure that loader_params contains the keys "batch_size" and "shuffle" corresponding to those 
    respective parameters in the PyTorch DataLoader class. 
    Outputs:
        dataloader:      a PyTorch dataloader object to be used in training/testing
    """
    dataloader = DataLoader(dataset, **loader_params)
    return dataloader

"""
3. (a) Build a neural network class.
"""
class FinetuneNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here. Remember that you will be performing finetuning
        in this network so follow these steps:
        
        1. Initialize convolutional backbone with pretrained model parameters.
        2. Freeze convolutional backbone.
        3. Initialize linear layer(s). 
        """
        super().__init__()
        ################# Your Code Starts Here #################

        # 1. Initialize convolutional backbone with pretrained model parameters
        self.model = resnet18()
        self.model.load_state_dict(torch.load('resnet18.pt'))

        # 2. Freeze convolutional backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # 3. Initialize linear layer(s)
        self.model.fc = torch.nn.Linear(512, 8)
        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################

        return self.model(x)
        ################## Your Code Ends here ##################


"""
3. (b)  Build a model
"""
def build_model(trained=False):
    """
    Parameters:
        trained:         a bool value specifying whether to use a model checkpoint

    Outputs:
        model:           the model to be used for training/testing
    """
    net = FinetuneNet()
    return net


"""
4.  Build a PyTorch optimizer
"""
def build_optimizer(optim_type, model_params, hparams):
    """
    Parameters:
        optim_type:      the optimizer type e.g. "Adam" or "SGD"
        model_params:    the model parameters to be optimized
        hparams:         the hyperparameters (dict type) for usage with learning rate 

    Outputs:
        optimizer:       a PyTorch optimizer object to be used in training
    """
    if optim_type == "Adam":
        optimizer = torch.optim.Adam(params=model_params, **hparams)
    else:
        optimizer = torch.optim.SGD(params=model_params, **hparams)
    return optimizer


"""
5. Training loop for model
"""
def train(train_dataloader, model, loss_fn, optimizer):
    """
    Train your neural network.

    Iterate over all the batches in dataloader:
        1.  The model makes a prediction.
        2.  Calculate the error in the prediction (loss).
        3.  Zero the gradients of the optimizer.
        4.  Perform backpropagation on the loss.
        5.  Step the optimizer.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        model:              the model to be trained
        loss_fn:            loss function
        optimizer:          optimizer
    """

    ################# Your Code Starts Here #################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the appropriate device
    model.train()     # Set the model to training mode

    for images, labels in train_dataloader:
        # Move images and labels to the appropriate device
        images, labels = images.to(device), labels.to(device)

        # 1. The model makes a prediction
        predictions = model(images)

        # 2. Calculate the error in the prediction (loss)
        loss = loss_fn(predictions, labels)

        # 3. Zero the gradients of the optimizer
        optimizer.zero_grad()

        # 4. Perform backpropagation on the loss
        loss.backward()

        # 5. Step the optimizer
        optimizer.step()
    ################## Your Code Ends here ##################


"""
6. Testing loop for model
"""
def test(test_dataloader, model):
    """
    This part is optional.

    You can write this part to monitor your model training process.

    Test your neural network.
        1.  Make sure gradient tracking is off, since testing set should only
            reflect the accuracy of your model and should not update your model.
        2.  The model makes a prediction.
        3.  Calculate the error in the prediction (loss).
        4.  Print the loss.

    Parameters:
        test_dataloader:    a dataloader for the testing set and labels
        model:              the model that you will use to make predictions


    Outputs:
        test_acc:           the output test accuracy (0.0 <= acc <= 1.0)
    """

    # test_loss = something
    # print("Test loss:", test_loss)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for features, labels in test_dataloader:
            features, labels = features.to(device), labels.to(device)
            predictions = model(features)

            correct_predictions += (predictions.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    test_accuracy = correct_predictions / total_samples
    return test_accuracy


"""
7. Full model training and testing
"""
def run_model():
    """
    The autograder will call this function and measure the accuracy of the returned model.
    Make sure you understand what this function does.
    Do not modify the signature of this function (names and parameters).

    Please run your full model training and testing within this function.

    Outputs:
        model:              trained model
    """
    # Define hyperparameters
    learning_rate = 0.001
    num_epochs = 1
    batch_size = 128
    # Define datapath
    data_files = [
        "cifar10_batches/data_batch_1",
        "cifar10_batches/data_batch_2",
        "cifar10_batches/data_batch_3",
        "cifar10_batches/data_batch_4",
        "cifar10_batches/data_batch_5"
    ]
    # Create train and test datasets with the appropriate transforms
    train_transform = get_preprocess_transform("train")
    test_transform = get_preprocess_transform("test")

    train_dataset = build_dataset(data_files, transform=train_transform)
    test_dataset = build_dataset(data_files=["cifar10_batches/test_batch"], transform=test_transform)

    # Create train and test dataloaders
    train_dataloader = build_dataloader(train_dataset, loader_params={"batch_size": batch_size, "shuffle": True})
    test_dataloader = build_dataloader(test_dataset, loader_params={"batch_size": batch_size, "shuffle": False})

    # Create your model, loss function, and optimizer
    model = build_model(trained=False)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer("Adam", model.parameters(), {"lr": learning_rate})

    # Train and test the model
    for epoch in range(num_epochs):
        train(train_dataloader, model, loss_fn, optimizer)
        test_acc = test(test_dataloader, model)
        print(f"Epoch: {epoch + 1}/{num_epochs}, Test Acc: {test_acc:.4f}")

    return model
