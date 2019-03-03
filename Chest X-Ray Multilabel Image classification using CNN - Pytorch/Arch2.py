#!/usr/bin/env python
# coding: utf-8

# In[ ]:


################################################################################
# CSE 253: Programming Assignment 3
# Winter 2019
# Code author: Jenny Hamer (+ modifications by Tejash Desai)
#
# Filename: baseline_cnn.py
#
# Description:
#
# This file contains the starter code for the baseline architecture you will use
# to get a little practice with PyTorch and compare the results of with your
# improved architecture.
#
# Be sure to fill in the code in the areas marked #TODO.
################################################################################


# PyTorch and neural network imports
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

# Data utils and dataloader
import torchvision
from torchvision import transforms, utils
from xray_dataloader_zscored import ChestXrayDataset, create_split_loaders

import matplotlib.pyplot as plt
import numpy as np
import os



class Arch2CNN(nn.Module):
    """
<<<<<<< HEAD
    conv1 -> maxpool -> conv2 -> maxpool -> conv3 -> conv4 ->maxpool -> conv5 -> conv6 -> maxpool -> conv7 -> conv8 -> maxpool -> fc1 -> fc2 -> fc3 (outputs)
=======
    conv1 -> conv2 -> maxpool -> conv3 -> conv4 -> conv5 -> maxpool -> fc1 -> fc2 -> fc3 (outputs)
>>>>>>> 6652e3cfb72835ac4a7c802c9a703b59d5f63ae6
    """

    def __init__(self):
        super(Arch2CNN, self).__init__()

        # conv1: 1 input channel, 4 output channels, [3x3] kernel size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)

        # Add batch-normalization to the outputs of conv1
        self.conv1_normed = nn.BatchNorm2d(4)

        # Initialized weights using the Xavier-Normal method
        torch_init.xavier_normal_(self.conv1.weight)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)

        #TODO: Fill in the remaining initializations replacing each '_' with
        # the necessary value based on the provided specs for each layer

        #TODO: conv2: 4 input channels, 8 output channels, [3x3] kernel, initialization: xavier
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)
        self.conv2_normed = nn.BatchNorm2d(8)
        torch_init.xavier_normal_(self.conv2.weight)
        #Maxpool
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1)


        #TODO: conv3: X input channels, 12 output channels, [8x8] kernel, initialization: xavier
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv3_normed = nn.BatchNorm2d(16)
        torch_init.xavier_normal_(self.conv3.weight)
        #TODO: conv4: X input channels, 10 output channels, [6x6] kernel, initialization: xavier
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.conv4_normed = nn.BatchNorm2d(16)
        torch_init.xavier_normal_(self.conv4.weight)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1)
        #TODO: conv5: X input channels, 8 output channels, [5x5] kernel, initialization: xavier
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3)
        self.conv5_normed = nn.BatchNorm2d(8)
        torch_init.xavier_normal_(self.conv5.weight)

        self.conv6 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3)
        self.conv6_normed = nn.BatchNorm2d(8)
        torch_init.xavier_normal_(self.conv6.weight)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1)

        #TODO: Apply max-pooling with a [3x3] kernel using tiling (*NO SLIDING WINDOW*)
        self.conv7 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3)
        self.conv7_normed = nn.BatchNorm2d(8)
        torch_init.xavier_normal_(self.conv7.weight)

        self.conv8 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3)
        self.conv8_normed = nn.BatchNorm2d(8)
        torch_init.xavier_normal_(self.conv8.weight)
        self.pool5 = nn.MaxPool2d(kernel_size=4, stride=4)


        # Define 2 fully connected layers:
        #TODO: fc1
        self.fc1 = nn.Linear(in_features=122*122*8, out_features=512)

        self.fc1_normed = nn.BatchNorm1d(512)
        torch_init.xavier_normal_(self.fc1.weight)

        #TODO: fc2
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc2_normed = nn.BatchNorm1d(128)
        torch_init.xavier_normal_(self.fc2.weight)

        #TODO: fc3
        self.fc3 = nn.Linear(in_features=128, out_features=14)
        torch_init.xavier_normal_(self.fc3.weight)

        #TODO: Output layer: what should out_features be?
        self.out_features = 14


    def forward(self, batch):
        """Pass the batch of images through each layer of the network, applying
        non-linearities after each layer.

        Note that this function *needs* to be called "forward" for PyTorch to
        automagically perform the forward pass.

        Params:
        -------
        - batch: (Tensor) An input batch of images

        Returns:
        --------
        - logits: (Variable) The output of the network
        """

        # Apply first convolution, followed by ReLU non-linearity;
        # use batch-normalization on its outputs
        batch = func.rrelu(self.conv1_normed(self.conv1(batch)))

        batch = self.pool1(batch)
        # Apply conv2 and conv3 similarly
        batch = func.rrelu(self.conv2_normed(self.conv2(batch)))
        batch = self.pool2(batch)

        batch = func.rrelu(self.conv3_normed(self.conv3(batch)))
        batch = func.rrelu(self.conv4_normed(self.conv4(batch)))
        batch = self.pool3(batch)
        batch = func.rrelu(self.conv5_normed(self.conv5(batch)))
        batch = func.rrelu(self.conv6_normed(self.conv6(batch)))
        # Pass the output of conv3 to the pooling layer
        batch = self.pool4(batch)
        batch = func.rrelu(self.conv7_normed(self.conv7(batch)))
        batch = func.rrelu(self.conv8_normed(self.conv8(batch)))
        # Pass the output of conv3 to the pooling layer
        batch = self.pool5(batch)

        # Reshape the output of the conv3 to pass to fully-connected layer
        batch = batch.view(-1, self.num_flat_features(batch))

        # Connect the reshaped features of the pooled conv3 to fc1
        batch =  func.rrelu(self.fc1_normed(self.fc1(batch)))
        batch =  func.rrelu(self.fc2_normed(self.fc2(batch)))

        # Connect fc1 to fc2 - this layer is slightly different than the rest (why?)
        batch = self.fc3(batch)


        # Return the class predictions
        #TODO: apply an activition function to 'batch'
        #batch = func.sigmoid(batch)
        return batch

    def num_flat_features(self, inputs):

        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s

        return num_features

