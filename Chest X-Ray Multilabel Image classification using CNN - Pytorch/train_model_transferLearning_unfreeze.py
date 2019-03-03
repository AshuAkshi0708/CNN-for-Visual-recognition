#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from Arch1 import *
#from transferLearning import Arch1CNN
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
import torchvision
from torchvision import models,transforms
import numpy as np
from xray_dataloader_zscored_TL import get_weights,create_split_loaders,ChestXrayDataset


# Setup: initialize the hyperparameters/variables
num_epochs = 50         # Number of full passes through the dataset
#<<<<<<< HEAD
batch_size = 32   # Number of samples in each minibatch
#=======
#batch_size = 64          # Number of samples in each minibatch
#>>>>>>> 6652e3cfb72835ac4a7c802c9a703b59d5f63ae6
learning_rate = 0.001
seed = np.random.seed(1) # Seed the random number generator for reproducibility
p_val = 0.1              # Percent of the overall dataset to reserve for validation
p_test = 0.2             # Percent of the overall dataset to reserve for testing

#TODO: Convert to Tensor - you can later add other transformations, such as Scaling here
transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
transform2 = transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(p=1.0),transforms.ToTensor()])
#weights = get_weights();
# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")


#Get class imbalance weights

weights = get_weights()
weights = weights.to(computing_device)

# Setup the training, validation, and testing dataloaders
train_loader1, val_loader1, test_loader1 = create_split_loaders(batch_size, seed, transform=transform,
                                                             p_val=p_val, p_test=p_test,
                                                             shuffle=True, show_sample=False,
                                                             extras=extras)

train_loader2, val_loader2, test_loader2 = create_split_loaders(batch_size, seed, transform=transform2,
                                                             p_val=p_val, p_test=p_test,
                                                             shuffle=True, show_sample=False,
                                                             extras=extras)
train_loader_list = [train_loader1,train_loader2]
#train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([train_loader,train_loader2]))
val_loader_list = [val_loader1, val_loader2]
test_loader_list=[test_loader1,test_loader2]
#val_loader = torch.utils.data.ConcatDataset([val_loader,val_loader2])
#test_loader = torch.utils.data.ConcatDataset([test_loader,test_loader2])



#print("")
print("1 VGG 16 BN \n 2 VGG 19 BN \n 3 Resnet 34  ")
#<<<<<<< HEAD
#input_model = int(input("Choose the model:"))
input_model = 3
input_setting = 0
print("1 Freeze parameters\n 2 Unfreeze parameters")



#input_setting = int(input("Enter your choice:"))
#=======
#input_model = int(input("Choose the model:"))
print("1 Freeze parameters\n 2 Unfreeze parameters")
#input_setting = int(input("Enter your choice:"))
#>>>>>>> 6652e3cfb72835ac4a7c802c9a703b59d5f63ae6




# Instantiate a BasicCNN to run on the GPU or CPU based on CUDA support
## VGG_16_bn
if input_model == 1:
    model = models.vgg16_bn(pretrained=True)
    n_inputs = model.classifier[6].in_features
elif input_model == 2:
## VGG_19_bn
    model = models.vgg19_bn(pretrained = True)
    n_inputs = model.classifier[6].in_features
## VGG_19_bn
elif input_model == 3:
    model = models.resnet34(pretrained = False)
    n_inputs = model.fc.in_features
if input_setting == 1:
    for param in model.features.parameters():## Freezing all the layers
        param.requires_grad = False
    for param in model.classifier.parameters():## Freezing all the layers
        param.requires_grad = False
# add last linear layer (n_inputs -> 5 flower classes)
# new layers automatically have requires_grad = True
last_layer = nn.Linear(n_inputs, 14)
##VGG16 and VGG19 last layer change
if input_model == 1 or input_model == 2:
    model.classifier[6] = last_layer
elif input_model ==3:
    model.fc = last_layer

# print out the model structure
print(model)
model.load_state_dict(torch.load('TL_trained_unfreeze_1.pt'))
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)









#TODO: Define the loss criterion and instantiate the gradient descent optimizer
#<<<<<<< HEAD
criterion = torch.nn.MultiLabelSoftMarginLoss(weight = weights) #TODO - loss criteria are defined in the torch.nn package
#criterion = torch.nn.BCELoss()
#=======
#criterion = torch.nn.MultiLabelSoftMarginLoss() #TODO - loss criteria are defined in the torch.nn package
#criterion = torch.nn.BCELoss()
#>>>>>>> 6652e3cfb72835ac4a7c802c9a703b59d5f63ae6
#TODO: Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate) #TODO - optimizers are defined in the torch.optim package

print("initialised criterion and architecture!!")





# In[ ]:


# Track the loss across training
total_loss = []
avg_minibatch_loss = []
validation_loss = []

# Begin training procedure
for epoch in range(num_epochs):
    print("Entered the Training loop : epoch ", epoch)

    N = 50
    N_minibatch_loss = 0.0

    # Get the next minibatch of images, labels for training
    model.train()
    for train_loader in train_loader_list:
        for minibatch_count, (images, labels) in enumerate(train_loader, 0):

            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            images, labels = images.to(computing_device), labels.to(computing_device)

            # Zero out the stored gradient (buffer) from the previous iteration
            optimizer.zero_grad()

            # Perform the forward pass through the network and compute the loss
            outputs = model(images)
#<<<<<<< HEAD
            loss = criterion(outputs, labels)
#=======
#            loss = criterion(torch.sigmoid(outputs), labels)
#>>>>>>> 6652e3cfb72835ac4a7c802c9a703b59d5f63ae6

            # Automagically compute the gradients and backpropagate the loss through the network
            loss.backward()

            # Update the weights
            optimizer.step()

            # Add this iteration's loss to the total_loss
            total_loss.append(loss.item())
            N_minibatch_loss += loss

            #TODO: Implement cross-validation

            if minibatch_count % N == 0:

                # Print the loss averaged over the last N mini-batches
                N_minibatch_loss /= N
                print('Epoch %d, average minibatch %d loss: %.3f' %
                    (epoch + 1, minibatch_count, N_minibatch_loss))

                # Add the averaged loss over N minibatches and reset the counter
                avg_minibatch_loss.append(N_minibatch_loss)
                N_minibatch_loss = 0.0

    print("Finished", epoch + 1, "epochs of training")
    torch.save(model.state_dict(),"TL_trained_unfreeze_2.pt")

    #Validation
    temp_validation = 0
    model.eval()
    for val_loader in val_loader_list:
        for minibatch_count,(images,labels) in enumerate(val_loader,0):
            images,labels = images.to(computing_device),labels.to(computing_device)
            outputs = model(images)
            loss = criterion(outputs,labels)
            temp_validation += loss.item()
            del outputs,loss

    print("Validation loss after ",epoch," epochs=",temp_validation)
    validation_loss.append(temp_validation)
    if epoch >= 1 and validation_loss[-1] > validation_loss[-2]:
        break

print("Training complete after", epoch, "epochs")
torch.save(model.state_dict(),"TL_trained_unfreeze_2.pt")
