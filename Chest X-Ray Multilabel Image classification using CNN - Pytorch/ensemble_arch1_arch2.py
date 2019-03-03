from Arch1 import Arch1CNN
from Arch2 import Arch2CNN
import torch
import numpy as np
import torchvision
from torchvision import models, transforms
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset,DataLoader
from xray_dataloader_zscored import ChestXrayDataset
#from itertools import izip

batch_size = 16
p_val= 0.1
p_test = 0.2

transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
transform1 = transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])

arch1model = Arch1CNN()
arch2model = Arch2CNN()
arch1model.load_state_dict(torch.load('arch1_dropout.pt'))
arch2model.load_state_dict(torch.load('arch2_new.pt'))

arch1model.eval()
arch2model.eval()

use_cuda =  torch.cuda.is_available()

if use_cuda:
    computing_device = torch.device("cuda")
    num_workers = 1
    pin_memory = True
    print("Testing on GPU")
else:
    computing_device = torch.device("cpu")
    num_workers = 0
    pin_memory = False
    print("Testing on CPU")


test_ind = np.loadtxt("test_ind.txt").astype(np.int32)
dataset = ChestXrayDataset(transform)
sample_test = SubsetRandomSampler(test_ind)
test_loader = DataLoader(dataset, batch_size=batch_size,
                             sampler=sample_test, num_workers=num_workers, pin_memory= pin_memory)
dataset2 = ChestXrayDataset(transform1)
test_loader2 = DataLoader(dataset2,batch_size = batch_size,sampler = sample_test,num_workers = num_workers, pin_memory = pin_memory)
models = [arch1model,arch2model];

confusionMatrix = np.zeros((15,15))
	
minibatch_number = 0
for (images1,labels),(images2,labels2) in zip(test_loader,test_loader2):
    print("Minibatch number",minibatch_number)
    minibatch_number += 1
    images1, labels = images1.to(computing_device), labels.to(computing_device)
    arch1model.to(computing_device)
    logitsarch1 = arch1model(images1)
    predictionarch1 = (logitsarch1.cpu().detach().numpy() > 0).astype(np.int32)
    del logitsarch1
    arch1model.cpu()
    arch2model.to(computing_device)
    images2 = images2.to(computing_device)
    logitsarch2 = arch2model(images2)
    predictionarch2 = (logitsarch2.cpu().detach().numpy() > 0).astype(np.int32)
    del logitsarch2
    arch2model.cpu()
    # Taking the union to retain as much predictions as possible
    # Each model could have learned a feature better and thus would predict some classes better
    prediction = predictionarch1+predictionarch2
    prediction[prediction == 2] = 1
    labelsArray = (labels.cpu().detach().numpy()).astype(np.int32)
    
    for row in range(len(prediction)):
        indexPrediction = np.where(prediction[row] == 1)[0] + 1
        indexLabels = np.where(labelsArray[row] == 1)[0] + 1

        #Remove common elements
        commonElements = np.intersect1d(indexPrediction, indexLabels)
        for i in commonElements:
            confusionMatrix[i,i] += 1
        excessPrediction = np.setdiff1d(indexPrediction,commonElements)
        excessLabels = np.setdiff1d(indexLabels,commonElements)
        #Zero pad if either of the two arrays is null
        if len(excessPrediction) == 0 :
            excessPrediction = np.zeros(1,np.int32)
        if len(excessLabels) == 0:
            excessLabels = np.zeros(1,np.int32)
        for i in excessPrediction:
            for j in excessLabels:
                confusionMatrix[i,j] += 1
np.savetxt('Confusion_matrix_Ensemble.txt',confusionMatrix)

				












