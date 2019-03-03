import numpy as np
#from baseline_cnn import BasicCNN
from Arch1 import Arch1CNN
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset,DataLoader
#from Arch1 import Arch1CNN
from xray_dataloader_zscored import ChestXrayDataset
from torchvision import transforms


batch_size = 64
p_val = 0.1
p_test = 0.2

transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])

#Load model
model = Arch1CNN()
model.load_state_dict(torch.load("arch1_dropout.pt"))
model.eval()
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

model = model.to(computing_device)

#Get testing images
test_ind = np.loadtxt("test_ind.txt").astype(np.int32)
dataset = ChestXrayDataset(transform)
sample_test = SubsetRandomSampler(test_ind)
test_loader = DataLoader(dataset, batch_size=batch_size,
                             sampler=sample_test, num_workers=num_workers,
                              pin_memory=pin_memory)


#Not really seeing a fully unknown sample

confusionMatrix = np.zeros((15,15))

for minibatch_number,(images,labels) in enumerate(test_loader, 0):
    print("Minibatch number",minibatch_number)
    images, labels = images.to(computing_device), labels.to(computing_device)
    logits = model(images)
    prediction = (logits.cpu().detach().numpy() > 0).astype(np.int32)
    labelsArray = (labels.cpu().detach().numpy()).astype(np.int32)
    del logits
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
np.savetxt('Confusion_matrix_arch1_dropout.txt',confusionMatrix)








