import torch
import numpy as np
import matplotlib.pyplot as plt
from baseline_cnn import BasicCNN
<<<<<<< HEAD
from Arch1 import Arch1CNN
from Arch2 import Arch2CNN
import sys



def visualize_weights(weights):
    n_cols = 3
    n_rows = np.ceil(weights.shape[1]/n_cols)
    print(n_rows,n_cols)
    plt.figure(figsize = [3.2,2.4])
    weights = weights.detach().numpy()
    for i in range(weights.shape[1]):
        plt.subplot(n_rows,n_cols,i+1)
        plt.imshow(weights[1][i],cmap = "gray")
        plt.colorbar()

    #plt.suptitle("Architecture 1 Filter from the First Convolutional Layer")
    #plt.suptitle("Architecture 1 Filter from the Second Convolutional Layer")
    plt.suptitle("Architecture 1 Filter from the Third Convolutional Layer")
    plt.show()



#model= Arch1CNN()
model = BasicCNN()
#model = Arch2CNN()
model.load_state_dict(torch.load(sys.argv[1]))
visualize_weights(model.conv3.weight)

