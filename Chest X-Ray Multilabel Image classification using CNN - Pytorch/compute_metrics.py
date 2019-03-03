import numpy as np
import matplotlib.pyplot as plt
import sys

cmFile = sys.argv[1]
cm = np.loadtxt(cmFile).astype(np.int32)

#Compute accuracies
accuracy = np.zeros(15)
precision = np.zeros(15)
recall = np.zeros(15)

for i in range(15):
    mask = np.ones(cm.shape,bool)
    mask[i,:] = False
    mask[:,i] = False
    mask[i,i] = True

    accuracy[i] = np.sum(cm[mask])/np.sum(cm)

    if np.sum(cm[i,:]) != 0:
        precision[i] = cm[i,i]/(np.sum(cm[i,:]))

    recall[i] = cm[i,i]/np.sum(cm[:,i])

#Write out a LaTeX worthy confusion matrix
cmPercent =  cm/np.sum(cm,axis = 0) * 100

cmTable = open("cmTable_ensemble.txt","w+")

cmTable.write("\\begin{table}\n")
cmTable.write("\centering\n")

cmTable.write("\\begin{tabular}{c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}\n")
cmTable.write("&0&1&2&3&4&5&6&7&8&9&10&11&12&13&14\\\\ \n")
cmTable.write("\\hline\n")
for i in range(len(cmPercent)):
    cmTable.write(str(i)+"&")
    for j in range(len(cmPercent[i])):
        cmTable.write("%.1f" %(cmPercent[i,j]))
        if j < len(cmPercent[i]) -1 :
            cmTable.write("&")
    cmTable.write("\\\\\n")
    cmTable.write("\\hline\n")
cmTable.write("\\end{tabular}\n\\end{table}")

weights = np.array([0.42629763, 0.08048455, 0.02015054, 0.09204955, 0.14309236,0.03908578, 0.04571115, 0.00956563, 0.0381449 , 0.03340129,0.01748471, 0.01913125, 0.01148659, 0.02234593, 0.00156814])

print("Total precision=",np.sum(weights * precision))
print("Total recall=",np.sum(weights * recall))
print("Total BCR=",np.sum(weights * 0.5 * (precision + recall)))
#plt.imshow(cmPercent, cmap = "gray")
#plt.colorbar()
#plt.show()
