# coding: utf-8
from bayes import GaussianBayes
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import math
import scipy
import utils
from matplotlib.pyplot import *

def confusionmatrixcreator(predictedlabels, truelabels):
    confusionmatrix=np.zeros([len(np.unique(truelabels)),len(np.unique(truelabels))],float)
    for i in range(0,len(predictedlabels)):
        confusionmatrix[truelabels[i]][predictedlabels[i]]=confusionmatrix[truelabels[i]][predictedlabels[i]]+100/np.sum(truelabels == truelabels[i])
    return confusionmatrix

#Separation de donn�es
total_data, total_labels = utils.load_dataset("data12.csv")
print(len(total_labels))
X=[]
Y=[]
for zlurb in range(1,51):
    proportionseparation = zlurb#pourcentage des données a mettre dans le test.
    labelstest=total_labels[0:proportionseparation]
    valeurstest=total_data[0:proportionseparation]
    valeurstrain=total_data[proportionseparation:100]
    labelstrain=total_labels[proportionseparation:100]
    
    for i in range(1,round(len(total_labels)/100)):
        labelteststart=100*i
        labeltestend=100*i+proportionseparation
        labeltrainend=100*(i+1)
        print(i)
        labelstest=np.concatenate((labelstest,total_labels[labelteststart:labeltestend]))
        valeurstest=np.concatenate((valeurstest,total_data[labelteststart:labeltestend]))
        labelstrain=np.concatenate((labelstrain,total_labels[labeltestend:labeltrainend]))
        valeurstrain=np.concatenate((valeurstrain,total_data[labeltestend:labeltrainend]))
    
    
    g = GaussianBayes(priors=None,diag=True)
    # Apprentissage
    g.fit(valeurstrain, labelstrain)
    
    # Score
    score = g.score(valeurstest, labelstest)
    X.append(score)
    print("precision : {:.2f}".format(score))
    
    neigh = KNeighborsClassifier(n_neighbors=3,weights='uniform', algorithm='brute')
    neigh.fit(valeurstrain, labelstrain)
    Y.append(np.sum(labelstest == neigh.predict(valeurstest)) / len(labelstest))
t = np.linspace(1, 50, 50)

plot(t, X)
plot(t, Y)
show()
#Confusion matrixes
print("confusion matrix for K-NN")
print(confusionmatrixcreator(neigh.predict(valeurstest),labelstest))

print("confusion matrix for the gaussian method")
print(confusionmatrixcreator(g.predict(valeurstest),labelstest))

input("Press any key to exit...")