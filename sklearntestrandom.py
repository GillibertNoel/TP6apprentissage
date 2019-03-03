# coding: utf-8
from bayes import GaussianBayes
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import math
import scipy
import utils
import random

def confusionmatrixcreator(predictedlabels, truelabels):
    confusionmatrix=np.zeros([len(np.unique(truelabels)),len(np.unique(truelabels))],float)
    for i in range(0,len(predictedlabels)):
        confusionmatrix[truelabels[i]][predictedlabels[i]]=confusionmatrix[truelabels[i]][predictedlabels[i]]+100/np.sum(truelabels == truelabels[i])
    return confusionmatrix

#Separation de donn�es
total_data, total_labels = utils.load_dataset("data2.csv")
print(len(total_labels))

proportionseparation = 20#pourcentage des données a mettre dans le test.
labelstest=total_labels[0:proportionseparation]
valeurstest=total_data[0:proportionseparation]
valeurstrain=total_data[proportionseparation:100]
labelstrain=total_labels[proportionseparation:100]

for i in range(1,round(len(total_labels)/100)):
    labelteststart=100*i
    labeltrainend=100*(i+1)
    labelstemp=total_labels[labelteststart:labeltrainend]
    datatemp=total_data[labelteststart:labeltrainend]
    u=list(range(0,100))
    random.shuffle(u)
    labeltempfin=[]
    datatempfin=[]
    for i in range(0,100):
        labeltempfin.append(labelstemp[u[i]])
        datatempfin.append(datatemp[u[i]])
    print(i)
    labelstest=np.concatenate((labelstest,labeltempfin[0:proportionseparation]))
    valeurstest=np.concatenate((valeurstest,datatempfin[0:proportionseparation]))
    labelstrain=np.concatenate((labelstrain,labeltempfin[proportionseparation:100]))
    valeurstrain=np.concatenate((valeurstrain,datatempfin[proportionseparation:100]))


g = GaussianBayes(priors=None,diag=True)
# Apprentissage
g.fit(valeurstrain, labelstrain)

# Score
score = g.score(valeurstest, labelstest)
print("precision : {:.2f}".format(score))
score = g.score(valeurstrain, labelstrain)
#sur les donnees d'entrainement j'obtiens usuellement des valeurs dans les alentours de 0.73(0.77 pour le cas diagonal)
print("precision : {:.2f}".format(score))

neigh = KNeighborsClassifier(n_neighbors=3,weights='uniform', algorithm='brute')
neigh.fit(valeurstrain, labelstrain)
KNeighborsClassifier()
print(np.sum(labelstest == neigh.predict(valeurstest)) / len(labelstest))

#Confusion matrixes
print("confusion matrix for K-NN")
print(confusionmatrixcreator(neigh.predict(valeurstest),labelstest))

print("confusion matrix for the gaussian method")
print(confusionmatrixcreator(g.predict(valeurstest),labelstest))

input("Press any key to exit...")