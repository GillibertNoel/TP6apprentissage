# coding: utf-8
from bayes import GaussianBayes
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import math
import scipy
import utils
#Separation de donn�es
total_data, total_labels = utils.load_dataset("data2.csv")
print(len(total_labels))

K = 2 #number of sections
gaussianresult=0
KNNresult=0
chunk=int(100/K)
for iter in range(0,K):
    teststart = chunk*iter
    testend = chunk*(iter+1)
    labelstest=total_labels[teststart:testend]
    valeurstest=total_data[teststart:testend]
    valeurstrain=np.concatenate((total_data[0:teststart],total_data[testend:100]))
    labelstrain=np.concatenate((total_labels[0:teststart],total_labels[testend:100]))
    
    for i in range(1,round(len(total_labels)/100)):
        labelstrainstart=100*i
        labelteststart=100*i+teststart
        labeltestend=100*i+testend
        labeltrainend=100*(i+1)
        labelstest=np.concatenate((labelstest,total_labels[labelteststart:labeltestend]))
        valeurstest=np.concatenate((valeurstest,total_data[labelteststart:labeltestend]))
        labelstrain=np.concatenate((np.concatenate((labelstrain,total_labels[labelstrainstart:labelteststart])),total_labels[labeltestend:labeltrainend]))
        valeurstrain=np.concatenate((np.concatenate((valeurstrain,total_data[labelstrainstart:labelteststart])),total_data[labeltestend:labeltrainend]))
    
    
    g = GaussianBayes(priors=None,diag=False)
    # Apprentissage
    g.fit(valeurstrain, labelstrain)
    
    # Score
    gaussianresult = gaussianresult + g.score(valeurstest, labelstest)

    neigh = KNeighborsClassifier(n_neighbors=3,weights='uniform', algorithm='brute')
    neigh.fit(valeurstrain, labelstrain)
    KNNresult = KNNresult + np.sum(labelstest == neigh.predict(valeurstest)) / len(labelstest)
gaussianresult=gaussianresult/K
KNNresult=KNNresult/K
print("gaussian average precision")
print(gaussianresult)
print("K-NN average precision")
print(KNNresult)
input("Press any key to exit...")