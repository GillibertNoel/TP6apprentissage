# coding: utf-8
from bayes import GaussianBayes
import sklearn
import numpy as np
import math
import scipy
import utils

#Separation de donn�es
total_data, total_labels = utils.load_dataset("data12.csv")
print(len(total_labels))

proportionseparation = 20#pourcentage des données a mettre dans le test.
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
score = g.score(valeurstest, labelstest)#usuellement cela vaut 100 pourcent
print("precision : {:.2f}".format(score))
score = g.score(valeurstrain, labelstrain)
#sur les donnees d'entrainement j'obtiens usuellement des valeurs dans les alentours de 0.73(0.77 pour le cas diagonal)
print("precision : {:.2f}".format(score))
input("Press any key to exit...")