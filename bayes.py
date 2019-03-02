import numpy as np
from typing import Union

from functools import reduce
class GaussianBayes(object):
    """ Classification by normal law by Bayesian approach
    """
    def __init__(self, priors:np.ndarray=None, diag=False) -> None:
        self.priors = priors    # a priori probabilities of classes
                                # (n_classes,)

        self.mu = None          #  mean of each feature per class
                                # (n_classes, n_features)
        self.sigma = None       # covariance of each feature per class
                                # (n_classes, n_features, n_features)
        self.diag = diag        # optimisation
        self.classes = None     #les classes

    
    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        X shape = [n_samples, n_features]
        maximum log-likelihood
        """
        n_obs = X.shape[0]
        n_classes = self.mu.shape[0]
        n_features = self.mu.shape[1]

        # initalize the output vector
        y = np.zeros(n_obs)
        #L'on parcourt tous les points a clasifier
        for i in range(0,n_obs):
            u=-float('inf') #plus petite valeur de correspondance
            #l'on parcourt les classes pour trouver la meilleure classe
            for z in range(0,n_classes):
                if self.diag:#les parties qui changent en calcul diagonal
                    mui=1/self.sigma[z]
                    detdesigma=reduce(lambda x,y: x*y,self.sigma[z])
                    half=mui*(X[i]-self.mu[z])
                else:
                    mui=np.linalg.inv(self.sigma[z])
                    half=np.dot(mui,X[i]-self.mu[z])
                    detdesigma=np.linalg.det(self.sigma[z])
                val=(-1/2)*np.dot(np.transpose(X[i]-self.mu[z]) , half)-np.log(np.sqrt(detdesigma))
                #l'on se moque de tout multiplier par un facteur constant car l'on compare des valeurs donc l'on peut ignorer les termes constants
                if self.priors!=None: #Prise en compte des à priori
                    val=np.log(self.priors[z])+val
                #print(val)
                if val>u:#la classe corresponds mieux donc l'on met a jour les paramétres de selection
                    u=val
                    y[i]=self.classes[z]
        return y
    
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """Learning of parameters
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        # number of random variables and classes
        n_features = X.shape[1]
        classes=np.unique(y)
        n_classes = len(classes)
        self.classes=classes
        # initialization of parameters
        self.mu = np.zeros((n_classes, n_features))
        if self.diag: #dans le cas diagonal je stoque que les diagonales
            self.sigma = np.zeros((n_classes, n_features))
        else:
            self.sigma = np.zeros((n_classes, n_features, n_features))
        # learning
        for n in range(0,n_classes):
            #faire la moyenne des valeurs des points d'etiquette n
            self.mu[n]=np.mean(X[(y == classes[n])])
            #faire la matrice de covariance liée a l'etiquette n
            u=np.cov(X[(y == classes[n])],rowvar=False)
            if self.diag: #le cas diagonal
                self.sigma[n]=np.diag(u)
                print("diagonal")
                print(self.sigma[n])
            else:
                self.sigma[n]=u
                    
        
        print("mu")
        print(self.mu)
        print("sigma")
        print(self.sigma)
            
    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Compute the precision
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        return np.sum(y == self.predict(X)) / len(X)
