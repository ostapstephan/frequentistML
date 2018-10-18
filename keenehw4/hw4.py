# Ostap Voynarovskiy
# Prof. Keene
# Frequentist ML 
# Oct 17 2018
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.utils import shuffle
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import KFold



features = np.random.uniform(size = (50,5000))
labels = np.random.randint(0,2,50)
######################################################################################
############ 	Feature selection and Cross Validation The Wrong way 		##########
######################################################################################

#check features and labels shapes
#print(features[:,1].flatten,labels,len(features[0]))

#generate corr the wrong way
corr = [np.corrcoef(features[:,feat],labels)[0,1] for feat in range(len(features[0]))]
#make it a np array
corr = np.asarray(corr)

#pick indecies of top 100 max correlations
max100 =corr.argsort()[-100:][::1]# features[:,corr.argsort()[-100:][::1]]

# Pull data
selectedFeat = features[:,max100]
selectedLab = copy.copy(labels)

# Genertate Model
model = KNN(n_neighbors = 1)

kf = KFold(n_splits = 5,shuffle=True)
#kf.get_n_splits(max100)


error=[]
#loop through all 50 examples of the 80:20 folding we are doing. 
for t in range(10):
	for trainIndex, testIndex in kf.split(selectedFeat):
		#print("Test: ", testIndex)
		trainFeat, testFeat = selectedFeat[trainIndex], selectedFeat[testIndex]
		trainLab,  testLab =  selectedLab[trainIndex],  selectedLab[testIndex]
		model.fit(trainFeat, trainLab)
		error.append(1-model.score(testFeat,testLab))
print("Test Error the Wrong Way: ",np.mean(error))
#print(np.mean(error))



######################################################################################
############ 	Feature selection and Cross Validation The Right way 		##########
######################################################################################

modelRight = KNN(n_neighbors = 1)
kfRight = KFold(n_splits = 5)
#print(kfRight)
e = []
for t in range(10):
	kfRight = KFold(n_splits = 5,shuffle=True)	
	for trInd, testInd in kf.split(features):
		#training features and labels
		f = features[trInd]
		l = labels[trInd]
		ft = features[testInd]
		lt = labels[testInd]
		#compute Corr of vars with training Data
		corr = np.asarray([np.corrcoef(f[:,feat],l)[0,1] for feat in range(len(f[0]))])
		#Pick 100 most correlated 
		max100 = corr.argsort()[-100:][::1]
		model.fit(f[:,max100], l)
		e.append(1-model.score(ft[:,max100],lt))

#print(e)
print("Test Error The correct Way: ", np.mean(e))




