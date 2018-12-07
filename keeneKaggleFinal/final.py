# Ostap Voynarovskiy
# Prof Keene 
# Final Kaggle Project
# Nov 10 2018

'''
To get an A on this project, I'm going to request you do the following on this dataset.

Linear regression with L1 penalties
Linear regression with L2 penalties
Random Forest
XGBoost tree or something similar 
Try a dimensionality reduction method on all, or a subset of your data 	
Try a feature selection method.
Try one regression method that we *have not* covered in class.
Put these in Kaggle Kernels(can be one big one if you want) but don't make them public until after the competition is over.
'''

import sklearn
import pandas as pd
import numpy as np
#import tensorflow as tf 
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn import linear_model


trainData = pd.read_csv("trainFeatures.csv")
testData = pd.read_csv("testFeatures.csv")

#pull all the columns with nans
NoNAN = trainData.dropna(axis='columns')
WithNAN = trainData.drop(NoNAN.columns,1)

Dropable = [] #init array to store the columns we wanna drop
for col in WithNAN.columns: #loop all cols with NANS
	if  WithNAN[col].value_counts().values.sum()<5000:
		# if more than 50% nans Then we append to drop list
		Dropable.append(str(col))


#dataframe wo too many nans but before we impute it
preImpute = WithNAN.drop(Dropable,1) 
# CONCATENATE WITH NoNAN IN FUTURE ^


for col in preImpute.columns:
	
	print(len(preImpute[col].value_counts(1)))
	print(preImpute[col].name)
	if len(preImpute[col].value_counts(1)) >1000: 
		# if there are more than 1000 overlapping entries its likely the data is continuous and we use mean
		
	else:
		# else categorical and we will use the mode
		
		preImpute.fillna()


	i = input()