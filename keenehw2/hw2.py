import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

#from tqdm import tqdm
#from sklearn.cluster import MeanShift
#from sklearn.cluster import KMeans

def mse(real,predict):
	e = predict.flatten() - real.flatten()
	e = float(np.sum(np.power(e,2))/len(real))
	return e

def genTrainAndVal(d): #split the features and labels of the training data 80:10:10 train, validation and test
	z = d.shape[0]	
	nv = int( z *.1) 	# num validation and test samp 
	#print (d[:nv].shape, d[nv:nv*2].shape, d[nv*2:].shape )
	return d[:nv], d[nv:nv*2], d[nv*2:]

###########################################
### 	Plain Old Linear Regression 	###
###########################################
dataBars = pd.read_csv("auto-mpg.csv") #read in the csv into a pandas array
dataBars.columns = ['MPG','Cylinders','Displacement','Horsepower','weight','Acceleration','Model Year', 'origin','name'] #name the collumns
dataBars = dataBars.drop("name",axis=1) #Drop the manufacturer name cuz its useless

#add bias term
dataBars['bias']=1

#shuffle data first 
dataBars = shuffle(dataBars)

#split data apart into train val and test
test,val,train = genTrainAndVal(dataBars)

# Pull Data apart into features and labels
testFeat  = np.array(test.drop("MPG",axis=1))
testLab   = np.array(test.MPG)
trainFeat = np.array(train.drop("MPG",axis=1))
trainLab  = np.array(train.MPG)
valFeat   = np.array(val.drop("MPG",axis=1))
valLab    = np.array(val.MPG)

# Perform Linear Regression 
inve = np.linalg.inv(np.matmul(np.transpose(trainFeat),trainFeat)) #the inverse part
trflab = np.transpose(trainFeat) #transpose features times labels
beta_hat = np.matmul(np.matmul(inve,trflab),trainLab)

# Generate predictions 
pred = np.matmul(testFeat, beta_hat)

#print(pred)
# Print the Mean squared error
print("Test predictions: ", mse(testLab,pred))




###########################################
### 		  Ridge Regression 			###
###########################################

dataBarsRR = dataBars.drop('bias',axis=1) #Drop the bias cuz its gonna be normalized 
dataBarsRR = dataBarsRR.drop('MPG',axis=1)
dataBarsLabRR = dataBars.MPG
#print("A",dataBarsRR.columns)
# Normalize Data
dataBarsRRMean = dataBarsRR.mean(axis=0)
dataBarsRR = (dataBarsRR-dataBarsRRMean)
dataBarsRRMax = dataBarsRR.max(axis=0) 
dataBarsRR = dataBarsRR/dataBarsRRMax
#print("Shapes: ",dataBarsRRMean.shape,dataBarsRRMax.shape)

bias = dataBars.MPG.mean(axis=0)
print("B",bias)
#split data apart into train val and test for feat and lab
testrF,valrF,trainrF = genTrainAndVal(dataBarsRR)
testrL,valrL,trainrL = genTrainAndVal(dataBarsLabRR)
# convert to np arrays
testFeat, valFeat, trainFeat = np.array(testrF),np.array(valrF),np.array(trainrF)
testLab, valLab, trainLab = np.array(testrL),np.array(valrL),np.array(trainrL)

n = 100000 #steps for the lambda optimization
Lambda = np.linspace(0,1000-1/n,n)
Identity = np.identity(trainFeat.shape[1])

lf =np.inf
best = np.inf
MSEa =[] 

for l in Lambda:
	inve = np.linalg.inv(np.matmul(np.transpose(trainFeat),trainFeat) +l*Identity)
	trflab = np.transpose(trainFeat) #transpose features times labels
	beta_hatR = np.matmul(np.matmul(inve,trflab),trainLab)+bias
	pred = np.matmul(valFeat, beta_hatR)
	r= mse(valLab,pred)
	MSEa.append([l,r])
	if (r < best):
		best = r
		lf = l 
	

fig1= plt.figure(1)

MSEa=np.asarray(MSEa)
MSEa=np.transpose(MSEa)
print("msea,",MSEa.shape)
plt.plot(MSEa[0],MSEa[1])

print("lf: ", lf)
plt.show()
inve = np.linalg.inv(np.matmul(np.transpose(trainFeat),trainFeat) +lf*Identity)
trflab = np.transpose(trainFeat) #transpose features times labels
beta_hatR = np.matmul(np.matmul(inve,trflab),trainLab)
pred = np.matmul(testFeat, beta_hatR)
print("Test predictions: ", mse(testLab,pred))



###########################################
### 		  Lasso Regression 			###
###########################################




