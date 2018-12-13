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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab

import sklearn
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras import regularizers
from keras import backend as K


BATCH_SIZE = 100
epochs = 100 	#we: achieve overfitting after like 15-20 epochs
DROP_RATE =.1
weight_decay = 1e-4

def FillNaVars(data, dr):
	#First drop everything that we wanna drop
	data = data.drop(dr,1)
	#Imputable = { 'CN_Priority', 'CEO_Salary'}
	#then impute the data with proper values
	
	#lol i forgot that the xgboost handles sparse data
	data.CN_Priority = data.CN_Priority.fillna(value=0)#0 #bool mostly 0 
	data.CEO_Salary = data.CEO_Salary.fillna(value=125546.0) #125546.0#idk actually? Replace with a model that is trained to predict salary
	#data.DataEntryAnalyst= data.DataEntryAnalyst.fillna(value=np.nan) #70 #categorical perhaps we should 1 hot	
	return data

trainData = pd.read_csv("trainFeatures.csv")
trainLabels = pd.read_csv("trainLabels.csv")
testData = pd.read_csv("testFeatures.csv")

#pull all the columns with nans
NoNAN = trainData.dropna(axis='columns')
WithNAN = trainData.drop(NoNAN.columns,1)

Dropable = [] 						#init array to store the columns we wanna drop
for col in WithNAN.columns: 		#loop all cols with NANS
	if  WithNAN[col].value_counts().values.sum() < 5000:
		Dropable.append(str(col))  	#if more than 50% nans Then we append to drop list


################## ALL THE COLLUMNS WE WANT TO DROP ###############
# Data with more than 5000 nan

Drop = ['Rpt_Comp_Emp', 'Reader2_Date', 'Incomplete', 'StatusID', 
'DonorAdvisoryDate', 'DonorAdvisoryText', 'ResultsID', 'DonorAdvisoryCategoryID',
'Direct_Support', 'Indirect_Support', 'Int_Expense', 'Depreciation', 
'Assets_45', 'Assets_46', 'Assets_47c', 'Assets_48c', 'Assets_49', 
'Assets_54', 'Liability_60']

#clean data which is garbage
garbageData =  ['ids', 'RatingID', 'erkey', 'AccountabilityID',  'previousratingid',
				'Rpt_Ver_Date','CauseID','RatingInterval','IRSControlID'] 

#test and train are imbalanced
imbalance = ['BaseYear','CNVersion','DataEntryAnalyst','EmployeeID','RatingYear','Tax_Year','Rpt_Ap_Emp']

Drop = Drop + garbageData + imbalance

########### END OF ALL THE COLLUMNS WE WANT TO DROP ###############

# Impute Data
trainDataClean = FillNaVars(trainData,Drop)
testDataClean  = FillNaVars(testData ,Drop) 

#drop nulls in labeled data 
trainDataClean['Labels'] = trainLabels.OverallScore 
q = pd.isnull(trainDataClean).any(1).nonzero()[0] #print('q',q)
trainDataClean = trainDataClean.drop(q,0)	#9999x33 conains labels 
testDataClean  = testDataClean.dropna(0) 	#2126x32


#### One Hot encode the Categorical Data ####

# This was found by looking at the clean data and seeing if whatever category was categorical	
categoricalData = ['AuditedFinancial_status','BoardList_Status','CN_Priority','exclude',
'ExcludeFromLists','Form990_status','Privacy_Status','RatingTableID', 'StaffList_Status']

for n in categoricalData: # Process the categorical data into one Hot vectors
	#Stack the data first
	z = [trainDataClean[n],testDataClean[n]] 
	# Concat stacked data
	z = pd.concat(z)
	# one hot encode
	z = pd.get_dummies(z,prefix=n)
	
	# remember slice size
	i = len(trainDataClean[n])

	# Add the one-hot vectors and slice the data into the proper train and test size
	trainDataClean = pd.concat([trainDataClean,z[:i]],1)
	testDataClean  = pd.concat([testDataClean, z[i:]],1)
	
	# drop the Original collumn labels cuz we now have the one-hot labels
	trainDataClean = trainDataClean.drop(n,1)
	testDataClean  = testDataClean.drop(n,1)
	
	# sanity check the sizes
	print('TD after shape:',trainDataClean.shape,testDataClean.shape)

	# toggle plotting to check if categorical data is evenly spread between train and test data
	if False:
		fig, ax = plt.subplots()
		index = np.arange(len(z.columns))
		bar_width = 0.35
		opacity = 0.8
		rects1 = plt.bar(index, z1, bar_width,color='b',label='train')
		rects2 = plt.bar(index + bar_width, z2, bar_width,color='g',label='test')
		plt.xlabel('Category')
		plt.ylabel('Mean')
		plt.title(n)
		plt.xticks(index + bar_width/2, (z.columns))
		plt.legend()
		pylab.savefig( 'graphs/testTrainDiff/'+str(n)+'.png')
		#plt.show()
		plt.cla()



def normalizeMyData(trainDataClean,testDataClean):
	#stack data first
	lab = pd.DataFrame(trainDataClean.Labels.values, columns=['Labels'])

	print("Label shape: ",lab.shape)
	z = [trainDataClean.drop('Labels',axis=1),testDataClean]
	z = pd.concat(z)
	print("z shape",z.shape)
	i = len(trainDataClean['Labels'])
	
	#remember cols for post normalization
	col = z.columns
	print('col shape',col.shape)
	x_stack = z.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x_stack)
	z = pd.DataFrame(x_scaled, columns=col)
	print("z shape",z.shape)
	#print("debug",z[5445:5450],'\n',lab[5445:5450])
	trainDataClean = pd.concat((z[:i],lab),1)
	print("trainDataClean shape",trainDataClean.shape)
	print('TD Shape Post norm:',trainDataClean.shape,testDataClean.shape)
	return trainDataClean,testDataClean

if True: #normalize Data
	trainDataClean,testDataClean = normalizeMyData(trainDataClean,testDataClean)


# Prepare the inputs for the model
train_X = trainDataClean.drop('Labels',axis=1).values
test_X  = testDataClean.values
train_y = trainDataClean.Labels.values

X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.2)
#print (X_train.shape, X_val.shape, y_train.shape, y_val.shape) #debug


#####################################
# 		   Models begin here		#
#####################################

def Ridge():
	def rmse(y_pred, y_true):
		return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

	print('Press enter to train a linear_model w Ridge')
	_ = input()

	regr = linear_model.Ridge(alpha = 1.2)
	regr.fit(X_train,y_train)
	print(X_val.shape)

	p = regr.predict(X_val)
	print ("rmse ",rmse(p, y_val))

	testPred = regr.predict(test_X)
	#results can be passed along and saved 

#Ridge()

def Lasso():
	def rmse(y_pred, y_true):
		return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

	
	print('Press enter to train a linear_model w Lasso')
	_ = input()

	regr = linear_model.Lasso(alpha = .2)
	regr.fit(X_train,y_train)
	
	p = regr.predict(X_val)
	print ("rmse ",rmse(p, y_val))

	testPred = regr.predict(test_X)
	#results can be passed along and saved 


#Lasso()

def myRandomForestClassifier():
	def rmse(y_pred, y_true):
		return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

	print('Press enter to train a Random Forest')
	_ = input()

	rndforest = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=8, n_jobs=4, verbose=0)
	
	print(X_train.shape,y_train.shape)
	rndforest.fit(X_train,y_train)

	p = rndforest.predict(X_val)
	print ("rmse ",rmse(p, y_val))
	
	testPred = rndforest.predict(test_X)

	#results can be passed along and save

#myRandomForestClassifier()

def RunXgboost():
	print('Press enter to start XGBoost')
	_ = input()

	# Load data into xgboost structure
	dtrain = xgb.DMatrix(X_train, label=y_train)
	dval = xgb.DMatrix(X_val, label=y_val)
	dtest = xgb.DMatrix(test_X, label=y_val)

	#print('test x shape',test_X.shape)
	#setup parameters
	param = {'max_depth': 8,'nthread':4 , 'eta': .04, 'silent': 0, 'objective': 'reg:linear', 'eval_metric':'rmse'} #'tree_method':'gpu_hist',
	evallist = [(dval, 'eval'), (dtrain, 'train')]

	#training
	num_round = 1000
	bst = xgb.train(param, dtrain, num_round, evallist)


	# Prediction
	ypred = bst.predict(dtest)
	print('press enter to save output to csv')
	i = input()


	#### Outputing predictions to csv ####

	#generate indecies
	ident =(np.arange(1,len(ypred)+1,dtype=int))
	out = np.vstack([ident,ypred ]).T

	# convert np to pandas dataframe and write it 
	out = pd.DataFrame(data= out,columns=['Id','OverallScore'])
	out.Id = out.Id.astype(int)
	out.to_csv(path_or_buf= 'output/xgboost.csv',index=False, float_format='%.3f')

	#saving a model post training
	#bst.save_model('models/1.model')

	#load a model
	#bst = xgb.Booster({'nthread': 4})  # init model
	#bst.load_model('model.bin')  # load data

#RunXgboost()



def DenseNet():
	print('Press enter to start training the Dense Net')
	_ = input()

	def root_mean_squared_error(y_true, y_pred):
		#define error metric
		return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
	
	# Convert to float 
	x_t, x_v, x_te = X_train.astype('float32'), X_val.astype('float32'),test_X.astype('float32')
	y_t, y_v 	   = y_train.astype('float32'), y_val.astype('float32')

	#print Shapes
	print("Training features shape: ", x_t.shape)
	print("Validation features shape: ",x_v.shape)
	print("val features shape: ", x_te.shape)

	

	#get input shape
	inputShapeDenseNet = (X_train.shape[1],) 
	
	model = Sequential()
	
	# layer 1
	model.add(Dense(200, input_shape=inputShapeDenseNet,activation="elu")) #first layer with a input shape of the net?
	
	model.add(BatchNormalization())
	model.add(Dropout(DROP_RATE))

	# layer 2
	model.add(Dense(150,activation="elu"))
	model.add(BatchNormalization())
	model.add(Dropout(DROP_RATE))

	# layer 3
	model.add(Dense(100,activation="elu"))
	model.add(BatchNormalization())
	model.add(Dropout(DROP_RATE))

	model.add(Dense(1))

	model.summary()

	model.compile(loss = root_mean_squared_error, optimizer=keras.optimizers.Adam(), metrics =[root_mean_squared_error])
	model.fit(x_t, y_t,	batch_size=BATCH_SIZE,
		epochs = epochs,
		verbose = 1,
		validation_data =(x_v,y_v))

	score = model.evaluate(x_v,y_v,verbose= 1)
	pred = predict(x_te)

	print("Val loss:", score[0])
	print("Val accuracy:", score[1])
	pred = predict(x_te) #predict test data

#DenseNet()




# plotting for the correlation of variables with the output
if False:
	b = trainLabels.OverallScore
	for n in trainDataClean.columns:
		a = trainDataClean[n]

		plt.title(str(a.name)+' vs '+str(b.name))
		plt.xlabel(a.name)
		plt.ylabel(b.name)

		plt.scatter(a,b,color = 'Black')
		pylab.savefig( 'graphs/clean/' + str(a.name) + '.png')
		plt.cla()
