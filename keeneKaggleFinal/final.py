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
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
import pylab
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def FillNaVars(data, dr):
	#First drop everything that we wanna drop
	data = data.drop(dr,1)

	#then impute the data with proper values
	#lol i forgot that the xgboost handles sparse data
	data.CN_Priority = data.CN_Priority.fillna(value=0)#0 #bool mostly 0 
	#data.DataEntryAnalyst= data.DataEntryAnalyst.fillna(value=np.nan) #70 #categorical perhaps we should 1 hot
	data.CEO_Salary = data.CEO_Salary.fillna(value=125546.0) #125546.0#idk actually? Replace with a model that is trained to predict salary
	
	return data

trainData = pd.read_csv("trainFeatures.csv")
trainLabels = pd.read_csv("trainLabels.csv")
testData = pd.read_csv("testFeatures.csv")

#pull all the columns with nans
NoNAN = trainData.dropna(axis='columns')
WithNAN = trainData.drop(NoNAN.columns,1)

Dropable = [] #init array to store the columns we wanna drop
for col in WithNAN.columns: #loop all cols with NANS
	if  WithNAN[col].value_counts().values.sum() < 5000:
		Dropable.append(str(col))  #if more than 50% nans Then we append to drop list

#dataframe wo too many nans but before we impute it
preImpute = WithNAN.drop(Dropable,1)
# CONCATENATE WITH NoNAN IN FUTURE ^

Imputable = { 'CN_Priority', 'CEO_Salary'}

'''
for col in Imputable:
	#print(len(preImpute[col].value_counts(1)))	
	#print("Data Name: ",preImpute[col].name)
	#print("Data type\n", preImpute[col].value_counts())
	#print("Non Nan values: ",sum(preImpute[col].value_counts()))
	#print("mean:", preImpute[col].mean())	
	#print("median:", preImpute[col].median())
	#print("mode:", preImpute[col].mode())

	if len(preImpute[col].value_counts(1)) >1000: 
		# if there are more than 1000 entries its likely the data is continuous and we use mean
		pass
	else:
		# else categorical and we will use the mode
		pass
		mode = preImpute[col].mode()[0]
		#preImpute.fillna()

	#i = input()
'''

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
imbalance = ['BaseYear','CNVersion','DataEntryAnalyst','EmployeeID','RatingYear','Rpt_Ap_Emp','Tax_Year']

Drop = Drop + garbageData + imbalance
########### END OF ALL THE COLLUMNS WE WANT TO DROP ###############


trainDataClean = FillNaVars(trainData,Drop)
testDataClean  = FillNaVars(testData ,Drop)


trainDataClean['Labels'] = trainLabels.OverallScore
trainDataClean = trainDataClean.dropna(0) #2126x39
testDataClean = testDataClean.dropna(0) #9999x40 conains labels


print(testDataClean.shape)
print(trainDataClean.shape)


#convert data to categorical
#This was found by looking at the clean data and seeing if whatever category was categorical
	
categoricalData = ['AuditedFinancial_status','BoardList_Status',
'CN_Priority','exclude', 'ExcludeFromLists','Form990_status',
'Privacy_Status','RatingTableID', 'StaffList_Status']


# Prepare the inputs for the model
train_X = trainDataClean.drop('Labels',axis=1).values
test_X  = testDataClean.values
train_y = trainDataClean.Labels.values 

X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.01)

print (len(categoricalData))

for n in categoricalData:
	#a = trainDataClean[n]
	#le = LabelEncoder()
	#z  = le.fit(trainDataClean[n])
	#z2 = le.fit(testDataClean[n])
	#print(z.classes_==z2.classes_)
	
	z = [trainDataClean[n],testDataClean[n]]
	z = pd.concat(z)
	z = pd.get_dummies(z,prefix=n)
	
	i = len(trainDataClean[n])

	trainDataClean = pd.concat((trainDataClean,z[:i]),1)
	testDataClean  = pd.concat([testDataClean, z[i:]],1)
	
	trainDataClean = trainDataClean.drop(n,1)
	testDataClean  = testDataClean.drop(n,1)
	
	print('TD after shape:',trainDataClean.shape,testDataClean.shape)

	# create plot
	'''
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
	'''

#####################################
# 		Training begins here		#
#####################################

print((X_train.shape))
print('Press enter to start')
_ = input()

'''
le = LabelEncoder()
big_X_imputed['Sex'] = le.fit_transform(big_X_imputed['Sex'])
'''


# Load data into xgboost structure
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(test_X, label=y_val)

#setup parameters
param = {'max_depth': 10,'nthread':4 , 'eta': .02, 'silent': 0, 'objective': 'reg:linear', 'eval_metric':'rmse'}#'tree_method':'gpu_hist',
evallist = [(dval, 'eval'), (dtrain, 'train')]

'''
lambda [default=1] L2 regularization term on weights. Increasing this value will make model more conservative.
alpha [default=0]  L1 regularization term on weights. Increasing this value will make model more conservative.
'''

#training
num_round = 1000
bst = xgb.train(param, dtrain, num_round, evallist)

# Prediction
ypred = bst.predict(dtest)
print('press enter to save output to csv')
i = input()


#Outputing predictions to csv

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


# plotting
'''
b = trainLabels.OverallScore
for n in trainDataClean.columns:
	a = trainDataClean[n]

	plt.title(str(a.name)+' vs '+str(b.name))
	plt.xlabel(a.name)
	plt.ylabel(b.name)

	plt.scatter(a,b,color = 'Black')
	pylab.savefig( 'graphs/clean/'+str(a.name)+'.png')
	plt.cla()
'''


