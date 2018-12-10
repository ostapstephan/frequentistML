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

#from sklearn.preprocessing import LabelEncoder

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


Imputable = {'DataEntryAnalyst', 'CN_Priority', 'CEO_Salary'}
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

#everything w nan
containsNan = {'RatingInterval', 'CN_Priority', 'previousratingid', 'DataEntryAnalyst',
'Rpt_Comp_Emp', 'Reader2_Date', 'Rpt_Ver_Date', 'Incomplete',
'StatusID', 'DonorAdvisoryDate', 'DonorAdvisoryText', 'IRSControlID',
'ResultsID', 'DonorAdvisoryCategoryID', 'CauseID', 'CEO_Salary',
'Direct_Support', 'Indirect_Support', 'Int_Expense', 'Depreciation',
'Assets_45', 'Assets_46', 'Assets_47c', 'Assets_48c', 'Assets_49',
'Assets_54', 'Liability_60'}

#data with more than 5000 nan and also no correlation AKA DIRTY
Drop = ['Rpt_Comp_Emp', 'Reader2_Date', 'Incomplete', 'StatusID', 
'DonorAdvisoryDate', 'DonorAdvisoryText', 'ResultsID', 'DonorAdvisoryCategoryID',
'Direct_Support', 'Indirect_Support', 'Int_Expense', 'Depreciation', 
'Assets_45', 'Assets_46', 'Assets_47c', 'Assets_48c', 'Assets_49', 
'Assets_54', 'Liability_60', 'previousratingid','Rpt_Ver_Date','CauseID','RatingInterval','IRSControlID']


#clean data which is garbage
garbageData =  ['ids', 'RatingID', 'erkey', 'AccountabilityID'] #[]
Drop = Drop + garbageData

def FillNaVars(data, dr):
	#First drop everything that we wanna drop
	data = data.drop(dr,1)

	#then impute the data with proper values
	data.CN_Priority = data.CN_Priority.fillna(value=0) #bool mostly 0 
	data.DataEntryAnalyst= data.DataEntryAnalyst.fillna(value=70) #categorical perhaps we should 1 hot
	data.CEO_Salary = data.CEO_Salary.fillna(value=125546.0) #idk actually? Replace with a model that is trained to predict salary
	
	return data


trainDataClean = FillNaVars(trainData,Drop)
testDataClean  = FillNaVars(testData ,Drop)
trainDataClean['Labels'] = trainLabels.OverallScore

trainDataClean = trainDataClean.dropna(0) 

print(testDataClean.shape)
print(trainDataClean.shape)


#convert data to categorical

#le = LabelEncoder()
#big_X_imputed['Sex'] = le.fit_transform(big_X_imputed['Sex'])




#AddLabels and shuffle

print('dROPPPED:',)

trainDataClean.dropna(0)
# Prepare the inputs for the model
train_X = trainDataClean.drop('Labels',axis=1).values
test_X  = testDataClean.values
train_y = trainDataClean.Labels.values 

X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.1)
print(type(X_train))
# You can experiment with many other options here, using the same .fit() and .predict()
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=10, learning_rate=0.05).fit(X_train, y_train)# ,silent=False
predictions = gbm.predict(X_val)
print(type(predictions))
err = np.sqrt(np.mean(np.square(np.subtract( y_val, predictions))))
print(err)




# Kaggle needs the submission to have a certain format
'''
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'], 'Survived': predictions })
submission.to_csv("submission.csv", index=False)
'''

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


#################################################################################
#################################################################################
#################################################################################
#################################################################################

'''
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the data
train_df = pd.read_csv('../input/train.csv', header=0)
test_df = pd.read_csv('../input/test.csv', header=0)

feature_columns_to_use = ['Pclass','Sex','Age','Fare','Parch']

# Join the features from train and test together before imputing missing values,
# in case their distribution is slightly different
big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])

# Fare column is just one value empty so we will fill it with median value
big_X['Fare'] = big_X['Fare'].fillna(big_X['Fare'].median())

# Creating probabilty distribution for ages from the existing column values
ages_probabilities = big_X['Age'].value_counts().to_frame()
ages_probabilities['index1'] = ages_probabilities.index
ages_probabilities = ages_probabilities.rename(columns={'Age': 'Count', 'index1': 'Age'})
ages_probabilities = ages_probabilities.reindex_axis(['Age','Count'], axis=1)
ages_probabilities = ages_probabilities.reset_index()
ages_probabilities = ages_probabilities.drop(["index"],axis=1)
ages_probabilities['Probability'] = ages_probabilities['Count'] / big_X['Age'].value_counts().sum()

input_ages_list = ages_probabilities['Age'].values.tolist()
props_ages_list = ages_probabilities['Probability'].values.tolist()
newAges = np.random.choice(input_ages_list, big_X['Age'].isnull().sum(), props_ages_list)

# fill Ages null values with this distribution
AgeNulls = big_X[pd.isnull(big_X['Age'])]
for i, ni in enumerate(AgeNulls.index[:len(newAges)]):
    big_X['Age'].loc[ni] = newAges[i]

big_X_imputed = big_X

# XGBoost doesn't (yet) handle categorical features automatically, so we need to change
# them to columns of integer values.
# See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more
# details and options
le = LabelEncoder()
big_X_imputed['Sex'] = le.fit_transform(big_X_imputed['Sex'])

# Prepare the inputs for the model
train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
train_y = train_df['Survived']

# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
predictions = gbm.predict(test_X)

# Kaggle needs the submission to have a certain format;
# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
# for an example of what it's supposed to look like.
submission = pd.DataFrame({ 'PassengerId': test_df['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)
'''