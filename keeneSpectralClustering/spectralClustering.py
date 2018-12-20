#!/bin/python3.6
#Ostap Voynarovskiy
#Frequentist Machine Learning 
#Dec 17 2018
#Professor Keene

### This is an implemeentation of Spectral clustering 
### as shown described in the following paper:
### https://ai.stanford.edu/~ang/papers/nips01-spectral.pdf

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn.cluster import KMeans
from scipy.linalg import eigh as largest_eigh

class Data(object):
	def __init__(self):
		
		i=0
		#actually nothing needs to happen here

	def genSpiral(self,nPoints):
		scale = 1
		offset = 1
		sigma = .1
		numSpirals = 2

		t = np.linspace(0,numSpirals*np.pi,num = nPoints)
		noise0 = sigma*np.random.normal(size=nPoints)
		noise1 = sigma*np.random.normal(size=nPoints)
		noise2 = sigma*np.random.normal(size=nPoints)
		noise3 = sigma*np.random.normal(size=nPoints)
		
		#add normal noise
		theta0 = -t*scale + noise0
		r0 = (t + offset) + noise1
		theta1= -t*scale + np.pi + noise2	#the addition of pi does a 180 degree shift
		r1 = (t + offset) + noise3

		#convert from polar to cartesian
		self.x0 = np.cos(theta0)*(r0)
		self.y0 = np.sin(theta0)*(r0)
		cat0 = [0]*nPoints 			# the categories
		self.x1 = np.cos(theta1)*(r1) 
		self.y1 = np.sin(theta1)*(r1)
		cat1 = [1]*nPoints 			# the categories
		return np.concatenate((self.x0,self.x1)),np.concatenate((self.y0,self.y1)) #, np.concatenate((cat0,cat1)) 		
	
	def genCircles(self,nPoints):
		scale = 1
		offset = 1
		sigma = .1
		#np.random.seed(1)
		t = np.linspace(0,3.5*np.pi,num = nPoints)
		noise0 = sigma*np.random.normal(size=nPoints)
		noise1 = sigma*np.random.normal(size=nPoints)
		noise2 = sigma*np.random.normal(size=nPoints)
		noise3 = sigma*np.random.normal(size=nPoints)
		noise4 = sigma*np.random.normal(size=nPoints)
		noise5 = sigma*np.random.normal(size=nPoints)

		# Add Gaussian noise
		theta0 = -t*scale + noise0
		r0 = ( offset) + noise1
		theta1= -t*scale + np.pi + noise2	#the addition of pi does a 180 degree shift
		r1 = (2*offset) + noise3
		theta2 = -t*scale + np.pi + noise4	#the addition of pi does a 180 degree shift
		r2 = (3*offset) + noise5

		#convert from polar to cartesian
		self.x0 = np.cos(theta0)*(r0)
		self.y0 = np.sin(theta0)*(r0)
		cat0 = [0]*nPoints 			# the categories
		self.x1 = np.cos(theta1)*(r1) 
		self.y1 = np.sin(theta1)*(r1)
		cat1 = [1]*nPoints 			# the categories
		self.x2 = np.cos(theta2)*(r2) 
		self.y2 = np.sin(theta2)*(r2)
		cat2 = [2]*nPoints 			# the categories

		a = np.concatenate((self.x0,self.x1,self.x2)) 
		b = np.concatenate((self.y0,self.y1,self.y2)) 
		return a, b #, np.concatenate((cat0,cat1)) 		


	def twoDimentionalGaussian(self,nPoints):
		sig = .5
		self.x0 = np.random.normal(1,sig, size=nPoints)
		self.y0 = np.random.normal(1,sig, size=nPoints)
		
		self.x1 = np.random.normal(-1,sig, size=nPoints)
		self.y1 = np.random.normal(-1,sig, size=nPoints)
		
		return np.concatenate((self.x0,self.x1)),np.concatenate((self.y0,self.y1)) 


def spectralCluster(s,k, sigma= .12):
	
	pointSet = np.asarray(list(zip(s[0],s[1])))
	n = len(pointSet)
	#sigma tuning param sweep over this param to find the val
	
	### Step 1 ###
	# Generate a, the Affinity matrix
	a = np.zeros((len(pointSet),len(pointSet)) )
	for i in range(len(pointSet)):
		for j in range(len(pointSet)):
			a[i][j] = np.power(np.linalg.norm(np.subtract(pointSet[i],pointSet[j])),2)
	a = np.exp(np.divide(np.multiply(a,-1), 2*sigma**2 ))	
	#make diag of 0
	for i in range(len(pointSet)):
		a[i][i] = 0
	
	### Step 2 ###
	# define d a diag matrix of sum of a's rows
	d = np.zeros((len(pointSet),len(pointSet)))
	for i in range(len(pointSet)):
		d[i][i]= np.sum(a[i]) #sum of rows of a
	
	# define L (eq given in the paper)
	da = np.matmul(np.power( np.linalg.inv(d) , 0.5 ),a)
	L  = np.matmul(da, np.power(np.linalg.inv(d),0.5)) #np.linalg.inv(d)
	
	### Step 3 ####
	# find the k biggest eigenvalues and vectors
	w, x = largest_eigh(L, eigvals=(n-k,n-1)) #gives largest k eigenvals

	### step 4 ###
	#create y matrix (x normalized )
	y = np.zeros((len(pointSet),len(pointSet)))
	norms = np.linalg.norm(x,axis=1) #find norm of each row
	y = x/norms[:,None] #divide each row by its norm 
	
	### step 5 ### 
	# Perform clustering on the y matrix
	kmeans = KMeans(n_clusters=k,max_iter=3000).fit(y)
	lab = np.expand_dims(np.asarray(kmeans.labels_),1)
	
	### step 6 ###
	#assign the elements to the proper cluster 
	labeledPoints = np.append(pointSet,lab,axis=1)
	points = labeledPoints[labeledPoints[:,2].argsort()]
	split = np.bincount(points[:,2].astype(int))

	i = 0 
	k = 0
	plot = []
	for j in split: #split for plotting
		k += j
		plot.append(points[i:k])
		i = k

	return plot , y #k arrays of items belonging to their cluster and the ymatrix


data = Data()
#d = data.genSpiral(200)
d = data.genCircles(200)
#d = data.twoDimentionalGaussian(200)

#chage to 2 if youre using the spiral or gaussian
numClusters = 3
plot, y = spectralCluster(d,numClusters)

# Plotting
fig1= plt.figure(1)
for i in range(len(plot)):
	print(plot[i].T[1].shape)
	plt.scatter(plot[i].T[0], plot[i].T[1])  #, color = col[i]

plt.title("Clustered Data: Circles")
w= plt.xlabel('x')
w.set_rotation(0)
h= plt.ylabel('y')
h.set_rotation(0)
plt.axis('equal')


# Plot the Y Matrix
fig2 =plt.figure(2)
y = y.T #transpose to plot
plt.scatter(y[0],y[1],color='green')

plt.title('Y Matrix Plot: Circles')
#no axis labels cuz im not sure what theyre supposed to be
plt.axis('equal')


plt.show()