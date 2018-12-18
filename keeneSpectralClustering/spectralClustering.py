#!/bin/python3.6
#Ostap Voynarovskiy
#Uns
#Sept 19 2018
#Professor Curro

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

BATCH_SIZE = 200
NUM_ITER = 4000# iterations of training 

class Data(object):
	def __init__(self):
		#create spirals
		nPoints = 200 
		self.index = np.arange(nPoints)
		self.nPoints = nPoints
		self.featx, self.featy, self.lab  = self.gen_spiral(nPoints)

	def gen_spiral(self,nPoints):
		scale = 1
		offset = 1
		sigma = .2

		t = np.linspace(0,3.5*np.pi,num = nPoints)
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
		return np.concatenate((self.x0,self.x1)),np.concatenate((self.y0,self.y1)), np.concatenate((cat0,cat1)) 		
	
	def genCircles(self,nPoints):
		return True

	def secondGaussian(self,nPoints):
		return True

	def kmeansClusterWithOOB(self):
		return True


def spectralCluster(self,pointSet,k):
	# Step 1
	sigma = 3#tuning param
	a = np.zeros(len(pointSet),len(pointSet))
	for i in range(len(pointSet)):
		for j in range(len(pointSet)):
			a[i][j] = np.linalg.norm(np.subtract(pointSet[i],pointSet[j]))

	a = np.divide(-1*np.power(a),2*sigma**2)

	# Step 2
	#define d

	for i in range(len(pointSet)):

	for 
	

	return data

fig1= plt.figure(1)

xc,yc = np.linspace(-15,15,500),np.linspace(-15,15,500) 
xv,yv = np.meshgrid(xc,yc)

feat = np.array(list(zip(xv.flatten(),yv.flatten())))

plt.contourf(xv,yv,cont.reshape((500,500)),[0,.5,1])
plt.scatter(data.x0,data.y0,color='white')
plt.scatter(data.x1,data.y1,color='black')

w= plt.xlabel('x')
h= plt.ylabel('y')
h.set_rotation(0)
w.set_rotation(0)




plt.axis('equal') #make it so that it isnt warped
plt.show()


