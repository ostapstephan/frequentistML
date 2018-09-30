import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans

'''
This Project analyzes the bars in NYC 

I found a dataset which gave me the geographic locations of all of the bars in NYC. 
My ultimate goal was to find the best place to get an apartment if I wanted to be closest to the most amount of bars. 

My initial approach was to use the Mean Shift Clustering Algorithm. However the results weren’t optimal as they only provided me with 3 points. Two which were too far and one that was in the east river. I had to change my approach.

The method that I ended up using was called K Means Clustering. This method allows me to specify how many clusters I would like then it iterates until it converges or runs the max allowable iterations. 

The clusters were then plotted by color and their centers shown in black.

In the future I hope to expand this to all bars in america if not the world. Initially i thought that i had found the data for the world that was 1.csv and 2.csv. But this was a very incomplete data set.
'''
# Data Taken From:
# https://www.kaggle.com/somesnm/heatmap-of-pubs-and-bars-of-new-york-city/data
# https://www.kaggle.com/datafiniti/breweries-brew-pubs-in-the-usa#8260_1.csv

# 1.csv and 2.csv were not complete data sets. so I decided to not use it
#data  = pd.read_csv("1.csv") 
#data2 = pd.read_csv("2.csv")
dataBars = pd.read_csv("bar.csv") #read in the csv into a pandas array

#data.columns=['address','categories','city','country','key','lat','longi','name','phones','postalCode','province','websites']
#data2.columns = ['identity','address','categories','city','country','hours','keys','lat','longi','menus','name','postalCode','province','twitter','websites']
dataBars.columns = ['Location Type','Incident Zip','City','Borough','lati','lon','num_calls'] #name the collumns


#only select the data with values nothing with NaN for position again csv 1 and 2 were ignored

#data = data[np.isfinite(data.longi)]
#data = data[np.isfinite(data.lat)]
#data2 = data2[np.isfinite(data2.longi)]
#data2 = data2[np.isfinite(data2.lat)]

dataBars = dataBars[np.isfinite(dataBars.lon)]
dataBars = dataBars[np.isfinite(dataBars.lati)]
fig1= plt.figure(1)

#plt.scatter(data.longi,data.lat)
#plt.scatter(data2.longi,data2.lat)
#plt.scatter(dataBars.lon,dataBars.lati)


dataComb = [[x, y] for x, y in zip(dataBars.lon, dataBars.lati)]
'''
ms = MeanShift()
ms.fit(dataComb)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
print([cluster_centers])
n_clusters_ = len(np.unique(labels))
print("Number of est clusters via Mean Shift Clustering:",n_clusters_)
[x,y] = list(zip(*cluster_centers))
plt.scatter(x,y,color = 'red')
'''
#number of clusters
clusNum = 30 
kmeans = KMeans(n_clusters = clusNum, random_state = 5, n_init = 10)

print(dataComb[0])
kmeans.fit(dataComb)

print(kmeans.labels_)
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# a and b are for the cluster centers
[a,b] = list(zip(*kmeans.cluster_centers_)) 

# x and y are for individual bars
[x, y] = list(zip( * dataComb))

# generate random rgb for each cluster
colorArr = np.random.rand(clusNum,3)

# Plot the US axes this was for csv 1 and 2
#plt.ylim(21,50)
#plt.xlim(-130,-70)

plt.title('Clustering of Bars NYC')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.scatter(x,y,c=colorArr[kmeans.labels_])#blot all bars in a cluster with the same color
plt.scatter(a,b,color = 'Black')

plt.show()

