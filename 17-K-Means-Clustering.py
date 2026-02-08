import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

data = make_blobs(n_samples= 200, n_features= 2, centers=4, cluster_std=1.8, random_state=101)
data[1]

plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='rainbow')
plt.show()

#Creating the Cluster

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)

kmeans.fit(data[0])

kmeans.cluster_centers_
kmeans.labels_

fig, (ax1,ax2) = plt.subplots(1, 2, sharey = True, figsize = (10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1], c = kmeans.labels_, cmap = 'rainbow')
ax2.set_title('Original')
ax2.scatter(data[0][:,0], data[0][:,1], c = data[1], cmap = 'rainbow')
plt.show()


##EXERCISE

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

college = pd.read_csv('C:\\Users\\Daniel.Gallo\\OneDrive - Xodus Group\\Documents\\Data Science\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\17-K-Means-Clustering\\College_Data.csv', index_col=0)
college.head()

college.info()
college.describe()

#EDA

sns.lmplot(college, x = 'Room.Board', y = 'Grad.Rate', hue = 'Private', fit_reg= False)
plt.show()

sns.scatterplot(college, x = 'Outstate', y = 'F.Undergrad', hue = 'Private')
plt.show()

g = sns.FacetGrid(college, hue = 'Private')
g = g.map(plt.hist, 'Outstate', bins = 20, alpha = 0.7)
plt.show()
