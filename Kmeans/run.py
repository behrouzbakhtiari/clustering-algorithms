import KMeans
import numpy as np
from matplotlib import pyplot as plt
def plot(x,labels,centroids):
    colors = np.array(['b', 'g', 'r', 'c', 'm', 'y'])
    fig, ax = plt.subplots()
    ax.scatter(x[:,0],x[:,1],color=colors[labels])
    ax.scatter(centroids[:,0],centroids[:,1],color='k', marker='*',s = 200)
    plt.show()
        
#load data
data = np.genfromtxt('iris.data',delimiter=',')
data = data[1:,0:4]

k = 3 #the number of clusters
max_iter = 100 #maximum number of iterations
c = KMeans.KMeans(k,max_iter)
c.fit(data)

print(c.labels)
print(c.centroids)
plot(data,c.labels,c.centroids)