import numpy as np
import math
import copy

class KMeans:
    """
        “How does the k-means algorithm work?” The k-means algorithm defines the centroid
         of a cluster as the mean value of the points within the cluster. It proceeds as follows. First,
         it randomly selects k of the objects in D, each of which initially represents a cluster mean
         or center. For each of the remaining objects, an object is assigned to the cluster to which
         it is the most similar, based on the Euclidean distance between the object and the cluster
         mean. The k-means algorithm then   iteratively improves the within-cluster variation.
         For each cluster, it computes the new mean using the objects assigned to the cluster in
         the previous iteration. All the objects are then reassigned using the updated means as
         the new cluster centers. The iterations continue until the assignment is stable, that is,
         the clusters formed in the current round are the same as those formed in the previous
         round.    
         Refrance: Jiawei Han, Micheline Kamber, Jian Pei Professor. 
                   Data Mining: Concepts and Techniques 3rd Edition
    """
    
    
    def __init__(self,k,max_iter):
        """
            Parameters
            ----------
            k: the number of clusters,
            max_iter: maximum number of iterations
        """           
        self.k = k
        self.max_iter = max_iter
        self.labels = np.zeros((0))
    
    def euclidean_Distance(self,p,ci):
        """
            measure euclidean distance between one object (p) and cluster center (ci)
            Parameters
            ----------
            p: one object from dataset
            ci: clusetr center             
        """
        return math.sqrt(sum(pow(a-b,2) for a, b in zip(p, ci)))
        #return np.linalg.norm(x - ci,axis=0)

    def initialize_Centroids(self,x):
        """
            arbitrarily choose k objects from x as the initial cluster centers
            Parameters
            ----------
            x: dataset            
        """

        self.centroids = np.zeros(shape = [self.k,x.shape[1]])
        
        # for choose k objects from x, generate k random number between 0 and len(x)
        indexs = np.random.randint(0,len(x),size = (self.k))
        for i in range(self.k):
            self.centroids[i] = x[indexs[i]]
            
    def update_Centroids(self,x):
        """
            computes the new center for each clusetr using the objects assigned to the cluster in
            the previous iteration. 
            Parameters
            ----------
            x: dataset                    
        """
        for i in range(self.k):
           self.centroids[i] = np.mean(x[self.labels[:] == i],axis=0)

    def clustering(self,x):
        self.labels = np.zeros([len(x)],dtype=int)
        for i in range(len(x)):
            min_Distance = float("inf")
            """
                an object is assigned to the cluster to which it is 
                the most similar, based on the Euclidean distance 
                between the object and the cluster center.
            """            
            for j in range(self.k):
                distance = self.euclidean_Distance(x[i],self.centroids[j])
                if distance < min_Distance:
                    min_Distance = distance
                    self.labels[i] = j

    def fit(self,x):
        old_Labels = np.zeros([len(x)],dtype=int)
        self.initialize_Centroids(x)
        for i in range(self.max_iter):
            self.clustering(x)
            self.update_Centroids(x)
            """
                The iterations continue until the assignment is stable, that is,
                the clusters formed in the current round are the same as those 
                formed in the previous round.
            """
            if  np.array_equal(self.labels, old_Labels):
                break
            old_Labels = copy.deepcopy(self.labels)


 
            
        
        


                
                
                
        
        
        
