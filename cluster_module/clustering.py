from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pickle

class Cluster :
    def __init__(self, data:np.ndarray) -> None:
        """
        Module to perform helpful operation with cluster and datapoints
        """
        self.set_data(data=data)
        self.centroids = None

        self.save_cluster_copies = False

        self._cluster = None
    
    def set_data(self, data:np.ndarray):
        """Change the dataset"""
        self.data = data


    def visualize_inertia(self, max_cluster:int):
        """
        Make an elbow plot of the different inertia for each possible cluster number within the max
        cluster number provided
        """
        inertias = []
        for i in range(1,max_cluster):
            kmeans = KMeans(n_clusters=i, n_init="auto")
            kmeans.fit(self.data)
            inertias.append(kmeans.inertia_)

        plt.plot(range(1,max_cluster), inertias, marker='o')
        plt.title('Elbow method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.show()

    def clustering(self, num_cluster:int):
        kmeans = KMeans(n_clusters=num_cluster, n_init="auto")
        kmeans.fit(self.data)

        self._cluster = kmeans
        self.centroids = kmeans.cluster_centers_

        return kmeans.labels_

    def find_cluster(self, data_point:np.ndarray, use_distance:bool=True):
        """
        Return the appropriate cluster and its centroid
        """
        cluster = 0
        if not use_distance :
            cluster = self._cluster.predict(data_point.reshape(1, -1))
            return cluster, self.centroids[cluster]
        
        # if use_distance = True
        center = self.centroids[cluster]
        min_distance = self.distance(x=data_point, y=center)
        
        for clust, center in enumerate(self.centroids[1:]) :
            clust += 1 # to offsef the start at index 1
            dist = self.distance(data_point, center)

            if min_distance > dist :
                min_distance = dist
                cluster = clust
        
        return cluster, self.centroids[cluster]
                
    def find_neighbors(self, datapoint:np.ndarray, num_neighbors:int):
        """
        Return a n number of closest neighbors in ascending order of distance
        """
        cluster_distance = [(x, self.distance(datapoint, self.centroids[x])) for x in range(len(self.centroids))]
        
        sorted_cluster = sorted(cluster_distance, key=lambda x: x[1])

        return [(x[0], self.centroids[x[0]]) for x in sorted_cluster[:num_neighbors]] 

    
    def save_cluster(self, file_name:str):
        with open(f"cluster_module/saves/{file_name}.pickle", "wb") as f :
            pickle.dump(self.centroids, f)

    def load_cluster(self, file_name:str):
        with open(f"cluster_module/saves/{file_name}.pickle", "rb") as f :
            self.centroids =  pickle.load(f)

    @staticmethod
    def distance(x, y) :
        x = x.reshape(-1)
        y = y.reshape(-1)
        distance = np.sum((x - y)**2)
        return distance