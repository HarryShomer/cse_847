import argparse
import numpy as np 
from scipy.stats import wasserstein_distance


class KMeans:
    """
    Implementation of standard K-Means clustering
    """
    def __init__(self, k):
        self.k = k   # Number of means
    

    def fit(self, X, stopping_criteria=10, max_iters=1000):
        """
        Fit to some data.

        Parameters:
        -----------
            X: np.array
                N x M matrix of data
            stopping_criteria: int
                Will stop when number of changes from one iteration to another is less than this
            max_iters: int
                Max number of iteration to recompute clusters until stop
        
        Returns:
        --------
        list
            Centroid of each cluster
        """
        self.clusters = []

        # 1. Randomly init clusters
        #    Do so in range of each feat dim
        for _ in range(self.k):
            c = np.random.uniform(X.min(axis=0), X.max(axis=0))
            self.clusters.append(c)

        # Init for first iter. -1 can't happen so all will change clusters
        prev_sample_to_cluster = np.array([-1 for _ in range(X.shape[0])])

        ### 2. Alternate between assigning group membership and re-computing centroid
        for _ in range(1, max_iters+1):
            sample_to_cluster = self.assign_membership(X)
            self.recompute_centroids(X, sample_to_cluster)

            clust_changes = (prev_sample_to_cluster != sample_to_cluster).sum()
            prev_sample_to_cluster = sample_to_cluster

            if clust_changes < stopping_criteria:
                break
        
        return self


    def recompute_centroids(self, X, sample_to_cluster):
        """
        Recompute the location of the centroids given the current group members
        """
        for k in range(self.k):
            clust_members = X[sample_to_cluster == k]

            # Happens when no samples are assigned to kth cluster
            if clust_members.shape[0] != 0:
                self.clusters[k] = clust_members.mean(axis=0)
    

    def assign_membership(self, X):
        """
        Assign each sample to a cluster via euclidean distance
        """
        sample_clusters = []

        for x in X:
            dists = [np.linalg.norm(x - c) for c in self.clusters]
            nearest_cluster = np.argmin(dists)
            sample_clusters.append(nearest_cluster)

        return np.array(sample_clusters)


class Spectral_KMeans:
    """
    Spectral Relaxation of K-Means
    """
    def __init__(self, k):
        self.k = k   # Number of means
        self.kmeans =  KMeans(self.k)


    def fit(self, X, stopping_criteria=10, max_iters=1000):
        """
        Fit to some data.

        Parameters:
        -----------
            X: np.array
                N x M matrix of data
            stopping_criteria: int
                Will stop when number of changes from one iteration to another is less than this
            max_iters: int
                Max number of iteration to recompute clusters until stop
        
        Returns:
        --------
        list
            Centroid of each cluster
        """
        self.clusters = []

        # 1. Convert to n x k matrix of top-k eigenvalues such that k << m
        top_k_eigvecs = self.transform_data(X)

        # 2. Run K-Means on smaller matrix
        self.clusters = self.kmeans.fit(top_k_eigvecs, stopping_criteria, max_iters)

        return self


    def transform_data(self, X):
        """
        Transform X -> topk eigenvectors of X^T X
        """
        eigvals, eigvecs = np.linalg.eigh(X @ X.T)
        topk_ix = np.argsort(eigvals)[::-1][:self.k]
        
        return eigvecs[topk_ix].T


    def assign_membership(self, X):
        """
        Just calls standard KMeans
        """
        X = self.transform_data(X)
        return self.kmeans.assign_membership(X)



def main():
    parser = argparse.ArgumentParser(description='Logistic Regression parameters')
    parser.add_argument('--K', help="Number of groups K", type=int, default=10)
    args = parser.parse_args()

    K = args.K

    # Seed so random dataset same each time
    np.random.seed(42)

    # Use a 1000 samples of dim 50
    data = np.random.uniform(0, 10, size=(1000, 50))

    kmeans_clusters = KMeans(K).fit(data) 
    spectral_clusters = Spectral_KMeans(K).fit(data)

    ### Analyze results
    kmeans_membership = kmeans_clusters.assign_membership(data)
    spectral_membership = spectral_clusters.assign_membership(data)

    kmeans_counts = np.unique(kmeans_membership, return_counts=True)[1]
    spectral_counts = np.unique(spectral_membership, return_counts=True)[1]
    print(f"Wasserstein for K={K} --> {wasserstein_distance(kmeans_counts, spectral_counts):.2f}")



if __name__ == "__main__":
    main()
