import scipy.io
import numpy as np 
from matplotlib.image import imsave
from matplotlib import pyplot as plt



class PCA:
    """
    Implementation of PCA
    """
    def __init__(self, num_components):
        self.num_components = num_components 


    def fit(self, X):
        """
        Fit PCA on data matrix X
        """
        # Center values of each row
        center_X = X - X.mean(axis=1, keepdims=True)

        # Calculate sample cov matrix
        # S = 1/n-1 X^T X
        cov = 1 / (X.shape[0] - 1) * center_X.T @ center_X

        # Get the top-k eigenvectors
        _, eigvectors = np.linalg.eig(cov)
        self.topk_eigs = eigvectors[:self.num_components]

        print(self.topk_eigs.shape)

        print(f"Reconstruction loss using {self.num_components} components:", round(self.reconstruction_loss(X), 2))

        return self


    def reconstruction_loss(self, X):
        """
        Reconstruction loss for given number of principal components
        """
        diff = X - (X @ self.topk_eigs.T @ self.topk_eigs)

        return np.linalg.norm(diff)**2

    
    def compress(self, X):
        """
        Compress imgs by reducing dimensionality
        """
        return X @ self.topk_eigs.T


    def reconstruct(self, X):
        """
        Reconstruct the reduced imgs
        """
        return X @ self.topk_eigs



def main():
    data = scipy.io.loadmat('USPS.mat')['A']

    for num_components in [10, 50, 100, 200]:
        pca = PCA(num_components).fit(data)
        compressed_imgs = pca.compress(data)
        reconstructed_imgs = pca.reconstruct(compressed_imgs)

        for i in range(2):
            img = reconstructed_imgs[i].reshape(16, 16)
            imsave(f"imgs/img-{i}_components-{num_components}.png", img, dpi=300)
            # plt.imshow(img1)
    

if __name__ == "__main__":
    main()
