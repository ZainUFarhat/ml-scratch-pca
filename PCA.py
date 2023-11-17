# numpy
import numpy as np

# pca
class PCA():

    """
    Description:
        My from scratch implementation of the Principal Component Analysis Algorithm 
    """

    # constructor
    def __init__(self, num_components):

        """
        Description:
            Constructor of our PCA class

        Parameters:
            num_components: number of dimensions we want to reduce our features to
        
        Returns:
            None
        """

        self.num_components = num_components
        # we will keep a reference to the components we reduce to in our class
        self.components = None
        # we will also keep a reference to the mean
        # this is so we can use it throughout our entire class and not have to return it and call it again anywhere
        self.mean = None

    # fit
    def fit(self, X):

        """
        Description:
            Reduces the number of dimensions of our data to the desired num_components passed by user 
        
        Parameters:
            X: features
        
        Returns:
            None
        """

        # find the mean of our initial features
        self.mean = X.mean(axis = 0)
        X = X - self.mean

        # find the covariance matrix of X but transpose X because functions needs samples as columns
        cov = np.cov(X.T)

        # eigenvectors, eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # transpose eigenvectors for easier calculations
        eigenvectors = eigenvectors.T

        # sort eigenvectors in descending order
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[indices]

        # choose the first "num_components" eigenvectors so we can transform into our new set of features
        self.components = eigenvectors[: self.num_components]

    # transform
    def transform(self, X):

        """
        Description:
            Projects our data from original shape to num_components
        
        Parameters:
            X: features to transform
        
        Returns:
            X_projected
        """

        # project our data
        X = X - self.mean
        X_projected = np.dot(X, self.components.T)

        # return
        return X_projected