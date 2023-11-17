# datasets
from datasets import *

# Gaussian Naive Bayes
from PCA import *

# utils
from utils import *

# set numpy random seed
np.random.seed(42)

def main():

    """
    Description:
        Performs our Principal Component Analysis
    
    Parameters:
        None
    
    Returns:
        None
    """

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    dataset_name = 'Breast Cancer'
    
    # create an instance of Datasets class
    datasets = Datasets()

    # load the breast cancer dataset
    X, y, class_names = datasets.load_breast_cancer()

    print(f'Loading {dataset_name} Dataset...')

    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nPrincipal Component Analysis\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Reducing Components...\n')

    # pca hyperparameters
    num_components = 2

    pca = PCA(num_components = num_components)
    pca.fit(X)
    X_projected = pca.transform(X)

    print('Before PCA, X.shape =', X.shape)
    print('After PCA, X.shape =', X_projected.shape)

    print('Done Projecting!') 
    print('---------------------------------------------------Plotting---------------------------------------------------')
    print('Plotting...')

    savepath = 'plots/bc/bc_pca.png'
    is_iris = False
    
    visualize_pca(X_projected, y, dataset_name, class_names, savepath, is_iris)

    print('Please refer to plots/bc directory to view reduced components.')
    print('--------------------------------------------------------------------------------------------------------------\n')

    ######################################################################################################################################

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    dataset_name = 'Iris'

    print(f'Loading {dataset_name} Dataset...')

    # load the iris dataset
    X, y, class_names = datasets.load_iris()

    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nPrincipal Component Analysis\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Reducing Components...\n')

    pca = PCA(num_components = num_components)
    pca.fit(X)
    X_projected = pca.transform(X)

    print('Before PCA, X.shape =', X.shape)
    print('After PCA, X.shape =', X_projected.shape)

    print('Done Projecting!') 
    print('---------------------------------------------------Plotting---------------------------------------------------')
    
    print('Plotting...')

    savepath = 'plots/iris/iris_pca.png'
    is_iris = True
    
    visualize_pca(X_projected, y, dataset_name, class_names, savepath, is_iris)

    print('Please refer to plots/iris directory to view reduced components.')
    print('--------------------------------------------------------------------------------------------------------------\n')
    #######################################################################################################################################

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    dataset_name = 'Diabetes'

    print(f'Loading {dataset_name} Dataset...')

    # load the diabetes dataset
    X, y, class_names = datasets.load_diabetes()

    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nPrincipal Component Analysis\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Reducing Components...\n')

    pca = PCA(num_components = num_components)
    pca.fit(X)
    X_projected = pca.transform(X)

    print('Before PCA, X.shape =', X.shape)
    print('After PCA, X.shape =', X_projected.shape)

    print('Done Projecting!') 
    print('---------------------------------------------------Plotting---------------------------------------------------')
    
    print('Plotting...')

    savepath = 'plots/db/db_pca.png'
    is_iris = False
    
    visualize_pca(X_projected, y, dataset_name, class_names, savepath, is_iris)
    
    print('Please refer to plots/db directory to view reduced components.')
    print('--------------------------------------------------------------------------------------------------------------')

    # return
    return None

if __name__ == '__main__':

    # run everything
    main()