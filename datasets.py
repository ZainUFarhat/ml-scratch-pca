# numpy
import numpy as np

# sklearn
from sklearn import datasets
from sklearn.preprocessing import minmax_scale


# datasets
class Datasets():

    """
    Description:
        Holds different classification datasets
    """

    # breast cancer
    def load_breast_cancer(self):

        """
        Description:
            Loads sklearn's Breast Cancer Dataset

        Parameters:
            None
        
        Returns:
            X, y, class_names
        """
        
        # load dataset
        data = datasets.load_breast_cancer()

        # load features, labels, and class names
        X, y, class_names = data.data, data.target, data.target_names

        # return
        return X, y, class_names

    # iris
    def load_iris(self):

        """
        Description:
            Loads sklearn's Iris Dataset

        Parameters:
            None
        
        Returns:
            X, y, class_names
        """
        
        # load dataset
        data = datasets.load_iris()

        # load features, labels, and class names
        X, y, class_names = data.data, data.target, data.target_names

        # return
        return X, y, class_names
    
    # diabetes
    def load_diabetes(self):

        """
        Description:
            Loads sklearn's Diabetes Dataset

        Parameters:
            None
        
        Returns:
            X, y_classification, class_names
        """

        # load the dataset
        data = datasets.load_diabetes()

        # load features, "labels", and class names
        X, y  = data.data, data.target

        # convert y to labels we want using median, if valua > median assign True, else False
        # we will use 1 (True) for has diabetes and 0 (False) for no diabetes
        y_classification = np.array([y > np.median(y)]).reshape(-1)   # (y > y.median()).astype(int)

        class_names = ['non-diabetic', 'diabetic']

        # return
        return X, y_classification, class_names