import os, sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from metrics.metrics import Metrics
from data_utils.data_preprocessing import DataPreprocessing
from cross_validation.cross_validation import CrossValidation

class AbstractClassifier:

    """
    Parent class of all project classifiers.

    Attributes:
        model : An object that defines the classifier model to implement.
        metrics : An object that defines the different metrics that can be used to evaluate a model.
        x_train : The features of the training data
        t_train : The targets of training data (the ground truth label)
        x_test :  The features of the testing data
        t_test : The targets of training data (the ground truth label)
    """
    def __init__(self, model, approch='0'):

    	return 0;

    def train(self):

    	return 0;

    def predict(self, x):

    	return 0;

    def evaluate(self, label="Training", metrics="accuracy"):

    	return 0;


    def setting_model(self, hyperparameters, kfold, metrics):

    	return 0;