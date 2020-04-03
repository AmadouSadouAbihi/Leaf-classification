import os, sys
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from metrics.metrics import Metrics
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))


class CrossValidation:
    """
    This class does k cross validation
    """
    def __init__(self, model, hyperparameters, kfold):

    	return 0;

    def fit_and_predict(self, x_train, t_train, x_test, t_test, metrics):

    	return 0;

    def get_score(self, x_test, y_test):

    	return 0;