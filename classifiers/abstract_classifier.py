#####
#   BA    Amadou     16 187 314 
#   YING  Xu         18 205 032
#   ABOU Hamza       17 057 836
###
import os, sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from metrics.metric import Metrics
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
        self.model = model
        self.metrics = Metrics()

        if approch == '1': # Only basic data without PCA
            self.X_train = DataPreprocessing(pca=False).loadNumericTrainData()
            self.X_test = DataPreprocessing(pca=False).loadNumericTestData()
            self.t_train = DataPreprocessing(pca=False).getTrainTargets()
            self.t_test = DataPreprocessing(pca=False).getTestTargets()

        elif approch== '2': # Only basic data without PCA
            self.X_train = DataPreprocessing(pca=True).loadNumericTrainData()
            self.X_test = DataPreprocessing(pca=True).loadNumericTestData()
            self.t_train = DataPreprocessing(pca=True).getTrainTargets()
            self.t_test = DataPreprocessing(pca=True).getTestTargets()

        elif approch == '3': # Only combined data without PCA
            self.X_test = DataPreprocessing(pca=False).combineNumericAndImageTestData()
            self.X_train = DataPreprocessing(pca=False).combineNumericAndImageTrainData()
            self.t_train = DataPreprocessing(pca=False).getTrainTargets()
            self.t_test = DataPreprocessing(pca=False).getTestTargets()

        elif approch == '4': # Only combined data with PCA
            self.X_test = DataPreprocessing(pca=True).combineNumericAndImageTestData() 
            self.X_train= DataPreprocessing(pca=True).combineNumericAndImageTrainData()
            self.t_train = DataPreprocessing(pca=True).getTrainTargets()
            self.t_test = DataPreprocessing(pca=True).getTestTargets()



    def train(self):

    	self.model.fit(self.X_train, self.t_train)

    def predict(self, x):

    	return self.model.predict(x)

    def evaluate(self, label="Training", metrics="accuracy"):
        """
         label : Training | Testing 
         metrics : accuracy | confusion_matrics
        """
        if label == 'Training':
            x, y = self.X_train, self.t_train
        else:
            x, y = self.X_test, self.t_test

        if metrics == "accuracy":
            self.metrics.accuracy(self.model, y, x, label)

        elif metrics == "confusion_matrix":
            self.metrics.confusion_matrix(self.model, y, x, label)


    def setting_model(self, hyperparameters, kfold, metrics):

        cross_validate_model = CrossValidation(self.model, hyperparameters, kfold)
        cross_validate_model.fit_and_predict(self.X_train, self.t_train, self.X_test, self.t_test, metrics)
        
        return cross_validate_model.get_score(self.X_test, self.t_test)
