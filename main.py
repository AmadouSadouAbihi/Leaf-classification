import os, sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
import numpy as np

from classifiers.logistic_regression import LogisticRegressionClassifier
from classifiers.random_forest import RandomForestAlgorithmClassifier
from classifiers.svm_classifier import SVMClassifier
from classifiers.adaboost import AdaBoostAlgorithmClassifier
from classifiers.Neural_network import Neuralnetwork
from classifiers.LinearDiscriminantAnalysis_Classifier import LinearDiscriminantClassifier

# python3 classifier  metirics approch
def main():
	if sys.argv[1] == 'logistic_regression':
		if sys.argv[2] in ['accuracy', 'confusion_matrix']:

			model = LogisticRegressionClassifier(approch=sys.argv[4])
            model.train()
            model.evaluate(label="Training", metrics=sys.argv[2])
            model.evaluate(label="Testing", metrics=sys.argv[2])

            #cross validatation
            kfold = 5
            hyperparameters = {
                    'penalty': ['l1', 'l2'],
                    'C': np.logspace(0, 4, 10)
                }
            model.setting_model(hyperparameters, kfold, sys.argv[2])

    elif sys.argv[1] == 'random_forest':
    	if sys.argv[2] in ['accuracy', 'confusion_matrix']:

    		model = RandomForestAlgorithmClassifier(mode=sys.argv[4])
            model.train()
            model.evaluate(label="Training", metrics=sys.argv[2])
            model.evaluate(label="Testing", metrics=sys.argv[2])

            kfold = 5

            n_estimators = [450, 470, 500]
            max_features = ['auto']
            max_depth = [80, 90, 110]
            max_depth.append(None)
            min_samples_split = [5]
            min_samples_leaf = [4]
            bootstrap = [True]

            hyperparameters = {
                    'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap
                }
            model.setting_model(hyperparameters, kfold, sys.argv[2])    

          
if __name__ == "__main__":
    main()