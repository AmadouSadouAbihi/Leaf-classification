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
			kfold = 5
	   		model = LogisticRegressionClassifier(approch=sys.argv[3])
                        model.train()
            		model.evaluate(label="Training", metrics=sys.argv[2])
            		model.evaluate(label="Testing", metrics=sys.argv[2])
            		hyperparameters = {
                   	 'penalty': ['l1', 'l2'],
                   	 'C': np.logspace(0, 4, 10)
                	}
            		model.setting_model(hyperparameters, kfold, sys.argv[2])

    	elif sys.argv[1] == 'random_forest':
    		if sys.argv[2] in ['accuracy', 'confusion_matrix']:
    			kfold = 5
			model = RandomForestAlgorithmClassifier(approch=sys.argv[3])
            		model.train()
            		model.evaluate(label="Training", metrics=sys.argv[2])
            		model.evaluate(label="Testing", metrics=sys.argv[2])
            		hyperparameters = {
                    		'max_depth': np.linspace(10, 100, num = 10),   # best=50
            			'n_estimators':range(88, 91) 
            		}
      			model.setting_model(hyperparameters, kfold, sys.argv[2]) 
    	elif sys.argv[1] == 'svm':
		if sys.argv[2] in ['accuracy', 'confusion_matrix']:
    			kfold = 5
            		model = SVMClassifier(approch=sys.argv[3])
			model.train()
            		model.evaluate(label="Training", metrics=sys.argv[2])
            		model.evaluate(label="Testing", metrics=sys.argv[2])
            		hyperparameters = {
                		'C': [0.1, 1, 10, 100, 1000],
                		'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                		'kernel':  ['rbf', 'sigmoid']
                	}
            		model.tunning_model(hyperparameters, kfold, sys.argv[2])  
    	elif sys.argv[1] == 'adaboost':
        	if sys.argv[2] in ['accuracy', 'confusion_matrix']:
        		kfold = 5
		    	model = AdaBoostAlgorithmClassifier(approch=sys.argv[3])
		    	model.train()
		    	model.evaluate(label="Training", metrics=sys.argv[2])
		    	model.evaluate(label="Testing", metrics=sys.argv[2])
			hyperparameters = {
		        	'base_estimator': [tree.DecisionTreeClassifier(max_depth=n) for n in range(13, 14)],
		        	'n_estimators': [50, 55, 60, 65, 70,80,85],
		        	'algorithm': ['SAMME.R', 'SAMME'],
		        	'learning_rate' : np.geomspace(0.01, 1, num=3)
		        }
		    	model.setting_model(hyperparameters, kfold, sys.argv[2])
	elif sys.argv[1] == 'neural_network':
	 	if sys.argv[2] in ['accuracy', 'confusion_matrix']:
        		kfold = 5
            		model = Neuralnetwork(approch=sys.argv[3])
            		model.train()
            		model.evaluate(label="Training", metrics=sys.argv[2])
            		model.evaluate(label="Testing", metrics=sys.argv[2])
            		hyperparameters = {
                		'hidden_layer_sizes': [(5, ), (5, 5)],
                		'activation': ['relu'],
                		'solver': ['identity', 'relu', 'tanh'],
                		'alpha': [1e-5, 3e-4],
                		'learning_rate_init':  [1e-2, 1e-3]
                	}
            		model.tunning_model(hyperparameters, kfold, sys.argv[2])
    	elif sys.argv[1] == 'linear_discriminant':
        	if sys.argv[2] in ['accuracy', 'confusion_matrix']:
        		kfold = 5
        		model = LinearDiscriminantClassifier(approch=sys.argv[3])
        		model.train()
            		model.evaluate(label="Training", metrics=sys.argv[2])
           	 	model.evaluate(label="Testing", metrics=sys.argv[2])
            		hyperparameters = {
             			'penalty' : ['l1','l2'],
                		'multi_class': ['auto'],
                		'solver': ['liblinear','identity', 'relu', 'tanh'],
                		'max_iter': [10000]
                	}
            		model.tunning_model(hyperparameters, kfold, sys.argv[2])		                    
if __name__ == "__main__":
    main()
