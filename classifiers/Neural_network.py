  
import os, sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from sklearn.neural_network import MLPClassifier
from classifiers.abstract_classifier import AbstractClassifier



class NeuralnetworkClassifier(AbstractClassifier):
    def __init__(self, hidden_layer_sizes=(10,), activation='relu', alpha=1e-2, learning_rate_init=1e-3, solver='adam', approch='0'):
        self.model = MLPClassifier(max_iter=1000,random_state=None,hidden_layer_sizes=hidden_layer_sizes,activation=activation,alpha=alpha, learning_rate_init=learning_rate_init,solver=solver)
        super().__init__(self.model, approch)
