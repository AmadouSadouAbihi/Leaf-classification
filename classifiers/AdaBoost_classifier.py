import os, sys

sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from sklearn.ensemble import AdaBoostClassifier
from classifiers.abstract_classifier import AbstractClassifier


class AdaBoostClassifier(AbstractClassifier):

    def __init__(self, approch='0'):
        super().__init__(AdaBoostClassifier(n_estimators=50, learning_rate=1), approch)