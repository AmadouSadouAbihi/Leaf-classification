#####
#   BA    Amadou     16 187 314 
#   YING  Xu         18 205 032
#   ABOU Hamza       17 057 836
###
import os, sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from sklearn.linear_model import LogisticRegression
from classifiers.abstract_classifier import AbstractClassifier


class LogisticRegressionClassifier(AbstractClassifier):

    def __init__(self, penalty='l1', C=1, approch='0'):
        super().__init__(LogisticRegression(penalty=penalty, solver="liblinear", C=C), approch)