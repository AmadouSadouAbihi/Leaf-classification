#####
#   BA    Amadou     16 187 314 
#   YING  Xu         18 205 032
#   ABOU Hamza       17 057 836
###
import os, sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from sklearn.ensemble import RandomForestClassifier
from classifiers.abstract_classifier import AbstractClassifier


class RandomForestAlgorithmClassifier(AbstractClassifier):

    def __init__(self, approch='0'):
        model = RandomForestClassifier(n_estimators=100)
        super().__init__(model, approch)