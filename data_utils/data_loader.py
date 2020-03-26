import pandas as pd 
import numpy as np
import random 
import cv2
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from PIL import Image
from tqdm import tqdm

class DataManager:

    train_file = "../dataset/train.csv"
    images_repo = "../dataset/images/"    
    r = random.randint(17, 27)
    X_data_train = pd.DataFrame()
    X_data_test = pd.DataFrame()
    X_img_train = pd.DataFrame()
    X_img_test = pd.DataFrame()
    X_all_train = pd.DataFrame()
    X_all_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()
    classes = []
    id_img_train=[]
    id_img_test=[]
def __init__(self,nb_test_data = 0.2, pca=False):
        ###nb_test_data is the percentage of test data from the original file
        self._nb_test_data = nb_test_data
        self._pca = pca
def _extractBasicData(self):
        """
        This function generates basic train and test data
        """
        train = pd.read_csv(self._train_file)
        
        s = LabelEncoder().fit(train.species)  
        self._classes = list(s.classes_)  
        classes_labels = s.transform(train.species)
        train = train.drop(['species'], axis=1)

        if self._pca:
            trainX = train.drop(['id'], axis=1)
            pca = PCA(n_components=0.85 ,svd_solver='full')
            pca.fit(trainX)
            trainX=pca.transform(trainX)
            train_df=pd.DataFrame.from_records(trainX)
            train_df.insert(loc=0, column='id', value=train['id'])
            train=train_df

        sss = StratifiedShuffleSplit(n_splits=1,  test_size=self._nb_test_data, random_state=self._r)
        for train_index, test_index in sss.split(train, classes_labels):  
            X_train, X_test = train.values[train_index], train.values[test_index]  
            self._y_train, self._y_test = classes_labels[train_index], classes_labels[test_index]

        self.id_img_train =  list(np.int_( X_train[:,0]))
        self.id_img_test =  list(np.int_( X_test[:,0])) 
        self.X_data_train = np.delete(X_train, 0, 1)
        self.X_data_test = np.delete(X_test, 0, 1)
# @ TO BE CONTINUED 