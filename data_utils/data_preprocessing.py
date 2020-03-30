import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from .data_loader import DataLoader

import random 
import cv2
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from PIL import Image
from tqdm import tqdm

class DataPreprocessing:
    
    images_repo = "../dataset/images/"    
    r = random.randint(17, 27)
    X_data_train = pd.DataFrame()
    X_data_test = pd.DataFrame()
    X_img_train = pd.DataFrame()
    X_img_test = pd.DataFrame()
    X_all_train = pd.DataFrame()
    X_all_test = pd.DataFrame()
    t_train = pd.DataFrame()
    t_test = pd.DataFrame()
    classes = []
    id_img_train=[]
    id_img_test=[]
    
    def __init__(self, nb_test_data = 0.2, pca=False):
        """ 
		 nb_test_data is the percentage of test data from the original file
		"""
        self.pca = pca
        self.nb_test_data = nb_test_data
        self.dataset = None
        self.train_data, self.test_data, self.sample_submission = DataLoader().load_data()
        
    def loadtBasicData(self):
        """
        This function generates basic train and test data
        """
        train = pd.read_csv(self.train_data)
        
        s = LabelEncoder().fit(train.species)  
        self.classes = list(s.classes_)  
        classes_labels = s.transform(train.species)
        train = train.drop(['species'], axis=1)

        if self.pca:
            trainX = train.drop(['id'], axis=1)
            pca = PCA(n_components=0.85 ,svd_solver='full')
            pca.fit(trainX)
            trainX=pca.transform(trainX)
            train_df=pd.DataFrame.from_records(trainX)
            train_df.insert(loc=0, column='id', value=train['id'])
            train=train_df

    
        for train_index, test_index in StratifiedShuffleSplit(n_splits=1,  test_size=self.nb_test_data, random_state=self.r).split(train, classes_labels):  
            X_train, X_test = train.values[train_index], train.values[test_index]  
            self.t_train, self.t_test = classes_labels[train_index], classes_labels[test_index]

        self.id_img_train =  list(np.int_( X_train[:,0]))
        self.id_img_test =  list(np.int_( X_test[:,0])) 
        self.X_data_train = np.delete(X_train, 0, 1)
        self.X_data_test = np.delete(X_test, 0, 1)

    def first_right_to_left (self,matrix):
        """
        This function extracts index of the first white pixel from right to left
        matrix: matrix of the image
        return: index_row :row number
                index_col :column number
        """
        j=len(matrix[0,:])-1   
        while max(matrix[:,j])!=float(1):
            j=j-1

        index_col=j

        i=0   
        while matrix[i,j]!=float(1):
            i=i+1

        index_row=i

        return index_row, index_col

    def first_left_to_right(self,matrix):
        """
        This function extracts index of the first white pixel from left to right
        matrix: matrix of the image
        return: index_row :row number
                index_col :column number
        """
        j=0
        while max(matrix[:,j])!=float(1):
            j=j+1
        index_col=j
        i=0
        while matrix[i,j]!=float(1):
            i=i+1
        index_row=i
        return index_row, index_col
    
    def first_top_to_bottom (self,matrix):  
        """
        This function extracts index of the first white pixel from top to bottom
        matrix: matrix of the image
        return: index_row :row number
                index_col :column number
        """
        i=0   
        while max(matrix[i,:])!=float(1):
            i=i+1

        index_row=i

        j=0  
        while matrix[i,j]!=float(1):
            j=j+1

        index_col=j

        return index_row, index_col

    def first_bottom_to_top (self,matrix):
        """
        This function extracts index of the first white pixel from bottom to top
        matrix: matrix of the image
        return: index_row :row number
                index_col :column number
        """
        i=len(matrix[:,0])-1   
        while max(matrix[i,:])!=float(1):
            i=i-1

        index_row=i

        j=0   
        while matrix[i,j]!=float(1):
            j=j+1

        index_col=j

        return index_row, index_col

    def remove_black_frame(self,image):
        """
        This function removes black frame surrounding the leaf in the image
        image: image object
        return: result :image object
        """        
        image_array = np.asarray(image) 

        left_r, left_c = self.first_left_to_right (image_array)
        right_r, right_c = self.first_right_to_left(image_array)
        top_r, top_c = self.first_top_to_bottom (image_array)
        bottom_r, bottom_c = self.first_bottom_to_top (image_array)

        image_array = image_array[top_r:bottom_r+1,left_c:right_c+1]
        result = Image.fromarray(image_array) 

        return result   #return an image
    # to be continued 