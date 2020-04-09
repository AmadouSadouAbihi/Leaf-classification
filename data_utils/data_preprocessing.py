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
            pca = PCA(n_components=0.9 ,svd_solver='full')
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
    
    def percentage_blackWhite(self,image):
        """
        This function calculates the percentage of black and white pixels in the image
        image: image object
        return: p0: percentage of black pixels
                p1:percentage of white pixels
        """        
        h = image.histogram()
        nt, n0, n1 = sum(h), h[0], h[-1]
        black = round(100*n0/nt,2) 
        white = round(100*n1/nt,2)  
        return black, white

    def ratio_width_length(self,image):
        """
        This function calculates the ratio between width & length
        image: image object
        return: width/length: ratio
        """                
        width, length =image.size  #largeur image[1,:],longueur image[:,1]
        return width/length

    def contour_Features(self,imagefile):   
        """
        This function calculates different image features based on contour-detection 
        imagefile: image path
        return: peak: nb of peaks of the contour
                eccentricity: eccentricity of the ellipse that fits the contour
                angle: deviation of the ellipse that fits the contour
                m: gradient of the line that fits the contour
                y0: image of the abscissa 0 by the equation of the line that fits the contour
        """
        original_color = cv2.imread(imagefile)
        original = cv2.cvtColor(original_color, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(original,(5,5),0)
        ret, thresh = cv2.threshold(blur,50,255,cv2.THRESH_BINARY)
        edges = cv2.Canny(thresh,100,200)
        contours, hierarchy = cv2.findContours(edges.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        LENGTH = len(contours)            
        big_cnt = np.vstack([contours[c] for c in range (0,LENGTH)])
        perimetre = cv2.arcLength(big_cnt,True)
        approx = cv2.approxPolyDP(big_cnt,0.01*perimetre,True)

        peak = len(approx)

        # angle at which object is directed Major Axis and Minor Axis lengths.
        (x, y), (MA, ma), angle = cv2.fitEllipse(big_cnt)

        a = ma / 2
        b = MA / 2
        eccentricity = math.sqrt(pow(a, 2) - pow(b, 2))
        eccentricity = round(eccentricity / a, 2)

        [vx,vy,x,y] = cv2.fitLine(big_cnt, cv2.DIST_L2,0,0.01,0.01) #vx,vy are normalized vector collinear to the line and x0,y0 is a point on the line
        m=vy[0]/vx[0]
        y0=y[0]-m*x[0]

        return peak, eccentricity, angle, m, y0
    
    def extractImagesCaracteristics(self,N):
        """
        This function calculates all image features using the previous methods
        N: list that contains ids of the images
        9 features for each image, nb_images=1584
        return: a dataframe of the features
        """   
        image_data= [[0] * 8 for _ in range(len(N))]  

        for i in tqdm(range(0,len(N))):
            imagefile=self.images_repo+str(N[i])+".jpg"
            image = Image.open(imagefile)
            image = image.convert('1')
            image = self.remove_frame(image)

            #percentage of black and white pixels
            image_data[i][0], image_data[i][1] = self.percentage_blackWhite(image) 
             
            image_data[i][2] = self.ratio_width_length(image)

            peak, eccentricity, angle ,m ,y0 = self.contour_Features(imagefile)
            
            image_data[i][3] = peak
            image_data[i][4] = eccentricity
            image_data[i][5] = angle
            image_data[i][6] = m
            image_data[i][7] = y0
        
        return pd.DataFrame(data=image_data, columns=['black_pxl%', 'white_pxl%', 'ratio_W/L','nb_peak','ellipse_eccentricity','ellipse_deviation','line_gradient','line_y0'])

    def extractImageData(self):
        """
        This function apply the function extractImageData to train and test data
        """ 
        if len(self.id_img_train) == 0 or len(self.id_img_test) == 0  :
            self.loadBasicData()

        if len(self.X_img_train) == 0:
            self.X_img_train=self.extractImagesCaracteristics(self.id_img_train).to_numpy()

        if len(self.X_img_test) == 0:
            self.X_img_test=self.extractImagesCaracteristics(self.id_img_test).to_numpy()
    
    #Public Methods 

    def loadNumericTrainData(self):
        """
        This function calls the private function loadBasicData() to extract train data
        :return: X_data_train: Train matrix
        """
        if len(self.X_data_train)==0 :
            self.loadBasicData()

        return self.X_data_train 

    def loadNumericTestData(self):          
        """
        This function calls the private function loadBasicData() to extract test data
        :return: X_data_test : Test matrix
        """
        if len(self.X_data_test)==0 :
            self.loadBasicData()

        return  self.X_data_test

    def getTrainTargets(self):        
        """
        This function calls the private function loadBasicData() to extract train Targets if they aren't already extracted
        :return: A vector of data classes
        """
        if len(self.t_train)==0 :
            self.loadBasicData()

        return self.t_train
    
    def getTestTargets(self):
        """
        This function calls the private function loadBasicData() to extract test Targets if they aren't already extracted
        :return: A vector of data classes
        """  
        if len(self.t_test)==0 :
            self.loadBasicData()

        return self.t_test
    
    def getListOfClasses(self):
        """
        This function  lists all the classes
        :return: vector of all classes
        """ 
        if len(self.classes)==0 :
            self.loadBasicData()

        return self.classes
    
    def loadImageTrainData(self):          
        """
        This function extract image features for train data
        :return: _X_img_train : Train image features matrix
        """
        if len(self.X_img_train)==0 :
            self.loadBasicData()

        return self.X_img_train
 
    def loadImageTestData(self):          
        """
        This function extract image features for test data
        :return: X_img_test  : Test image features matrix
        """
        if len(self.X_img_test)==0:
            self.loadBasicData()

        return self.X_img_test

    def combineNumericAndImageTrainData(self):          
        """
        This function merge basic train data with image features
        :return: _X_train: Train matrix
        """
        if len(self.X_all_train)==0:
            if len(self.X_img_train)==0 :
                self.loadBasicData()

            self.X_all_train=np.concatenate((self.X_data_train, self.X_img_train), axis=1)

        return self.X_all_train

    def combineNumericAndImageTestData(self):          
        """
        This function merge basic train data with image features
        :return: _X_test: Test matrix
        """
        if len(self.X_all_test)==0:
            if len(self.X_img_test)==0 :
                self.loadBasicData()

            self.X_all_test=np.concatenate((self.X_data_test, self.X_img_test), axis=1)

        return self.X_all_test