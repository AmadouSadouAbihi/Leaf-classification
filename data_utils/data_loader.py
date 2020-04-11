#####
#   BA    Amadou     16 187 314 
#   YING  Xu         18 205 032
#   ABOU Hamza       17 057 836
###
import os
import pandas as pd


class DataLoader:

    def __init__(self):
        self.path = 'dataset'
        self.train_data, self.test_data = None, None

    def load_data(self):
        """
        :return: Return the three raw datasets
        """
        self.train_data = pd.read_csv(os.path.join(self.path, 'train.csv'))
        self.test_data = pd.read_csv(os.path.join(self.path, 'test.csv'))
        

        return self.train_data, self.test_data
