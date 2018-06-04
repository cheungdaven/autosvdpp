import pandas as pd
from sklearn import cross_validation as cv
import numpy as np
from scipy.sparse import csc_matrix

class LoadData():
    PATH_100K = 'file/datasets/ml-100k/u.data'
    PATH_1M= 'file/datasets/ml-1m/movies_rating_new_id.csv'

    def __init__(self, test_ratio = 0.1):
        '''
        Intial the LoadData Class
        :param test_ratio: given test data ratio when loading data, default 0.1
        '''
        self.test_ratio = test_ratio

    def loadMovielens100K(self, path=PATH_100K):
        '''
        Load the Movielens 100k data set
        :param path: path of the dataset
        :return: train data and test data
        '''
        header = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(path, sep='\t', names=header)
        n_users = df.user_id.unique().shape[0]
        n_items = df.item_id.unique().shape[0]
        print("Number of users=" + str(n_users) + "; Number of items=" + str(n_items))
        train_data, test_data = cv.train_test_split(df, test_size=self.test_ratio)
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
        train_matrix = np.zeros((n_users, n_items))
        test_matrix = np.zeros((n_users, n_items))
        for line in train_data.itertuples():
            train_matrix[line[1] - 1, line[2] - 1] = line[3]
        for line in test_data.itertuples():
            test_matrix[line[1] - 1, line[2] - 1] = line[3]

        return train_matrix, test_matrix

    def loadMovielens1M(self, path=PATH_1M):
        '''
        Load the Movielens 1M data set
        :param path: path of the dataset
        :return: train data and test data
        '''
        header = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(path, sep='::', names=header)
        n_users = df.user_id.unique().shape[0] #n_users = 6040
        n_items = df.item_id.unique().shape[0] #n_items = 3952
        print("Number of users=" + str(n_users) + "; Number of items=" + str(n_items))
        train_data, test_data = cv.train_test_split(df, test_size=self.test_ratio)
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
        train_matrix = np.zeros((n_users, n_items))
        test_matrix = np.zeros((n_users, n_items))
        for line in train_data.itertuples():
            train_matrix[line[1] - 1, line[2] - 1] = line[3]
        for line in test_data.itertuples():
            test_matrix[line[1] - 1, line[2] - 1] = line[3]

        return train_matrix, test_matrix

    def loadMovieTweeting(self, path, test_size=0.2):
        header = ['user_id', 'item_id', 'rating', 'timestamp']
        df = pd.read_csv(path, sep='::', names=header)
        n_users = df.user_id.unique().shape[0]  # 46259
        n_items = df.item_id.unique().shape[0]  # 26586

        print("Number of users=" + str(n_users) + "; Number of items=" + str(n_items))

        train_data, test_data = cv.train_test_split(df, test_size=test_size)
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)

        train_row = []
        train_col = []
        train_rate = []
        count = 0
        for line in train_data.itertuples():
            count += 1
            train_row.append(line[1] - 1)
            train_col.append(line[2] - 1)
            train_rate.append(line[3])
        train_matrix = csc_matrix((train_rate, (train_row, train_col)), shape=(n_users, n_items))
        test_row = []
        test_col = []
        test_rate = []
        for line in test_data.itertuples():
            test_row.append(line[1] - 1)
            test_col.append(line[2] - 1)
            test_rate.append(line[3])
        test_matrix = csc_matrix((test_rate, (test_row, test_col)), shape=(n_users, n_items))
        return train_matrix, test_matrix