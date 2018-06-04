'''
Author: Shuai Zhang
Homepage: http://www.cse.unsw.edu.au/~z5122282/
E-mail: cheungshuai@outlook.com
If you use this code, please site the following paper:
AutoSVD++: An Efficient Hybrid Collaborative Filtering Model via Contractive Auto-encoders
'''
import numpy as np
import pickle

class AutoSVD():
    '''
    AutoSVD uses sparse matrix.
    '''
    def __init__(self, path_of_feature_file, epochs=30, num_factors=10, gamma1=0.01,
                 gamma2=0.01, lambda1=0.1, lambda2=0.1, beta=0.1):
        self.epochs = epochs
        self.num_factors = num_factors
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.lambda1 =lambda1
        self.lambda2 = lambda2
        self.beta = beta
        self.path_of_feature_file = path_of_feature_file


    def train(self, train_data):
        users, items = train_data.nonzero()
        num_of_ratings = len(users) #len(items)
        sum_of_ratings = 0
        for u, i in zip(users,items):
            sum_of_ratings += train_data._get_single_element(u,i)
        average_rating_train = float(sum_of_ratings/num_of_ratings)
        self.average_rating = average_rating_train

        m, n = train_data.shape
        self.item_features = self.readItemFeature()
        V = np.random.rand(self.num_factors, n) * 0.01
        U = np.random.rand(self.num_factors, m) * 0.01
        B_U = np.zeros(m)
        B_I = np.zeros(n)

        for epoch in range(self.epochs):
            for u, i in zip(users, items):
                error = train_data._get_single_element(u,i) - self.predict(average_rating_train, B_U[u], B_I[i], U[:, u], V[:, i], self.beta * self.item_features[i])

                #update parameters
                B_U[u] += self.gamma1 * (error - self.lambda1 * B_U[u])
                B_I[i] += self.gamma1 * (error - self.lambda1 * B_I[i])
                V[:, i] += self.gamma2 * (error * U[:, u] - self.lambda2 * V[:, i])
                U[:, u] += self.gamma2 * (error * (V[:, i] +self.beta * self.item_features[i]) - self.lambda2 * U[:, u])

        self.B_U = B_U
        self.B_I = B_I
        self.V = V
        self.U = U

    def predict(self, average_rating, b_u, b_i, U_u, V_i, itemfeatures):
        return average_rating + b_u + b_i + np.dot(U_u.T, (V_i +  itemfeatures))

    def evaluate(self, data_set):
        users, items = data_set.nonzero()
        num_of_ratings = len(users)
        sum_for_rmse = 0
        sum_for_mae = 0
        for u, i in zip(users, items):
            error = data_set._get_single_element(u,i) - self.predict(self.average_rating,self.B_U[u],self.B_I[i], self.U[:, u], self.V[:, i],self.beta * self.item_features[i])
            sum_for_rmse += error ** 2
            sum_for_mae += abs(error)
        rmse = np.sqrt(sum_for_rmse/num_of_ratings)
        mae = sum_for_mae/num_of_ratings

        print("AutoSVD RMSE = {:.5f}".format(rmse), "AutoSVD MAE = {:.5f}".format(mae))

        return rmse,mae

    def readItemFeature(self):
        with open(self.path_of_feature_file, 'rb') as fp:
            features = pickle.load(fp)
            return features