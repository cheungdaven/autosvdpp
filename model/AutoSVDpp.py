'''
Author: Shuai Zhang
Homepage: http://www.cse.unsw.edu.au/~z5122282/
E-mail: cheungshuai@outlook.com
If you use this code, please site the following paper:
AutoSVD++: An Efficient Hybrid Collaborative Filtering Model via Contractive Auto-encoders
'''
import numpy as np
import pickle
import time

class AutoSVDpp():
    '''
    This is a noraml version of AutoSVD++.
    '''
    def __init__(self, path_of_feature_file, epochs=20,num_factors=10,
                 gamma1=0.007, gamma2=0.007, lambda1=0.005, lambda2=0.015, beta=0.1):
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


        num_of_ratings = len(users)
        sum_of_ratings = 0
        item_by_users = {}
        for u, i in zip(users,items):
            sum_of_ratings += train_data[u,i]
            item_by_users.setdefault(u,[]).append(i)
        average_rating_train = float(sum_of_ratings/num_of_ratings)

        m, n = train_data.shape
        self.average_rating = average_rating_train
        self.item_by_users = item_by_users
        self.item_features = self.readItemFeature()
        V = np.random.rand(self.num_factors, n) * 0.1
        U = np.random.rand(self.num_factors, m) * 0.1
        Y = np.random.rand(n,self.num_factors) * 0.1
        B_U = np.zeros(m)
        B_I = np.zeros(n)

        start_time = time.time()
        for epoch in range(self.epochs):

            for u, i in zip(users, items):
                n_u = len(item_by_users[u])
                sqrt_n_u = np.sqrt(n_u)
                sum_y_j = np.zeros(self.num_factors)
                for j in item_by_users[u]:
                    sum_y_j += Y[j, :]
                dot = np.dot((sum_y_j / sqrt_n_u + U[:,u]).T, (V[:,i] + self.beta * self.item_features[i]))
                error = train_data[u, i] - (average_rating_train + B_U[u] + B_I[i] + dot)

                #update parameters
                B_U[u] += self.gamma1 * (error - self.lambda1 * B_U[u])
                B_I[i] += self.gamma1 * (error - self.lambda1 * B_I[i])
                V[:, i] += self.gamma2 * (error * (sum_y_j / np.sqrt(n_u) + U[:,u]) - self.lambda2 * V[:,i])
                U[:, u] += self.gamma2 * (error * ( V[:,i] +  self.beta * self.item_features[i]) - self.lambda2 * U[:,u])

                for item in item_by_users[u]:
                    Y[item, :] += self.gamma2 * (error * 1 / np.sqrt(n_u) * (V[:,i] + self.beta * self.item_features[i]) - self.lambda2 * Y[item, :])


        self.B_U = B_U
        self.B_I = B_I
        self.V = V
        self.U = U
        self.Y = Y

        #print("--- %s seconds --- in AutoSVD++ not efficient" % (time.time() - start_time))

    def updateStepSize(self, gama, coefficient= 0.9):
        return gama * coefficient

    def evaluate(self, data_set):
        users, items = data_set.nonzero()
        num_of_ratings = len(users)
        sum_for_rmse = 0
        sum_for_mae = 0
        for u, i in zip(users, items):
            n_u = len(self.item_by_users[u])
            sqrt_n_u = np.sqrt(n_u)
            sum_y_j = np.zeros(self.num_factors)
            for j in self.item_by_users[u]:
                sum_y_j += self.Y[j, :]
            dot = np.dot((sum_y_j / sqrt_n_u + self.U[:, u]).T, (self.V[:, i] + self.beta * self.item_features[i]))
            error = data_set[u, i] - (self.average_rating + self.B_U[u] + self.B_I[i] + dot)
            sum_for_rmse += error ** 2
            sum_for_mae += abs(error)
        rmse = np.sqrt(sum_for_rmse /num_of_ratings)
        mae = sum_for_mae/num_of_ratings
        print("AutoSVD++ RMSE = {:.5f}".format(rmse), "AutoSVD++ MAE = {:.5f}".format(mae))
        return rmse,mae

    def readItemFeature(self):
        with open(self.path_of_feature_file, 'rb') as fp:
            features = pickle.load(fp)
            return features