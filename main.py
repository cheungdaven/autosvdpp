from util.LoadDataSet import LoadData
from model.AutoSVD import AutoSVD
from model.AutoSVDppE import *

train_data, test_data = LoadData().loadMovielens1M()
autosvd = AutoSVDpp(path_of_feature_file="file/features/ML-1m-features-by-cAE.pickle")
autosvd.train(train_data=train_data, test_data=test_data)
# autosvd.evaluate(test_data)