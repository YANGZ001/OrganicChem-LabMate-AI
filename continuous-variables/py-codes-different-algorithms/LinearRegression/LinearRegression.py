from sklearn import datasets #引入数据库
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression #引入线性回归模型
import matplotlib.pyplot as plt #引入matplotlib的pypolt
from sklearn.model_selection import train_test_split
import  time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print(time.strftime("%Y/%m/%d  %H:%M:%S"))## 带日期的12小时格式
filename = 'train_data_16.csv'
train = pd.read_csv(filename,engine='python')
# train.shape
array = train.values
X = array[:,1:-1] #从第一列到倒数第二列
# dis = cdist(X, X)
Y = array[:,-1]
#

train_X, test_X, train_y, test_y = train_test_split(X, Y, random_state=1, train_size= 0.5)
# print('train_X:', train_X)
# print('train_y:', train_y)
# print('test_X:', test_X)
# print('test_y:', test_y)

model = LinearRegression(normalize=True, n_jobs=-1,) # 模型选择用线性回归
model.fit(train_X, train_y) #拟合x,y
predic_y = model.predict(test_X)

print("prediction of test_X:", model.predict(test_X))
print("true test_y: ", test_y)
#
# plt.scatter(list(range(5)),test_y,c = "b",label='test_y')
# plt.scatter(list(range(5)),predic,c = "r", label='predic of y')
# plt.legend(loc='best')
# # plt.xlabel("test_X")
# # plt.ylabel('test_y')
# plt.show()
# print("intercept:", model.intercept_) #跟y轴的交点
# print("coef:", model.coef_) #有13个自变量

MSE = mean_squared_error(test_y, predic_y)
MAE = mean_absolute_error(test_y, predic_y)
r2 = model.score(test_X, test_y)
score_sum = [MAE, MSE, r2]
print("params:", model.get_params()) #得到model模型的参数
print("score of R^2:", r2) #给模型打分,线性回归是用的决定系数R^2
print('mean_squared_error', MSE)
print('mean_absolute_error', MAE)
# plt.bar(range(len(score_sum)), height=score_sum)
# plt.show()
#X 是samples， y是true value of X
