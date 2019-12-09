from sklearn import datasets #引入数据库
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt #引入matplotlib的pypolt
from sklearn.model_selection import train_test_split
import  time

print(time.strftime("%Y/%m/%d  %H:%M:%S"))## 带日期的12小时格式
filename = 'train_data.csv'
train = pd.read_csv(filename,engine='python')
# train.shape
array = train.values
X = array[:,1:-1] #从第一列到倒数第二列
# dis = cdist(X, X)
Y = array[:,-1]
#

train_X, test_X, train_y, test_y = train_test_split(X, Y, random_state=1, train_size= 0.5)
model = RandomForestRegressor(n_estimators=600, max_depth=4, max_features='sqrt') # 模型选择用线性回归
model.fit(train_X, train_y) #拟合x,y
predic = model.predict(test_X)
# print("prediction:", model.predict(test_X[:4]))
# print("true", test_y[:4])
plt.scatter(list(range(5)),test_y,c = "b")
plt.scatter(list(range(5)),predic,c = "r")

# plt.xlabel("test_X")
# plt.ylabel('test_y')
plt.show()
# print("intercept:", model.intercept_) #跟y轴的交点
# print("coef:", model.coef_) #有13个自变量

print("params:", model.get_params()) #得到model模型的参数
print("score", model.score(test_X, test_y)) #给模型打分,线性回归是用的min_squared_error
