import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import time
print('Time starts at: ', time.strftime("%Y/%m/%d  %H:%M:%S"))## 带日期的12小时格式
print('\n')


#load data
filename = 'train_data_16.csv'
train = pd.read_csv(filename,engine='python')
# train.shape
array = train.values
X = array[:,1:-1] #从第一列到倒数第二列
# X.shape
y = array[:,-1]
# Y

#General stuff
seed = 1234
nMAE_lst = list()
print("Find the best parameters: ")
i=10
kfold = KFold(n_splits = i, random_state = seed)
scoring = 'neg_mean_absolute_error'
model = SVR()

#Parameters to tune
param_grid = [{'kernel': ['rbf'],
                'gamma': [1e-3, 1e-4],
                 'C': [1, 10, 100, 1000]},
                 {'kernel': ['poly'],
                 'gamma': [1e-3, 1e-4],
                 'C': [1, 10, 100, 1000]},
                {'kernel': ['linear'],
                'C': [1, 10, 100, 1000]}
                ]


#search best parameters and train
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, n_jobs=6)
grid_result = grid.fit(X, y)

#print the best data cranked out from the grid search
# np.savetxt('best_score', ["best_score: %s" % grid.best_score_, ], fmt ='%s')
best_params = pd.DataFrame([grid.best_params_], columns=grid.best_params_.keys())
# best_params.to_csv('Records of Kfold:2-10.txt', sep= '\t')
# np.savetxt('Records of Kfold:2-10', ["best_params: %s" % grid.best_params_, ], fmt ='%s')
print('KFoldCV = {}, best_score: {}, best_params:{}'.format(i, grid.best_score_, grid.best_params_))
with open('Records of SVM.txt', 'a') as f:
    content = 'KFoldCV = '+str(i)+'\n'+'best_score: '+str(grid.best_score_)+'\n'
    f.write(content)
    f.close()

#Predict the future using test_data_6.csv
model2 = grid.best_estimator_
# model2 = RandomForestRegressor(n_estimators = grid.best_params_['n_estimators'], max_features = grid.best_params_['max_features'], max_depth = grid.best_params_['max_depth'], random_state = seed)
svm_fit = model2.fit(X, y)
#load data
filename2 = 'test_data_6_2.csv'
test = pd.read_csv(filename2,engine='python')
# train.shape
array = test.values
test_X = array[:,1:-1] #从第一列到倒数第二列
# X.shape
test_y = array[:,-1]
# Y
predictions_y = model2.predict(test_X) #对每个test_X，模型有一个predictions
# predictions_df = pd.DataFrame(data=predictions_y, columns=['Prediction']) #model prediction

#scoring: 用测试集与预测集比较看数据。
MAE = mean_absolute_error(test_y, predictions_y)
MSE = mean_squared_error(test_y, predictions_y)
r2 = r2_score(test_y, predictions_y)
with open('Records of SVM.txt', 'a') as f:
    content = '\n'+'MAE = '+str(MAE)+'\n'+'MSE = '+str(MSE)+'\n'+ 'r2 = '+str(r2)+'\n'
    f.write(content)
    f.close()

#cross_val_score： 因为gridsearch的时候已经用了X，y做交叉验证，得到的模型已经是最优，现在再遍历一次没有意义
# for k in range(2, 11):
#     scores = cross_val_score(model2, X, y, cv=k, scoring='neg_mean_absolute_error')
#     print('cross_val_score ,cv={},score = {}'.format(k, scores.mean()))
print('You are awesome! Man!')
print('\n')
print(time.strftime("%Y/%m/%d  %H:%M:%S"))## 带日期的12小时格式
print('\n')
