# '''
# Please read the license file.
# LabMate.AI was designed to help identifying optimized conditions for chemical reactions.
# You will need the Python libraries below (NumPy, Pandas and Scikit-learn) and 10 random reactions to run LabMate.AI.
# '''

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import dump
import random
import time
# pwd
print('Welcome! Let me work out what is the best experiment for you to run...')
# print("Time: %s", time.time())
print(time.strftime("%Y/%m/%d  %H:%M:%S"))## 带日期的12小时格式
#
# '''
# The training data should be a tab separated file named 'train_data.txt'. The first column of the file is the reaction identifier and the last column is the objective variable (target). The columns in between correspond to descriptors. Otherwise please change accordingly.
# See example files
# '''
# pwd
# cd ActiveLearning-master/
filename = 'train_data716.txt'
train = pd.read_csv(filename, sep= '\\t', engine='python')
print(train.head())#.head查看头5行
# type(train)

array = train.values
# array
X = array[:,:-1] #从第1列到倒数第二列，注意up to but not including
# X
Y = array[:,-1]
# Y #Y是最后一列数据





'''
General settings below. These do not need to be changed.
The seed value is what makes the whole process deterministic. You may choose to change this number. #种子数很重要，决定性的
The possible number of estimators, max_features and max_depth is a good compromise, but may need to be adapted, if the number of features (columns) is very different.
'''

seed = 1 #随机数种子=1
kfold = KFold(n_splits = 10, shuffle=True, random_state = seed) #10折，数据重新洗牌，随机数种子1
scoring = 'neg_mean_absolute_error' #评分：neg_mean_absolute_error'
model = RandomForestRegressor(random_state=seed) #建模,随机数种子1
estimators = np.arange(100, 1050, 50) #估计器数量，100到1050，取50个数
# estimators
estimators_int = np.ndarray.tolist(estimators) #把array 变成list
# estimators_int
param_grid = {'n_estimators':estimators_int, 'max_features':('auto', 'sqrt'), 'max_depth':[None, 2, 4]}


print('All good till now. I am figuring out the best method to analyze your data. Bear with me...')


'''
This section makes LabMate.AI search for the best hyperparameters autonomously.
It will also save a file with the best score and store the ideal hyperparameters for future use.
'''

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, n_jobs=6) #n_jobs要变成4核试试才行，先6核不要怕
grid_result = grid.fit(X, Y)
np.savetxt('best_score.txt', ["best_score: %s" % grid.best_score_], fmt ='%s')
best_params = pd.DataFrame([grid.best_params_], columns=grid.best_params_.keys())


print('... done! It is going to be lightspeed from here on out! :)')


'''
This section loads all possible reactions (search space) and deletes all previously executed reactions from that file.
The file has the same format as the training data, but no "Target" column. Please check example file.
'''

filename2 = 'all_combos.txt'
df_all_combos = pd.read_csv(filename2, sep= '\\t', engine='python')
df_train_corrected = train.iloc[:,:-1] #测试样本数据
unseen = pd.concat([df_all_combos, df_train_corrected], sort = True).drop_duplicates(keep=False) #把all和train合并，去重
array2 = unseen.values #去重后的数据传入array2
X2 = array2[:,:] #X2的数据为： 合并后的数据，从第二列到最后一列，都是已知参数
# X2.shape
df_all_combos2 = df_all_combos.iloc[:,:] #不要第一列pyridine 不知道为什么？？？





'''
LabMate.AI predicts the future in this section. It builds the model using the best hyperparameter set and predicts the reaction yield (numeric value) for each instance.
For your reference, the method creates a file with the feature importances
'''

model2 = RandomForestRegressor(n_estimators = grid.best_params_['n_estimators'], max_features = grid.best_params_['max_features'], max_depth = grid.best_params_['max_depth'], random_state = seed)
#model2使用之前网格搜索得到的最优参数
RF_fit = model2.fit(X, Y) #用X Y拟合model2
predictions = model2.predict(X2) #预测 X2值
# np.isnan(X2)
predictions_df = pd.DataFrame(data=predictions, columns=['Prediction'])
feat_imp = pd.DataFrame(data=model2.feature_importances_, index=list(df_all_combos2.columns.values), columns=['Feature_importances'])
#得到model2的特征重要性，构建一个dataframe
feat_imp = feat_imp.sort_values(by=['Feature_importances'], ascending = False)
#排序





'''
LabMate.AI calculates variances for the predictions, which allows prioritizing the next best experiment, and creates a table with all the generated information.
'''

all_predictions = []
for e in model2.estimators_:
    all_predictions += [e.predict(X2)]
    #随机森林里产生了很多的estimators 用所有的估计器去预测X2的数据，返回预测值，构建一个包含所有预测值的list
# all_predictions
variance = np.var(all_predictions, axis=0) #计算偏差，以横轴？？
variance_df = pd.DataFrame(data=variance, columns=['Variance'])

assert len(variance) == len(predictions) # control line
initial_data = pd.DataFrame(data=array2, columns = list(unseen.columns.values))
df = pd.concat([initial_data, predictions_df, variance_df], axis=1)  #得到新的表格，包含原始值，预测值，偏差





'''
LabMate.AI now selects the next reaction to be performed.
'''
#看它用什么选
feat_imp_T = feat_imp.transpose() # creates a table with a single row stating the importance (0-1 scale) of each variable
keys1 = list(feat_imp_T.keys()) # collects the names of the features
keys2 = list(feat_imp_T.keys()) # same as above
keys1.insert(7,'Prediction') # Inserts "Prediction" in position 7 of the previously generated list
keys2.insert(7, 'Variance') # Inserts "Variance" in position 7 of the previously generated list

df_sorted = df.sort_values(by=[keys1[-1], keys1[0]], ascending=[False, False]) # Fetches the table with the predictions and variance and sorts: 1) high prediction first; 2) most important feature second (descending order) for overlapping predictions
preliminary = df_sorted.iloc[0:5] # Collects the first five columns 最优数据
df_sorted2 = preliminary.sort_values(by=[keys2[-1], keys2[0]], ascending=[True, False]) # Sorts the top five rows by: 1) Low variance first; 2) most important feature second (descending order) for overlapping predictions
toPerform = df_sorted2.iloc[0] # First row is the selected reaction







'''
Save files
'''

feat_imp.to_csv('feature_importances.txt', sep= '\t')
best_params.to_csv('best_parameters.txt', sep= '\t')
toPerform.to_csv('selected_reaction.txt', sep = '\t', header=False)
df_sorted.to_csv('predictions.txt', sep = '\t')
filename3 = 'random_forest_model_grid.sav'
dump(grid, filename3)

print('You are all set! Have a good one, mate!')




'''
After performing the reaction simply edit the train_data.txt file with the reaction conditions used and target value, before running the script again. Enjoy and happy chemistry :)
'''
print(time.strftime("%Y/%m/%d  %H:%M:%S"))## 带日期的12小时格式
