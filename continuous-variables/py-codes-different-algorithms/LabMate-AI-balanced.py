import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import dump
import time
#load data
# pwd
# cd Desktop/project/experiments
print(time.strftime("%Y/%m/%d  %H:%M:%S"))## 带日期的12小时格式
filename = 'train_data.csv'
train = pd.read_csv(filename, engine='python')
array = train.values
# array
X = array[:,1:-1] #从第一列到倒数第二列
Y = array[:,-1]
# X
#General stuff
seed = 1234  #****
kfold = KFold(n_splits = 10, random_state = seed) #10折，参数需要变化吗？
scoring = 'neg_mean_absolute_error'
model = RandomForestRegressor(random_state=seed)

#Parameters to tune
estimators = np.arange(100, 1050, 50)  #估计器是100到1050的50个数，可不可以变化呢？
estimators_int = np.ndarray.tolist(estimators)
param_grid = {'n_estimators':estimators_int, 'max_features':('auto', 'sqrt'), 'max_depth':[None, 2, 4]}

#search best parameters and train
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, n_jobs=6)
grid_result = grid.fit(X, Y)

#print the best data cranked out from the grid search
np.savetxt('best_score.txt', ["best_score: %s" % grid.best_score_], fmt ='%s')
best_params = pd.DataFrame([grid.best_params_], columns=grid.best_params_.keys())

#Predict the future
filename2 = 'all_combos.csv'
df_all_combos = pd.read_csv(filename2)
# df_all_combos
# df_train_corrected = train.iloc[:,:-1] #取出train数据的前7列
# unseen = pd.concat([df_all_combos, df_train_corrected], sort = False).drop_duplicates(keep=False )
#这一部分是把train的前七列数据取出，与all_combos.txt数据拼接在一起，我觉得这完全是错的因为没有必要啊，明明你的前7列是从all_combos里面取出的，所以all_combos一定已经包含了，因此这一步是多此一举
# array2 = unseen.values
array2 = df_all_combos.values
# array2.shape

# np.where(np.isnan(array2))
# array2[27000:,0] #不知道为什么多了10行，每一行开头都是nan

X2 = array2[:,1:]
# X2
# X2.shape


model2 = RandomForestRegressor(n_estimators = grid.best_params_['n_estimators'], max_features = grid.best_params_['max_features'], max_depth = grid.best_params_['max_depth'], random_state = seed)
RF_fit = model2.fit(X, Y)
predictions = model2.predict(X2)
# np.where(np.isnan(X2))
# X2.shape
predictions_df = pd.DataFrame(data=predictions, columns=['Prediction'])
feat_imp = pd.DataFrame(model2.feature_importances_, index=['nap', '2meth', 'Ligan2Metal', 'Cat', 'Ketone', 'base', 'i_PrOH', 'Time', 'Temperature', 'concentration'], columns=['Feature_importances'])

#get individual tree preds
all_predictions = []
for e in model2.estimators_:
    all_predictions += [e.predict(X2)]

#get variance and dataframe
variance = np.var(all_predictions, axis=0)
variance_df = pd.DataFrame(data=variance, columns=['Variance'])

assert len(variance) == len(predictions)

#concatenate tables
initial_data = pd.DataFrame(data=array2, columns = ['Iteration', 'nap', '2meth', 'Ligan2Metal', 'Cat', 'Ketone', 'base', 'i_PrOH', 'Time', 'Temperature', 'concentration'])
df = pd.concat([initial_data, predictions_df, variance_df], axis=1)

if len(Y) < 19:
	df_sorted = df.sort_values(by=['Variance', 'Catalyst'], ascending=[False, True])
	toPerform = df_sorted.iloc[0]

elif len(Y) >= 19 and np.max(Y[10:]) >= 4 * np.max(Y[:9]):
	df_sorted = df.sort_values(by=['Prediction', 'Catalyst'], ascending=[False, True])
	preliminary = df_sorted.iloc[0:5]
	df_sorted2 = preliminary.sort_values(by=['Variance', 'Catalyst'], ascending=[True, True])
	toPerform = df_sorted2.iloc[0]

else:
	df_sorted = df.sort_values(by=['Prediction', 'Catalyst'], ascending=[False, True])
	preliminary = df_sorted.iloc[0:10]
	df_sorted2 = preliminary.sort_values(by=['Variance', 'Catalyst'], ascending=[False, True])
	toPerform = df_sorted2.iloc[0]

#save data
feat_imp.to_csv('feature_importances.txt', sep= '\t')
best_params.to_csv('best_parameters.txt', sep= '\t')
toPerform.to_csv('selected_reaction.txt', sep = '\t',header= False)
df_sorted.to_csv('predictions.txt', sep = '\t')
filename3 = 'random_forest_model_grid.sav'
dump(grid, filename3)

print('Have a good one, mate!')
print(time.strftime("%Y/%m/%d  %H:%M:%S"))## 带日期的12小时格式
