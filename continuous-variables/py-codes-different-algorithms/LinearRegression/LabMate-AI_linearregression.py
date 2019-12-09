import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import dump
import time
from scipy.spatial.distance import cdist
#load data
# cd Desktop/project/experiments//knnregressor
# pwd
print(time.strftime("%Y/%m/%d  %H:%M:%S"))## 带日期的12小时格式

filename = 'train_data_16.csv' ,engine='python')
# train.shape
array = train.values
X = array[:,1:-1] #从第一列到倒数第二列
# dis = cdist(X, X)
Y = array[:,-1]
#

#General stuff
seed = 1234
#find the best params:
for i in range(2,11):
    kfold = KFold(n_splits = i, random_state = seed) #splits =2, 说明samples=5，neighbors<=5
    scoring = 'neg_mean_absolute_error'#可以变
    model = LinearRegression()

    #Parameters to tune
    param_grid = [{'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}]
    #search best parameters and train
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, n_jobs=6, verbose=0)
    grid_result = grid.fit(X, Y)
    # print(grid.cv_results_)
    # print(grid.cv_results_['split1_test_score'])

    #print the best data cranked out from the grid search
    # np.savetxt('best_score.txt', ["best_score: %s" % grid.best_score_], fmt ='%s')
    best_params = pd.DataFrame([grid.best_params_], columns=grid.best_params_.keys())

    #Predict the future
    filename2 = 'all_combos.csv'
    df_all_combos = pd.read_csv(filename2)
    # df_all_combos.head()
    # df_all_combos.shape #注意，这里新导入的数据，第一列是索引，没有用
    # df_train_corrected = train.iloc[:,:-1]
    # df_train_corrected.shape
    # unseen = pd.concat([df_all_combos, df_train_corrected], sort= False).drop_duplicates(keep=False)
    # unseen.shaped
    array2 = df_all_combos.values
    # array2
    X2 = array2[:,1:]
    # X2
    model2 = grid.best_estimator_
    # model2 = KNeighborsRegressor(n_neighbors = grid.best_params_['n_neighbors'], weights = grid.best_params_['weights'], max_depth = grid.best_params_['max_depth'], random_state = seed)
    knn_fit = model2.fit(X, Y)
    predictions = model2.predict(X2)
    predictions_df = pd.DataFrame(data=predictions, columns=['Prediction'])
    # feat_imp = pd.DataFrame(model2.feature_importances_, index=['nap', '2meth', 'Ligan2Metal', 'Cat', 'base', 'i_PrOH', 'Time', 'Temperature', 'concentration'], columns=['Feature_importances'])
    # feat_imp = feat_imp.sort_values(by=['Feature_importances'], ascending = False)

    # #get individual tree preds
    # all_predictions = []
    # for e in model2.estimators_:
    #     all_predictions += [e.predict(X2)]

    #get variance and dataframe
    # variance = np.var(all_predictions, axis=0)
    # variance_df = pd.DataFrame(data=variance, columns=['Variance'])

    # assert len(variance) == len(predictions)

    #concatenate tables
    initial_data = pd.DataFrame(data=array2, columns = ['Iteration', 'nap', '2meth', 'Ligan2Metal', 'Cat', 'base', 'i_PrOH', 'Time', 'Temperature', 'concentration'])
    df = pd.concat([initial_data, predictions_df], axis=1)

    #getting a list to organize sorting
    # feat_imp_T = feat_imp.transpose()
    keys1 = ['nap', '2meth', 'Ligan2Metal', 'Cat', 'base', 'i_PrOH', 'Time', 'Temperature', 'concentration']
    keys2 = ['nap', '2meth', 'Ligan2Metal', 'Cat', 'base', 'i_PrOH', 'Time', 'Temperature', 'concentration']
    keys1.insert(9,'Prediction')
    # keys2.insert(7, 'Variance')


    #select the reaction (selecting for max everything)
    df_sorted = df.sort_values(by=[keys1[-1], keys1[0]], ascending=[False, False])
    preliminary = df_sorted.iloc[0:5]
    df_sorted2 = preliminary.sort_values(by=[keys2[-1], keys2[0]], ascending=[True, False])
    toPerform = df_sorted2.iloc[0]



    #save data
    # feat_imp.to_csv('feature_importances.txt', sep= '\t')
    best_params.to_csv('best_parameters.txt', sep= '\t')
    toPerform.to_csv('selected_reaction.txt', sep = '\t', header=False)
    df_sorted.to_csv('predictions.csv')
    filename3 = 'random_forest_model_grid.sav'
    dump(grid, filename3)

    print('Have a good one, mate!')
    print(time.strftime("%Y/%m/%d  %H:%M:%S"))## 带日期的12小时格式
