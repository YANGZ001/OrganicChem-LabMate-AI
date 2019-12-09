import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import time
print('Time starts at: ', time.strftime("%Y/%m/%d  %H:%M:%S"))## 带日期的12小时格式
print('\n')
# pwd
# cd ..
# cd greedy/
# cd Desktop/project/experiments/greedy

#load data
filename = 'train_data_22.csv'
train = pd.read_csv(filename,engine='python')
# train.shape
array = train.values
X = array[:,1:-1] #从第一列到倒数第二列
# X.shape
y = array[:,-1]
# Y
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1234, train_size=0.625)#0.625 ==10/16

#General stuff
seed = 1234
nMAE_lst = list()
for i in range(2,11):
    # i=10
    print('KFoldCV = ', i)
    print('\n')
    kfold = KFold(n_splits = i, random_state = seed)
    scoring = 'neg_mean_absolute_error'
    model = RandomForestRegressor(random_state=seed)

    #Parameters to tune
    estimators = np.linspace(10, 1050, 100,dtype='int') #100个从10-1050的数
    # estimators
    estimators_int = np.ndarray.tolist(estimators)
    max_depth = np.linspace(1, 5, 5, dtype='int') #从1-10的10个数
    # max_depth
    max_depth_lst = np.ndarray.tolist(max_depth)
    max_depth_lst.append(None)
    param_grid = {'n_estimators':estimators_int, 'max_features':('auto', 'sqrt'), 'max_depth':max_depth_lst}

    #search best parameters and train
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold, n_jobs=6)
    grid_result = grid.fit(train_X, train_y)

    #print the best data cranked out from the grid search
    # np.savetxt('best_score', ["best_score: %s" % grid.best_score_, ], fmt ='%s')
    print('best_score:', grid.best_score_) #scoring using neg_mean_absolute_error
    print('\n')
    best_params = pd.DataFrame([grid.best_params_], columns=grid.best_params_.keys())
    # best_params.to_csv('best_parameters.txt', sep= '\t')
    # np.savetxt('Records', ["best_params: %s" % grid.best_params_, ], fmt ='%s')
    print('best_params:', grid.best_params_)
    print('\n')
    #Predict the future

    # filename2 = 'all_combos.csv'
    # df_all_combos = pd.read_csv(filename2)
    # df_all_combos.head()
    # df_all_combos.shape #注意，这里新导入的数据，第一列是索引，没有用
    # df_train_corrected = train.iloc[:,:-1]
    # df_train_corrected.shape
    # unseen = pd.concat([df_all_combos, df_train_corrected], sort= False).drop_duplicates(keep=False)
    # unseen.shaped
    # array2 = df_all_combos.values
    # X2 = array2[:,1:]
    # X2.shape
    model2 = grid.best_estimator_
    # model2 = RandomForestRegressor(n_estimators = grid.best_params_['n_estimators'], max_features = grid.best_params_['max_features'], max_depth = grid.best_params_['max_depth'], random_state = seed)
    RF_fit = model2.fit(train_X, train_y)
    predictions_y = model2.predict(test_X) #对每个test_X，模型有一个predictions
    predictions_df = pd.DataFrame(data=predictions_y, columns=['Prediction']) #model prediction

    feat_imp = pd.DataFrame(model2.feature_importances_, index=['nap', '2meth', 'Ligan2Metal', 'Cat', 'base', 'i_PrOH', 'Time', 'Temperature', 'concentration'], columns=['Feature_importances'])
    # feat_imp
    feat_imp = feat_imp.sort_values(by=['Feature_importances'], ascending = False)
    print('Feature_importances:', feat_imp)#特征重要性表格，按重要性排序
    print('\n')
    # feat_imp.to_csv('feature_importances.txt', sep= '\t')

    #get individual tree preds
    all_predictions = []
    for e in model2.estimators_:
        all_predictions += [e.predict(test_X)]

    #get variance and dataframe
    variance = np.var(all_predictions, axis=0)
    variance_df = pd.DataFrame(data=variance, columns=['Variance']) #variance from every estimator

    assert len(variance) == len(predictions_y)

    #concatenate tables
    initial_data = pd.DataFrame(data=test_X, columns = ['nap', '2meth', 'Ligan2Metal', 'Cat', 'base', 'i_PrOH', 'Time', 'Temperature', 'concentration'])
    # initial_data
    Neg_MAE = -mean_absolute_error(test_y, predictions_y)
    nMAE_lst.append(Neg_MAE)
    print("Neg_MAE:", Neg_MAE)
    print('\n')
    with open('Records_Neg_MAE.txt', 'a') as f:
        content = 'KFoldCV = '+str(i)+'\n'+'Neg_MAE: '+str(Neg_MAE)+'\n'
        f.write(content)
        f.write('***********\n')
        f.close()
    df = pd.concat([initial_data, predictions_df, variance_df], axis=1)
    print("initial_data+predictions+variance:", df)
    print('\n')
    # #getting a list to organize sorting
    # feat_imp_T = feat_imp.transpose()
    # # feat_imp_T
    # keys1 = list(feat_imp_T.keys())
    # # keys1
    # keys2 = list(feat_imp_T.keys())
    # keys1.insert(9,'Prediction')
    # keys1
    # keys2.insert(9, 'Variance')
    # keys1[-1]

    # #select the reaction (selecting for max everything)
    # df_sorted = df.sort_values(by=[keys1[-1], keys1[0]], ascending=[False, False])
    # # df_sorted
    # preliminary = df_sorted.iloc[0:5]
    # df_sorted2 = preliminary.sort_values(by=[keys2[-1], keys2[0]], ascending=[True, False])
    # toPerform = df_sorted2.iloc[0]



    # #save data
    # feat_imp.to_csv('feature_importances.txt', sep= '\t')
    # best_params.to_csv('best_parameters.txt', sep= '\t')
    # toPerform.to_csv('selected_reaction.txt', sep = '\t', header=False)
    # df_sorted.to_csv('predictions.csv')
    # filename3 = 'random_forest_model_grid.sav'
    # dump(grid, filename3)

    print('You are awesome! Man!')
    print('\n')
    print(time.strftime("%Y/%m/%d  %H:%M:%S"))## 带日期的12小时格式
    print('\n')
plt.bar(range(len(nMAE_lst)), nMAE_lst)
plt.title('Learning bar of neg_mean_absolute_error')
plt.savefig('Learning bar of neg_mean_absolute_error')
