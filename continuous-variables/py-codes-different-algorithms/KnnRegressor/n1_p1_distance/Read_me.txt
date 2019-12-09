The knn algorithm uses the neighbors of a point to predict, which means it is not suitable for nano data. 
The GridSearchCV has garnered the best parameters for us, i.e. neighbor = 1, weights = distance, and choose p=1, manhattan distance.
Kfold( n_splits = 10)