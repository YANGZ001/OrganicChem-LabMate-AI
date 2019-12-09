# distance
import numpy as np
import pandas as pd
#from scipy.spatial.distance import cdist
#
#def eculiean_dist(vec1, vec2):
#    vec1_scaled = scale(vec1)
#    vec2_scaled = scale(vec2)
#    dist = np.linalg.norm((vec1_scaled-vec2_scaled), ord= 2)
#    return dist

filename = 'train_data_30.csv'
train = pd.read_csv(filename)
train = train[['nap (mmol)', '2meth (mmol)', 'Ligan2Metal', 'Cat (mol%)',
       'base (equiv.)', 'i_PrOH (equiv)', 'Time (Hrs)', 'Temperature (C)',
       'concentration (M)']] #只把list里的标签对应变量的列传入train

array = train.values #把train文件的值传给array
array_m = array.mean(axis=0) #axis=0表示沿着列，取 平均

unseen = pd.read_csv(r'C:\Users\Lenovo\Desktop\project\Experiments\RandomForest\30_data_expanded\predictions.csv')
unseen = unseen[['nap', '2meth', 'Ligan2Metal', 'Cat', 'base','i_PrOH', 'Time', 'Temperature', 'concentration']]
unseen_data = unseen.values

tmp = unseen_data-array_m  #每一个值都减去平均
dist = np.sum(np.square(tmp), axis=1) #得到一个1048575的，算了，先不管，你现在的目标是让这个函数发挥作用
args = np.argsort(dist)[::-1] #[::-1]反过来排序，从大到小
print(args[:10])#输出前10个

#
##下面的代码是要得到两两相比，距离最小的
#x = cdist(array, array).flatten()
#args = x.argsort()# 从小到大排
#args = args[::-1] #倒叙】
#args
#ma = []
#_ = 0
#for i1 in range(30):
#    for i2 in range(30):
#        ma.append([i1, i2])
#for i in range(10):
#    print(ma[args[i]])
