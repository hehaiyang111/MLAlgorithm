"""
函数说明:KNN
time: 2020-11-2
author:hhy
"""
import pandas as pd
import numpy as np
train_set = pd.read_csv("fruit_data.txt",sep='\t')
# print(train_set)
# 制作数据集相关
label = np.array(train_set['fruit_label'])
print(label)
train = pd.concat([train_set['mass'] , train_set['width'], train_set['height'], train_set['color_score']],axis=1)
train = np.array(train)
# 标准化数据集
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
train = scalar.fit_transform(train)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# 五折交叉验证   每次取1/5当做测试集
X_train, X_test, y_train, y_test = train_test_split(train, label, train_size=0.8, random_state=1)
# distance：权重距离的倒数, uniform：统一权重   callable:用户自定义方法
knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
knn.fit(X_train, y_train)

# 预测
y_pred = knn.predict(X_test)
from sklearn import *
print("model accuracy: ",metrics.accuracy_score(y_test, y_pred))