# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 17:21:02 2021

@author: young

"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Knn:

    def __init__(self):
        pass

    def load_data(self):
        # 这个数据集前两列数据，第二类和第三类的花会混在一起，所以这里选用第一列和第三列数据
        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
        data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']
        x = data.iloc[0:150, [0, 2]].values
        y = data.iloc[0:150, 4].values
        y[y == 'Iris-setosa'] = 0
        y[y == 'Iris-versicolor'] = 1
        y[y == 'Iris-virginica'] = 2
        plt.scatter(data['sepal length'][0:50], data['petal length'][0:50], color='red', marker='o', label='setosa-0')
        plt.scatter(data['sepal length'][50:100], data['petal length'][50:100], color='blue', marker='^',
                    label='versicolor-1')
        plt.scatter(data['sepal length'][100:150], data['petal length'][100:150], color='green', marker='s',
                    label='virginica-2')
        plt.xlabel('sepal length')
        plt.ylabel('petal length')
        return x, y

    def cal_distance(self, x1, x2):
        # 这里直接用欧氏距离，如果使用其他距离改成相应的计算方式即可
        distance = np.linalg.norm(x1 - x2)
        return distance

    def train(self, train_x, train_y, test_data, k=1):
        # 把k设置为变量，便于之后验证时测试不同数值以确认是否过拟合或不足
        data_len = train_x.shape[0]  # 算一下输入的训练数据长度
        distance_list = []  # 初始一个记录各数据之间长度的列表
        for i in range(data_len):
            dis = self.cal_distance(train_x[i], test_data)
            distance_list.append(dis)
        k_list = np.argsort(distance_list)[:k]
        # 把预测数据对应位置的学习数据中分类标签的索引找到，对归属分类进行记录
        label_list = train_y[k_list]
        result = np.argmax(np.bincount(label_list.tolist()))
        '''
        这里需要注意一下numpy中bincount这个方法，这个方法只能计算int类型且数值为正
        数的数据，所以用字符串或者负数来标记数据的时候需要注意
        '''
        return result


if __name__ == '__main__':
    test = np.array([6.0, 3.2])
    knn = Knn()
    plt.figure()
    data_x, data_y = knn.load_data()
    group_result = knn.train(data_x, data_y, test)
    print("测试的数据属于分类{}".format(group_result))
    plt.plot(test[0], test[1], color='k', label='test_point', marker='o')
    plt.legend(loc='best')
    plt.show()
