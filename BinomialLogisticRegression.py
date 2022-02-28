# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 17:09:20 2022

@author: duguy

特别说明：
这里实现的是二项逻辑斯蒂回归模型算法，输入的数据也是二维的
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, learning_rate=0.01, threshold=1e-4, max_iter=200):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.threshold = threshold  # 梯度函数的阈值，用于判断循环是否结束
        self.w = None

    def load_data(self):
        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
        data.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']
        x = data.iloc[0:100, :2].values
        y = data.iloc[0:100, -1].values
        y[y == 'Iris-setosa'] = 0
        y[y == 'Iris-versicolor'] = 1
        plt.scatter(data['sepal length'][0:50], data['sepal width'][0:50], color='red', marker='o', label='setosa-0')
        plt.scatter(data['sepal length'][50:100], data['sepal width'][50:100], color='blue', marker='o',
                    label='versicolor-1')
        plt.xlabel('sepal length')
        plt.ylabel('petal length')
        return x, y

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def final(self, x):
        return -(self.w[2] + self.w[0] * x) / self.w[1]

    def fit(self, data_input, data_label):
        data_input = np.array(data_input)
        m = data_input.shape[0]
        # 先在data_input即输入的X加上一列，按照教材第92页描述，增加的数值为1
        data_input = np.column_stack((data_input, np.ones((m, 1))))
        n = data_input.shape[1]
        self.w = np.zeros(n)  # 初始化w(权重)，和X一样多一列，多出来的即为b，初始为0
        k = 0
        for i in range(self.max_iter):
            for j in range(data_input.shape[0]):
                grad_var = np.dot(self.w, data_input[j])
                '''
                这里简单介绍一下梯度函数的计算方式。在教材93页的模型参数估计中，我们得到了L(w)这个对数似然函数。然后这个对数似然函数对w求偏导。
                '''
                grad = data_input[j] * (data_label[j] - self.sigmoid(grad_var))
                grad_norm = np.linalg.norm(grad)
                # 按照梯度下降法第三步，当梯度函数L2范数小于阈值时，停止迭代，否则则更新权重w
                if grad_norm < self.threshold:
                    print('梯度小于阈值，w与k结果是:', self.w, k)
                    return self.w, k
                else:
                    self.w += self.lr * grad
                    k += 1
        print('w与k的结果是：', self.w, k)
        return self.w, k

    def predict(self, test_data):
        test_data.append(1)
        p = self.sigmoid(np.dot(self.w, test_data))
        if p >= 0.5:
            p = 1
        else:
            p = 0
        return p


if __name__ == '__main__':
    logistic = LogisticRegression()
    plt.figure()
    data_x, data_y = logistic.load_data()
    test = [5.3, 2.5]
    result_w, result_k = logistic.fit(data_x, data_y)
    test_result = logistic.predict(test)
    print('测试数据的分类结果为：', test_result)
    x_points = np.arange(4, 8)
    model_y = logistic.final(x_points)
    plt.plot(test[0], test[1], color='k', label='test_point', marker='o')
    plt.plot(x_points, model_y)
    plt.legend(loc='best')
    plt.show()
