# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 15:52:16 2021

@author: young
"""
import numpy as np
from matplotlib import pyplot as plt


class Perceptron:

    def __init__(self, input_x, input_y,):
        self._x = input_x
        self._y = input_y
        self._lr = 1  # learning_rate学习率,和书里一样定义为1

    def sign(self, w, x, b):
        result = np.dot(w, x) + b
        return result

    def final(self, w, x, b):
        return -(w[0] * x+b)/w[1]

    def train_basic(self):  # 感知机原始形式
        data_len = self._y.shape[-1]
        w = np.zeros(np.shape(self._x)[1])  # 根据输入的数据集维度初始化w
        count = 0
        b = 0
        while True:
            learn_status = True
            for i in range(data_len):
                if self._y[i] * self.sign(w, self._x[i], b) <= 0:
                    learn_status = False
                    w = w + self._lr * self._y[i] * self._x[i]
                    b = b + self._lr * self._y[i]
                    count += 1
                    continue
            if learn_status:
                break
        print("原始形式求得w为{}，b为{}，共学习了{}次".format(w, b, count))
        return w, b

    def train_dual(self):  # 感知机对偶形式
        data_len = self._y.shape[-1]
        g_matrix = self._x.dot(self._x.T)
        # 计算Gram矩阵，这个矩阵是用于后面误分条件中常数项的内积,即alpha和y各项的乘积
        al = np.zeros(data_len)  # 和书里一样alpha初始定义为0
        count = 0
        b = 0
        while True:
            learn_status = True
            for i in range(data_len):
                w_matrix = np.sum(al * self._y * g_matrix[i])
                if self._y[i] * (w_matrix + b) <= 0:
                    learn_status = False
                    al[i] = al[i] + self._lr
                    b = b + self._lr * self._y[i]
                    count += 1
                    continue
            if learn_status:
                break
        w = (al * self._y.T).dot(self._x)
        print("对偶形式求得w为{}，b为{}，共学习了{}次".format(w, b, count))
        return w, b

    def paint(self, w, b):
        data_len = self._y.shape[-1]
        plt.figure()
        positive = []
        negative = []
        x_axis_p = []
        y_axis_p = []
        x_axis_n = []
        y_axis_n = []

        for i in range(data_len):  # 先把输入数据集的样本点按照正负分个类
            if self._y[i] > 0:
                positive.append(self._x[i])
            elif self._y[i] <= 0:
                negative.append(self._x[i])

        for i in positive:
            x_axis_p.append(i[0])
            y_axis_p.append(i[1])

        plt.scatter(x_axis_p, y_axis_p, label='positve', color='g', marker='o')

        for i in negative:
            x_axis_n.append(i[0])
            y_axis_n.append(i[1])

        plt.scatter(x_axis_n, y_axis_n, label='nagative', color='r', marker='x')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.axis([-1, 5, -1, 5])
        plt.plot(self._x, self.final(w, self._x, b))
        plt.legend(loc='best')
        plt.show()


if __name__ == '__main__':
    data_x = np.array([[3, 3], [4, 3], [1, 1]])
    data_y = np.array([1, 1, -1])
    perceptron = Perceptron(data_x, data_y)
    w_basic, b_basic = perceptron.train_basic()
    w_dual, b_dual = perceptron.train_dual()
    perceptron.paint(w_basic, b_basic)
