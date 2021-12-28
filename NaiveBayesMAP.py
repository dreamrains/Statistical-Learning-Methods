# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 20:04:07 2021

@author: young
"""
import numpy as np


class NaiveBayes:

    def __init__(self, lamda=1):
        self.p_y = {}  # P(Y)的概率
        self.p_xy = {}  # P{X|Y}的概率
        self.lamda = lamda  # lamda按照书中默认为1，为方便调试设置为变量

    # 专门计算各个概率的函数
    def cal_prob(self, cal_list, data_len):
        #  传进来的cal_data是个列表，专门用于计算事件的概率或条件概率
        #  data_len在计算先验概率时对应书中的K，条件概率时为Sj，为了方便计算这里设置为变量
        keys = set(cal_list)
        p = {}
        for i in keys:
            p[i] = (cal_list.count(i) + self.lamda) / (len(cal_list) + data_len * self.lamda)
        return p

    def train(self, train_data, train_label):
        m, n = np.shape(train_data)  # m,n分别为数据的行、列数，用于之后的循环求各个概率
        label = set(train_label)
        self.p_y = self.cal_prob(train_label, len(label))
        for y in label:
            x_list = [train_data[i] for i in range(m) if train_label[i] == y]
            #  上面这个列表是为了获取标记分类以后，对应输入X的列表，即当Y=Ck时的列表，用于之后的循环求概率
            feature = np.array(x_list)
            for j in range(n):
                feature_list = feature[:, j].tolist()
                self.p_xy[str(j) + '|' + str(y)] = self.cal_prob(feature_list, len(set(feature_list)))
        print("先验概率为：", self.p_y)
        print('各概率为：', self.p_xy)

    def predict(self, test_data):
        p_target = {}
        for y in self.p_y:
            p_target[y] = self.p_y[y]
            for i, target in enumerate(test_data):
                p_target[y] *= self.p_xy[str(i) + '|' + str(y)][str(target)]
        # 专门说一下字典中的max,不传key时按按字典中的key排序取最大，传入key以后按传入的值排序取最大
        return p_target, max(p_target, key=p_target.get)


if __name__ == "__main__":
    data_x = [
        [1, 'S'],
        [1, 'M'],
        [1, 'M'],
        [1, 'S'],
        [1, 'S'],
        [2, 'S'],
        [2, 'M'],
        [2, 'M'],
        [2, 'L'],
        [2, 'L'],
        [3, 'L'],
        [3, 'M'],
        [3, 'M'],
        [3, 'L'],
        [3, 'L'],
    ]
    data_y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    test = [2, 'S']
    naive_bayes = NaiveBayes(data_y)
    naive_bayes.train(data_x, data_y)
    p_result, label_result = naive_bayes.predict(test)
    print("测试的数据为{}，求得的分类概率为{},属于分类{}".format(test, p_result, label_result))
