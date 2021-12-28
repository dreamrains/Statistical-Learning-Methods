# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 20:13:50 2021

@author: young
"""
import numpy as np


class NaiveBayes:

    def __init__(self):
        self.p_y = {}  # P(Y)的概率
        self.p_xy = {}  # P{X|Y}的概率

    # 专门计算各个先验概率的函数
    def cal_prob(self, cal_list):
        #  传进来的cal_list是个列表，专门用于计算事件的先验概率或条件概率
        keys = set(cal_list)
        p = {}
        for i in keys:
            p[i] = cal_list.count(i) / len(cal_list)
        return p

    def train(self, train_data, train_label):
        m, n = np.shape(train_data)  # m,n分别为数据的行、列数，用于之后的循环求各个概率
        self.p_y = self.cal_prob(train_label)
        label = set(train_label)
        for y in label:
            x_list = [train_data[i] for i in range(m) if train_label[i] == y]
            #  上面这个列表是为了获取标记分类以后，对应输入X的列表，即当Y=Ck时的列表，用于之后的循环求概率
            feature = np.array(x_list)
            for j in range(n):
                feature_list = feature[:, j].tolist()
                self.p_xy[str(j) + '|' + str(y)] = self.cal_prob(feature_list)
        print("先验概率为：", self.p_y)
        print('各概率为：', self.p_xy)

    def predict(self, test_data):
        p_target = {}
        for y in self.p_y:
            p_target[y] = self.p_y[y]
            for i, target in enumerate(test_data):
                p_target[y] *= self.p_xy[str(i) + '|' + str(y)][str(target)]
        #  专门说一下字典中的max,不传key时按字典中的key排序取最大，传入key以后按传入的值排序取最大        
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
    naive_bayes = NaiveBayes()
    naive_bayes.train(data_x, data_y)
    p_result, label_result = naive_bayes.predict(test)
    print("测试的数据为{}，求得的分类概率为{},属于分类{}".format(test, p_result, label_result))
