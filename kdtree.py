# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 16:21:22 2021

@author: young
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Node:

    def __init__(self, split_column, data, data_label, left=None, right=None):
        self.col = split_column  # 记录一下分割点的维度
        self.data = data
        self.label = data_label
        self.left = left
        self.right = right


class KdTree:

    def __init__(self):
        self.nearest = None  # 初始化一个最近点
        self.category = None  # 最终结果的分类
        self.distance_nearest = 0

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

    # 计算点之间距离使用欧氏距离，如果使用其他距离改成相应的计算方式即可
    def cal_distance(self, x1, x2):
        distance = np.linalg.norm(x1 - x2)
        return distance

    # 构造kd树
    def build_tree(self, data_input, data_label, depth=0):
        if not len(data_input):
            return None
        m, n = np.shape(data_input)  # m,n分别为行、列数，分别用于求中位数切分点索引以及计算分层
        split_axis = depth % n
        median_index = np.argpartition(data_input[:, split_axis], m // 2, axis=0)[m // 2]
        # python中的//整除是向下取整的
        split_point = data_input[median_index, split_axis]
        node_x = data_input[data_input[:, split_axis] == split_point]
        node_y = data_label[data_input[:, split_axis] == split_point]
        left_x = data_input[data_input[:, split_axis] < split_point]
        left_y = data_label[data_input[:, split_axis] < split_point]
        right_x = data_input[data_input[:, split_axis] > split_point]
        right_y = data_label[data_input[:, split_axis] > split_point]
        tree_node = Node(split_axis, node_x, node_y,)
        tree_node.left = self.build_tree(left_x, left_y, depth + 1)
        tree_node.right = self.build_tree(right_x, right_y, depth + 1)
        return tree_node

    # 搜索KD树
    def search_tree(self, node, data_predict, depth=0):
        return self.search(node, data_predict, depth)

    # 搜索KD树的过程
    def search(self, node, data_predict, depth=0):
        if node is not None:
            n = len(data_predict)  # n为列数
            split_axis = depth % n
            if data_predict[split_axis] < node.data[0][split_axis]:
                self.search(node.left, data_predict, depth + 1)
            elif data_predict[split_axis] > node.data[0][split_axis]:
                self.search(node.right, data_predict, depth + 1)
            # 计算一下需要预测的数据与节点的欧式距离，用于之后的判断
            distance = self.cal_distance(data_predict, node.data)
            # 下面是根据递归查询的结果判断后将最近点的数据、标签分类和距离更新
            if distance < self.distance_nearest or self.nearest is None:
                self.nearest = node.data
                self.category = node.label
                self.distance_nearest = distance
            if (abs(data_predict[split_axis] - node.data[0][split_axis])) <= self.distance_nearest:
                if data_predict[split_axis] < node.data[0][split_axis]:
                    self.search(node.right, data_predict, depth + 1)
                else:
                    self.search(node.left, data_predict, depth + 1)
        return self.nearest, self.category


if __name__ == '__main__':
    # data_x = np.array([[2, 3],
    #                     [5, 4],
    #                     [9, 6],
    #                     [4, 7],
    #                     [8, 1],
    #                     [7, 2]])
    # data_y = np.array([1, 2, 3, 4, 5, 6])
    kdtree = KdTree()
    plt.figure()
    data_x, data_y = kdtree.load_data()
    test_x = np.array([7, 6])
    kd_node = kdtree.build_tree(data_x, data_y)
    result, category = kdtree.search_tree(kd_node, test_x)
    print('最接近的点是{}，属于分类{}'.format(result, category))
    plt.plot(test_x[0], test_x[1], color='k', label='test_point', marker='o')
    plt.legend(loc='best')
    plt.show()
