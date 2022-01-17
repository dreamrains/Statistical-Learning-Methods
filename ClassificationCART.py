# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 13:38:51 2022

@author: young

特别说明：
这次写的二叉树打印的时候和书中不一样，但是用测试数据和之前ID3算法写出来的结果是一样的。不太懂，如果有知道的朋友欢迎交流。
"""
import numpy as np


class Node:

    def __init__(self, splitting_feature=None, data=None, label=None, left=None, right=None):
        self.feature = splitting_feature  # 切分的特征，这里记录的是索引
        self.data = data  # 切分点
        self.label = label  # 对应的标签
        self.left = left  # 左子树
        self.right = right  # 右子树


class DecisionTree:

    def __init__(self, critical_value=0.1):
        self.critical_value = critical_value  # 特征集阈值
        self.tree = {}

    # 计算特征的基尼指数
    def cal_gini(self, label):
        # label是某个特征集合中某个具体数据的标签数组，例如本程序中第一列的青年对应的标签数据，即前5行数据
        label_list = label.tolist()
        label_len = len(label_list)
        label_set = set(label_list)
        data_prob = {}
        result = 0
        for i in label_set:
            data_prob[i] = (label_list.count(i) / label_len) ** 2
            result += data_prob[i]
        return 1-result

    # 计算针对某个特征的基尼指数
    def cal_feature_gini(self, label1, label2):
        # 传入的两个标签列表分别为某个具体的特征以及不包含该特征的对应的标签数组
        gini1 = self.cal_gini(label1)
        gini2 = self.cal_gini(label2)
        data_len = len(label1) + len(label2)
        gini = len(label1) / data_len * gini1 + len(label2) / data_len * gini2
        return gini

    # 构建CART二叉树的流程
    def build_tree(self, train_data, train_label):
        node = Node()
        best_gini = float('inf')
        best_feature = 0
        best_point = ''
        if len(set(train_label)) == 1:
            node.label = train_label[0]
            return node
        if len(train_data) == 0:
            return None
        for i in range(train_data.shape[1]):
            feature_set = set(train_data[:, i])
            for j in feature_set:
                label1 = train_label[train_data[:, i] == j]
                label2 = train_label[train_data[:, i] != j]
                feature_gini = self.cal_feature_gini(label1, label2)
                if feature_gini < best_gini:
                    best_gini = feature_gini
                    best_feature = i
                    best_point = j
        x1 = train_data[train_data[:, best_feature] == best_point]
        x2 = train_data[train_data[:, best_feature] != best_point]
        y1 = train_label[train_data[:, best_feature] == best_point]
        y2 = train_label[train_data[:, best_feature] != best_point]
        node.feature = best_feature
        node.data = best_point
        node.left = self.build_tree(x1, y1)
        node.right = self.build_tree(x2, y2)
        return node

    def creat_tree(self, train_data, train_label):
        self.tree = self.build_tree(train_data, train_label)
        return self.tree

    def predict(self, node, test_data):
        if node.feature is not None:
            if test_data[node.feature] == node.data:
                return self.predict(node.left, test_data)
            else:
                return self.predict(node.right, test_data)
        else:
            return node.label


if __name__ == '__main__':
    data_x = np.array([
        ['青年', '否', '否', '一般'],
        ['青年', '否', '否', '好'],
        ['青年', '是', '否', '好'],
        ['青年', '是', '是', '一般'],
        ['青年', '否', '否', '一般'],
        ['中年', '否', '否', '一般'],
        ['中年', '否', '否', '好'],
        ['中年', '是', '是', '好'],
        ['中年', '否', '是', '非常好'],
        ['中年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '好'],
        ['老年', '是', '否', '好'],
        ['老年', '是', '否', '非常好'],
        ['老年', '否', '否', '一般'],
    ])
    data_y = np.array(['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否'])
    test = ['老年', '否', '是', '非常好']
    dt = DecisionTree()
    tree_node = dt.creat_tree(data_x, data_y)
    test_result = dt.predict(tree_node, test)
    print('测试数据的分类结果为：', test_result)
