# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:31:12 2022

@author: young

创建树部分参考：https://github.com/fengdu78/lihang-code/blob/master/%E7%AC%AC05%E7%AB%A0%20%E5%86%B3%E7%AD%96%E6%A0%91/5.DecisonTree.ipynb
"""
import numpy as np


class Node:

    def __init__(self, data, c, left=None, right=None):
        self.data = data  # 切分点对应的输入值
        self.c = c  # 切分点的c，对应书中的cm，即y集合的均值
        self.left = left  # 小于切分点s的部分，这里定义为左子树
        self.right = right  # 大于切分点s的部分，这里定义为右子树


class DecisionTree:

    def __init__(self):
        self.tree = {}

    # 获取最优切分变量j与切分点s以及对应的输出值
    def build_tree(self, data_input, data_value):
        if len(data_input) == 1:
            return Node(data_input[0], np.mean(data_value))
        else:
            m, n = np.shape(data_input)
            judge_value = np.zeros((m, n))
            '''
            在书中例题中x只有一列，但是数据集的输入可能存在多列，为了用于多列输入数据集的计算所以还是多加了一个列的循环.
            如果只是想要解答课后的这道题，并不需要两个循环，只需要用如下方式往下写即可
            for i in data_input:
                left_y = data_value[data_input <= i]
                right_y = data_value[data_input > i]
            '''
            for i in range(n):
                for a in range(m):
                    judge_point = data_input[a][i]
                    left_y = data_value[data_input[:, i] <= judge_point]
                    right_y = data_value[data_input[:, i] > judge_point]
                    left_c = np.mean(left_y)
                    # 以下这么处理是因为当a为0时right_y没有元素，导致right_c为nan，会有一个异常提示，尽量不要让nan出现影响后续计算
                    if len(right_y) == 0:
                        right_c = 0
                    else:
                        right_c = np.mean(right_y)
                    left_judge = np.sum((left_y - left_c) ** 2)
                    right_judge = np.sum((right_y - right_c) ** 2)
                    judge_value[a][i] = left_judge + right_judge
            # 当np.where方法只传入条件时，返回的是对应值的索引坐标，可以获得切分的点索引
            judge_index = np.where(judge_value == np.min(judge_value))
            s = judge_index[0][0]
            j = judge_index[1][0]
            c = np.mean(data_value)
            splitting_value = data_value[s]
            left_input = data_input[data_input[:, j] <= data_input[s, j]]
            left_value = data_value[data_input[:, j] <= data_input[s, j]]
            right_input = data_input[data_input[:, j] > data_input[s, j]]
            right_value = data_value[data_input[:, j] > data_input[s, j]]
            left_child = self.build_tree(left_input, left_value)
            right_child = self.build_tree(right_input, right_value)
            node = Node(splitting_value, c, left_child, right_child)
        return node

    # 用数据生成树
    def creat_tree(self, data_input, data_value):
        self.tree = self.build_tree(data_input, data_value)
        return self.tree


if __name__ == '__main__':
    data_x = np.arange(1, 11).reshape(-1, 1)
    data_y = np.array([4.50, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00])
    dt = DecisionTree()
    tree_node = dt.creat_tree(data_x, data_y)
