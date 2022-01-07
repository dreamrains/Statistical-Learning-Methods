# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 16:09:00 2022

@author: young

特别说明：
1. 这个决策树包含了ID3和C4.5两个算法，两者之间区别仅仅为在构建树的时候判断是否大于阈值的值为信息增益还是
信息增益比而已，实现的时候注意替换相应的返回值即可。
2. 本来想把剪枝也写好，但是发现剪枝的时候我设计了两层递归，实在不太满意。同时对于剪枝的算法我没有完全理解
等到我完全理解了再把这里的剪枝算法写出来

"""
import numpy as np
import pandas as pd


class Node:
    def __init__(self, feature=None, label=None):
        self.child = {}  # 记录该节点的子树
        self.feature = feature  # 记录该节点的特征，这里选择记录特征列索引
        self.label = label  # 记录该节点所属的分类


class DecisionTree:
    def __init__(self, critical_value=0.1):
        self.critical_value = critical_value  # 特征集阈值
        self.tree = {}

    # 计算数据集的经验熵
    def cal_exp_ent(self, dataset_list):
        # dataset_list为传入的特征列表，用以计算特征与标签列的经验熵
        data_len = len(dataset_list)
        data_set = set(dataset_list)
        data_count = {}
        for i in data_set:
            data_count[i] = dataset_list.count(i)
        exp_ent = -sum([i/data_len * np.log2(i/data_len) for i in data_count.values()])
        return exp_ent

    # 计算数据集的条件熵
    def cal_cond_ent(self, dataset, axis=0):
        # axis为列索引，即第几列
        m = len(dataset)
        feature_len = {}  # 这是用于记录每个特征中样本的长度，用于求条件熵的一个值
        cond_ent = 0
        data_list = [dataset[i][axis] for i in range(m)]  # 这是用于计算当前条件熵的那一列特征列表
        feature_set = set(data_list)
        for a in feature_set:
            feature_len[a] = data_list.count(a)  # 计算特征中每一个样本的长度
            feature_list = [i for i in dataset if i[axis] == a]
            label_list = np.array(feature_list)[:, -1].tolist()
            cond_ent += (feature_len[a]/m) * self.cal_exp_ent(label_list)
        return cond_ent

    # 开始计算最优特征
    def get_best_gain(self, dataset):
        n = np.shape(dataset)[1]
        entropy = self.cal_exp_ent(dataset[:, -1].tolist())  # 先计算H(D)
        info_gain_list = [(entropy-self.cal_cond_ent(dataset, i)) for i in range(n-1)]
        # info_gain_ratio_list = [(entropy-self.cal_cond_ent(dataset, i))/self.cal_exp_ent(feature[:, i].tolist()) for i in range(n-1)]
        '''
        上面注释掉的那一行计算的是增益信息比，如果要实现C4.5算法，则把这个函数里之后排序取索引值（就下面这一行）的列表改为增益信息比即可
        '''
        best_feature = np.argmax(np.array(info_gain_list))
        # print("最优特征的列索引为{},信息增益为{}".format(best_feature, info_gain_list[best_feature]))
        return best_feature, info_gain_list[best_feature]

    # 构建树
    def build_tree(self, train_data):
        # 输入的train_data训练数据是一个dataframe，如果是输入数组需要把数据集和标签集分开，需要做一些改动，整体变化不大就直接用dataframe处理
        label_array = train_data.iloc[:, -1]
        feature_name = train_data.columns
        node = Node()
        # 1. 如果数据集中所有实例属于同一类，将该类作为类标记并返回单节点树
        if len(set(label_array)) == 1:
            node.label = label_array[0]
            return node
        # 2. 如果特征集为空集，则返回实力数最大的类作为类标记
        if len(feature_name[0:-1]) == 0:  # 这里默认分类在最后一列了，所以判断的是除分类以外的列数是否为0
            label = label_array.value_counts()
            label_result = label.sort_values(ascending=False).index[0]
            node.label = label_result
            return node
        # 3. 计算各个特征的信息增益，选择信息增益最大的特征（如果为C4.5算法，这一步则为选最大信息增益比，修改上方函数get_best_gain返回值即可）
        best_feature_index, best_info_gain = self.get_best_gain(np.array(train_data))
        best_feature_name = feature_name[best_feature_index]
        # 4. 判断上一步得到的信息增益是否大于阈值，小于则返回一个单节点树，取实例数最大的类作为类标记并返回树
        if best_info_gain < self.critical_value:
            label = label_array.value_counts()
            label_result = label.sort_values(ascending=False).index[0]
            node.label = label_result
            return node
        # 5. 如果信息增益大于阈值，将选择的特征分割为多个非空子集，并取实例数最大的类作为类标记，构建子节点，由节点及其子节点构成树，并返回树
        node.feature = best_feature_index
        feature_set = set(train_data.iloc[:, best_feature_index])
        for i in feature_set:
            sub_data_list = [j for j in train_data.itertuples(index=False) if j[best_feature_index] == i]
            sub_data = pd.DataFrame(sub_data_list)
            # 6. 数据集删除掉信息增益最大的特征集以后，使用这个处理后的数据集开始递归前5步
            sub_train_data = sub_data.drop(columns=[best_feature_name])
            node.child[i] = self.build_tree(sub_train_data)
        return node

    # 学习后生成一棵树
    def create_tree(self, train_data):
        self.tree = self.build_tree(train_data)
        return self.tree

    # 剪枝流程（还未完成），这个流程还有很多不懂的地方，先更新这么多
    def prune(self, node):
        # 先计算树中叶节点的数量，并将叶节点的分类标签存入列表
        leaf_node_label = []

        def get_leaf_node(tree_node, label):
            if tree_node.child:
                for i in tree_node.child:
                    if not tree_node.child[i].child:
                        label.append(tree_node.child[i].label)
                    else:
                        get_leaf_node(tree_node.child[i], label)
            else:
                label.append(tree_node.label)
            return label, len(label)
        label1, quantity1 = get_leaf_node(node, leaf_node_label)
        print('标签列表为{}，叶节点数量为{}'.format(label1, quantity1))
        return None

    # 预测测试数据的结果
    def predict(self, test_data, node):
        if node.feature is not None:
            for i in node.child:
                if test_data[node.feature] == i:
                    return self.predict(test_data, node.child[test_data[node.feature]])
        else:
            return node.label


if __name__ == '__main__':
    data = [
        ['青年', '否', '否', '一般', '否'],
        ['青年', '否', '否', '好', '否'],
        ['青年', '是', '否', '好', '是'],
        ['青年', '是', '是', '一般', '是'],
        ['青年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '一般', '否'],
        ['中年', '否', '否', '好', '否'],
        ['中年', '是', '是', '好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['中年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '非常好', '是'],
        ['老年', '否', '是', '好', '是'],
        ['老年', '是', '否', '好', '是'],
        ['老年', '是', '否', '非常好', '是'],
        ['老年', '否', '否', '一般', '否'],
    ]
    features = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    data_df = pd.DataFrame(data, columns=features)
    test = ['老年', '否', '是', '非常好']
    dt = DecisionTree()
    dt_node = dt.create_tree(data_df)
    test_result = dt.predict(test, dt_node)
    # leaf_node = dt.prune(dt_node)
    print('测试数据的分类结果为：', test_result)
