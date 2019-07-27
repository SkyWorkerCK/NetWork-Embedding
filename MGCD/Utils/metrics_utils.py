# -*- coding: utf-8 -*-
# @Author : chxgong
# @E-mail : chx.gong@foxmail.com
"""
评价标准
"""
import numpy as np
import networkx as nx
from networkx.algorithms.community.quality import modularity
from sklearn.metrics import f1_score,normalized_mutual_info_score

def calc_nmi(com_true,com_pred):#兰德指数
    """
    :param com_true: a list that saving ground truth community. e.g. [[1,2,3],[4,5,6],[7,8],...] <type : int>
    :param com_pred: a list that saving community predicted by algorithm. e.g. [[1,2,3],[4,5,6],[7,8],...] <type : int>
    :return:
    """
    # reconstruct  list --> ndarray
    comm_list=[]
    len_true = 0
    len_pred = 0
    for com in com_true:
        len_true += len(com)
        for val in com:
            comm_list.append(val)

    for com in com_pred:
        len_pred += len(com)

    assert len_true == len_pred, "the element number of com_true should be equal to com_pred."

    y_true = np.zeros(len_true, dtype=np.int32)
    y_pred = np.zeros(len_pred, dtype=np.int32)

    index_true = 0
    for com in com_true:
        for val in com:
            y_true[comm_list.index(val)] = index_true
        index_true += 1
    index_pred = 0
    for com in com_pred:
        for val in com:
            y_pred[comm_list.index(val)] = index_pred
        index_pred += 1

    return normalized_mutual_info_score(labels_true=y_true,labels_pred=y_pred)

def calc_micro_f1(com_true,com_pred):#F1评价值
    """

    :param com_true: a list that saving ground truth community. e.g. [[1,2,3],[4,5,6],[7,8],...] <type : int>
    :param com_pred: a list that saving community predicted by algorithm. e.g. [[1,2,3],[4,5,6],[7,8],...] <type : int>
    :return:
    """
    # reorder the sort of community

    # reconstruct  list --> ndarray
    comm_list = []
    len_true = 0
    len_pred = 0
    for com in com_true:
        len_true += len(com)
        for val in com:
            comm_list.append(val)

    for com in com_pred:
        len_pred += len(com)

    assert len_true == len_pred, "the element number of com_true should be equal to com_pred."

    y_true = np.zeros(len_true, dtype=np.int32)
    y_pred = np.zeros(len_pred, dtype=np.int32)

    index_true = 0
    for com in com_true:
        for val in com:
            y_true[comm_list.index(val)] = index_true
        index_true += 1
    index_pred = 0
    for com in com_pred:
        for val in com:
            y_pred[comm_list.index(val)] = index_pred
        index_pred += 1

    return f1_score(y_true=y_true,y_pred=y_pred,average='micro')


def calc_macro_f1(com_true, com_pred):
    """

    :param com_true: a list that saving ground truth community. e.g. [[1,2,3],[4,5,6],[7,8],...] <type : int>
    :param com_pred: a list that saving community predicted by algorithm. e.g. [[1,2,3],[4,5,6],[7,8],...] <type : int>
    :return:
    """
    # reconstruct  list --> ndarray
    comm_list = []
    len_true = 0
    len_pred = 0
    for com in com_true:
        len_true += len(com)
        for val in com:
            comm_list.append(val)

    for com in com_pred:
        len_pred += len(com)

    assert len_true == len_pred, "the element number of com_true should be equal to com_pred."

    y_true = np.zeros(len_true, dtype=np.int32)
    y_pred = np.zeros(len_pred, dtype=np.int32)

    index_true = 0
    for com in com_true:
        for val in com:
            y_true[comm_list.index(val)] = index_true
        index_true += 1
    index_pred = 0
    for com in com_pred:
        for val in com:
            y_pred[comm_list.index(val)] = index_pred
        index_pred += 1

    return f1_score(y_true=y_true,y_pred=y_pred,average='macro')

def calc_modularity(G, communities):#计算模块度
    """

    :param G:
    :param communities:
    :return:
    """
    comms=[]
    for com in communities:
        comms.append(set(com))

    return modularity(G=G,communities=comms)

def calc_significance_value_unweighted(G, communities):#单个社区的值

    edges=list(G.edges())
    edges_numbre_in_communities=0
    total_edges_number=len(edges)
    for com in communities:
        for vi,vj in edges:
            if vi in com and vj in com:
                edges_numbre_in_communities+=1
    sig_value=edges_numbre_in_communities/total_edges_number

    return sig_value