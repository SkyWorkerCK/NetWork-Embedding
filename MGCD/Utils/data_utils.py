# -*- coding: utf-8 -*-
# @Author : chxgong
# @E-mail : chx.gong@foxmail.com

import networkx as nx
import pandas as pd
from os.path import join as path_join

def clac_community_by_feat(input_file,path,number_of_feature,feature_pos):
    """

    :param input_file:
    :param path:
    :param number_of_feature:
    :param feature_pos:
    :return: a list saving community result
    """

    feat_start,feat_end=feature_pos # include start posation and end posation
    res_comms=[]

    columns_name = []
    for i in range(1, number_of_feature + 1):
        columns_name.append('feat{}'.format(i))
    feat_for_groupby=[]
    for i in range(feat_start,feat_end+1):
        feat_for_groupby.append('feat{}'.format(i))

    df=pd.read_table(path_join(path,input_file),sep=' ',header=None,index_col=0)

    df.index.name='node_id'
    df.columns=columns_name

    grouped=df.groupby(feat_for_groupby)
    for key in grouped.groups:
        res_comms.append(list(grouped.groups[key]))

    return res_comms

def union_ground_truth_and_graph(ground_truth_file,edge_file,path='data',sep=' '):
    """

    :param ground_truth_file:
    :param edge_file:
    :param path:
    :param sep:
    :return: new ground truth. <type : list>
    """

    # load graph in networkx.Graph()
    g = nx.Graph()  # a graph to save the input network
    with open(path_join(path, edge_file), 'r') as fin:
        for line in fin.readlines():
            cur_line = line.strip().split(sep)
            g.add_edges_from([cur_line])
    nodes=set(g.nodes())

    # load ground truth
    ground_truth=[]
    ground_truth_tmp = []
    with open(path_join(path,ground_truth_file),'r') as fin:
        for line in fin.readlines():
            cur_line = line.strip().split(sep)
            ground_truth_tmp.append(cur_line)

    for com in ground_truth_tmp:
        for val in com[:]:
            if val not in nodes:
                com.remove(val)

    # reorder ground truth by the number of nodes in every communities
    for _ in range(len(ground_truth_tmp)):
        number_of_nodes_in_com = []
        for com in ground_truth_tmp:
            number_of_nodes_in_com.append(len(com))
        index=number_of_nodes_in_com.index(max(number_of_nodes_in_com))

        ground_truth.append(ground_truth_tmp[index])
        ground_truth_tmp.remove(ground_truth_tmp[index])


    ground_truth_1d_list=[]
    for com in ground_truth:
        ground_truth_1d_list.extend(com)

    assert nodes==set(ground_truth_1d_list),"erroe! not equal!"

    # saving new ground truth
    with open(path_join(path,ground_truth_file),'w') as fout:
        for com in ground_truth:
            if len(com)>0:
                fout.write('{}\n'.format(' '.join([str(val) for val in com])))

def reorder_pred_and_true_community(com_pred, com_true):
    """

    :param com_pred:
    :param com_true:
    :return:
    """
    res_com_pred = []
    for sub_com_true in com_true:
        number_of_intersection = []
        for sub_com_pred in com_pred:
            number_of_intersection.append(len(set(sub_com_pred) & set(sub_com_true)))

        index = number_of_intersection.index(max(number_of_intersection))
        res_com_pred.append(com_pred[index])
        com_pred.remove(com_pred[index])


    return res_com_pred