# -*- coding: utf-8 -*-          
# @Author : chxgong
# @E-mail : chx.gong@foxmail.com

import numpy as np
from os.path import join as path_join
from os.path import dirname
from os import makedirs


def load_embeddings(input_file_name,path='data'):
    """
    load embedding from file
    :param input_file_name: input file name
    :param path:
    :return: a ndarray saving current embedding
    """
    ret = []
    with open(path_join(path, input_file_name+'.txt'), 'r') as file:
        for line in file:
            tokens = line.strip().split('\t')
            node_values = [float(val) for val in tokens[1].strip().split(' ')]
            ret.append(node_values)
    ret = np.array(ret, dtype=np.float32)
    return ret

def save_embeddings(model, output_file_name, embedding_type=None, path='data'):
    """
    save embedding in a file
    file structure = <node_name>\t<feature_1> <feature_2>......
    :param model:
    :param output_file_name:
    :param embedding_type: embedding to save. including 'node','context','community'.
    :param path:
    :return:
    """
    assert embedding_type is not None,"embedding_type cannot be None. please input right value. accept 'node','context','community'."
    full_path = path_join(path,output_file_name+'.txt')
    makedirs(dirname(full_path),exist_ok=True)

    with open(full_path,'w') as file:
        if embedding_type=='node':
            node_index=0
            for node_emb in model.node_embs:
                file.write('{}\t{}\n'.format(model.nodes[node_index],' '.join([str(val) for val in node_emb])))
                node_index+=1
        elif embedding_type=='context':
            node_index=0
            for context_emb in model.context_embs:
                file.write('{}\t{}\n'.format(model.nodes[node_index],' '.join([str(val) for val in context_emb])))
                node_index+=1
        elif  embedding_type=='community':
            node_index = 0
            for node_emb in model.node_embs:
                file.write('{}\t{}\n'.format(model.nodes[node_index], ' '.join([str(val) for val in node_emb])))
                node_index += 1
        else:
            raise ValueError("embedding_type error. please input right value. accept 'node','context','community'.")

def load_community(input_file_name,path='data'):
    """
    load community from input file.
    :param input_file_name:
    :param path:
    :return: a list saves community. e.g. [[1,2,3],[4,5,6],[7,8],...] <type : int>
    """
    ret = []
    with open(path_join(path, input_file_name), 'r') as file:
        for line in file:
            com = []
            tokens = line.strip().split(' ')
            for val in tokens:
                com.append(val)
            ret.append(com)
    return ret

def save_community(communities,output_file_name, path='data'):
    """

    :param communities:
    :param output_file_name:
    :param path:
    :return:
    """
    full_path = path_join(path,output_file_name)
    makedirs(dirname(full_path),exist_ok=True)

    with open(full_path,'w') as file:
        for com in communities:
            file.write('{}\n'.format(' '.join([str(val) for val in com])))

def load_ground_truth_community(input_file_name,sep,path='data'):
    """
    load ground truth community from dataset.
    <file struct>    node_id  community_id
    :param input_file_name:
    :param sep:
    :param path:
    :return:
    """
    com_id=[]
    with open(path_join(path, input_file_name), 'r') as file:
        for line in file:
            tokens = line.strip().split(sep)
            if tokens[1] not in com_id:
                com_id.append(tokens[1])

    ret = [[] for _ in range(len(com_id))]

    with open(path_join(path, input_file_name), 'r') as file:
        for line in file:
            tokens = line.strip().split(sep)
            ret[com_id.index(tokens[1])].append(tokens[0])
    return ret

def save_metric_result(output_file_name,layer,nmi,micro_f1,macro_f1,significance_value,path='data'):
    """

    :param output_file_name:
    :param layer:
    :param nmi:
    :param modularity_true:
    :param modularity_pred:
    :param micro_f1:
    :param macro_f1:
    :param path:
    :return:
    """
    with open(path_join(path,output_file_name)+'.txt','a') as file:
        file.write('============ Layer {} Metrics Result ============\n'.format(layer))
        file.write(' NMI : {:.4f}\n micro f1 : {:.4f}\n macro f1 : {:.4f}\n significance value : {:.4f}\n\n'.format(nmi,micro_f1,macro_f1,significance_value))


def save_weakening_rate(output_file_name,wr,path='data'):
    with open(path_join(path,output_file_name)+'.txt','a') as file:
        file.write(' Weakening Rate {:.2f}\n'.format(wr))