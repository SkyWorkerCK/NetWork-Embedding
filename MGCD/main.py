# -*- coding: utf-8 -*-
# @Author : chxgong
# @E-mail : chx.gong@foxmail.com

import logging as log

import Utils.io_utils as io_utils
from HCDModel.model import Model
from HCDModel.node_emb import Node2Vec
from HCDModel.context_emb import Context2Vec
from HCDModel.community_emb import Community2Vec
import Utils.metrics_utils as mertics
import Utils.data_utils as data_utils

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.INFO)
#日志记录
if __name__ == '__main__':

    input_file_name = '107.edges'
    output_file_name = 'result'
    path = 'data/facebook/107'

    #################################################
    gmm_k = [93,326,153]  #高斯模型的K值
    feature_pos = [[385,442],[101,220],[529,573]]
    ##################################################

    dim = 28 #嵌入维度
    neg_table_size = 1e8 #负采样表的大小
    negative = 5
    lr = 0.025 #学习率
    num_process = 1
    num_sampling = 1e5

    # weakening_rate=[0.2,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5]# weakening rate for weakening community structure
    #根据不同的弱化比例进行对比
    weakening_rate=[0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5]

    ground_truth_community = [] #已经划分的社区表
    for i, j in feature_pos:
        ground_truth_community.append('feat_layer_{}_{}'.format(i, j))

    # init model初始化模型
    model = Model(dim=dim,neg_table_size=neg_table_size)
    model.load_data(input_file_name, sep=' ', weighted_graph=False, path=path)
    model.init_embedding()
    model.init_alias_table()
    model.init_neg_table()

    # learning algorithm 学习算法
    # init node_learner, context_learner, comm_learner
    node_learner = Node2Vec(init_lr=lr)
    context_learner = Context2Vec(init_lr=lr, negative=negative)
    comm_learner = Community2Vec(init_lr=lr)

    # init train node embeddings and context embeddings
    node_learner.train(model=model, num_sampling=num_sampling, num_process=num_process)
    context_learner.train(model=model,num_sampling=num_sampling,num_process=num_process)

    # save node embeddings and context embeddings
    # 保存嵌入后的向量
    io_utils.save_embeddings(model=model,output_file_name=output_file_name+'_node_emb',embedding_type='node',path=path)
    io_utils.save_embeddings(model=model,output_file_name=output_file_name+'_context_emb',embedding_type='context',path=path)

    for wr in weakening_rate:
        io_utils.save_weakening_rate(output_file_name=output_file_name + '_metric',wr=wr,path=path)
        model.node_embs = io_utils.load_embeddings(input_file_name=output_file_name + '_node_emb',path=path)
        layer = 1
        for k in gmm_k:
            model.fit_gmm(k=k, reg_covar=1e-6, n_init=10)
            comm_learner.train(model=model, k=k, num_sampling=num_sampling, num_process=num_process)

            # fit GMM with current embeddings for community detection
            # community detection and save results
            model.fit_gmm(k=k, reg_covar=1e-6, n_init=10)
            comms = model.community_detection(community_number=k)

            com_true = io_utils.load_community(input_file_name=ground_truth_community[layer - 1], path=path)
            com_pred = data_utils.reorder_pred_and_true_community(com_pred=comms, com_true=com_true)
            io_utils.save_community(communities=com_pred, output_file_name=output_file_name + '_layer{}'.format(layer),
                                    path=path)

            nmi = mertics.calc_nmi(com_true=com_true, com_pred=com_pred)
            micro_f1 = mertics.calc_micro_f1(com_true=com_true, com_pred=com_pred)
            macro_f1 = mertics.calc_macro_f1(com_true=com_true, com_pred=com_pred)
            significance_value = mertics.calc_significance_value_unweighted(G=model.g, communities=com_true)
            io_utils.save_metric_result(output_file_name=output_file_name + '_metric', layer=layer, nmi=nmi,
                                        micro_f1=micro_f1, macro_f1=macro_f1, significance_value=significance_value,
                                        path=path)

            print('============WR {:.2f} Layer {} Metrics Result ============'.format(wr,layer))
            print(' NMI : {:.4f}'.format(nmi))
            print(' Micro F1 : {:.4f}'.format(micro_f1))
            print(' Macro F1 : {:.4f}'.format(macro_f1))
            print(' Significance Value : {:.4f}'.format(significance_value))
            print('===============================================')

            if len(gmm_k)>1:

                # weakening current community structure
                model.weaken_community(community=com_pred,weakening_rate=wr)
                # io_utils.save_embeddings(model=model, output_file_name=output_file_name + '_weaken_{}'.format(layer), embedding_type='node',path=path)

                layer+=1

    print('Community detection done.')