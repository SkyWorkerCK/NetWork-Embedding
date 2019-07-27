# -*- coding: utf-8 -*-
# @Author : chxgong
# @E-mail : chx.gong@foxmail.com
import numpy as np
import time
from scipy.special import expit as sigmoid

import logging as log

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s:  %(message)s', level=log.DEBUG)

class Node2Vec(object):
    """
    class that train node embedding
    """
    def __init__(self,init_lr=0.025):
        """

        :param init_lr: init learning rate in SGD.  <0.025 as defult>
        :return:
        """
        self.lr=init_lr
        self.lr_min = 0.0001

    def train(self,model,num_sampling=1e6,num_process=1):
        """
        train model's node embedding with num_sampling times sampling.
        :param model:
        :param num_sampling: number of sampling.  <1e6 as defult> 采样数量
        :param num_process: number of threads.  <1 as defult>
        :return:
        """
        log.info('train node embedding ------ START.')
        count, last_count, cur_sample_count = 0, 0, 0
        total_sample = np.uint32(num_sampling)

        start_time=time.time()
        #节点嵌入
        while 1:
            # judge for exit
            if count>total_sample/num_process+2:
                break

            # adaptive learing rate
            if count-last_count>10000:
                cur_sample_count += count - last_count
                last_count=count
                self.lr=max(self.lr_min, self.lr*(1-cur_sample_count/(np.float32(total_sample+1))))

            # sample an edge from alias table
            cur_edge=model.edges[model.sample_an_edge()]
            source_node_index = np.uint32(model.nodes.index(cur_edge[0]))
            target_node_index = np.uint32(model.nodes.index(cur_edge[1]))
            source_node_emb = model.node_embs[source_node_index]
            target_node_emb = model.node_embs[target_node_index]

            # train node embeddings
            grad = 1 - sigmoid(np.dot(target_node_emb, source_node_emb.T))
            # update source_node_emb and target_node_emb
            model.node_embs[source_node_index] += self.lr * grad * target_node_emb
            model.node_embs[target_node_index] += self.lr * grad * source_node_emb
            # normalize node embeddings
            # model.node_embs[source_node_index]=normalize(embedding=source_node_emb,batch=0)
            # model.node_embs[target_node_index]=normalize(embedding=target_node_emb,batch=0)
            count+=1
        run_time=time.time()-start_time

        log.info('train node embedding ------ DONE. run time {0:.3f} s'.format(run_time))
