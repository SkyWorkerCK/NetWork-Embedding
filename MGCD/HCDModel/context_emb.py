# -*- coding: utf-8 -*-
# @Author : chxgong
# @E-mail : chx.gong@foxmail.com
# @Time   : 2018/5/4 14:51
import numpy as np
import time
from scipy.special import expit as sigmoid

import logging as log

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s:  %(message)s', level=log.DEBUG)

class Context2Vec(object):
    """
    class that train context embedding
    """

    def __init__(self,init_lr=0.025,negative=5):
        """

        :param init_lr: init learning rate in SGD.  <0.025 as defult>
        :param negative:
        :return:
        """
        self.lr=init_lr
        self.lr_min = 0.0001
        self.negative=negative


    def train(self,model,num_sampling=1e6,num_process=1):
        """
        train model's context embedding with num_sampling times sampling.
        :param model:
        :param num_sampling:
        :param num_process:
        :return:
        """
        count, last_count, cur_sample_count = 0, 0, 0
        total_sample = np.uint32(num_sampling)

        log.info('train context embedding ------ START.')
        start_time=time.time()
        while 1:
            # judge for exit
            if count > total_sample / num_process + 2:
                break

            # adaptive learing rate
            if count - last_count > 10000:
                cur_sample_count += count - last_count
                last_count = count
                self.lr = max(self.lr_min,self.lr * (1 - cur_sample_count / (np.float32(total_sample + 1))))

            # sample an edge from alias table
            cur_edge=model.edges[model.sample_an_edge()]
            source_node_index = np.uint32(model.nodes.index(cur_edge[0]))
            context_node_index = np.uint32(model.nodes.index(cur_edge[1]))
            node_emb = model.node_embs[source_node_index]
            context_emb = model.context_embs[context_node_index]

            # train context embedding
            grad1 = 1 - sigmoid(np.dot(context_emb, node_emb.T))
            tmp_node_emb = grad1 * context_emb
            tmp_context_emb = grad1 * node_emb
            # negative sampling
            for _ in range(self.negative):
                cur_context_emb = model.context_embs[
                    model.sample_a_node(exist_node_name=model.nodes[context_node_index])]
                grad2 = 0 - sigmoid(np.dot(cur_context_emb, node_emb.T))
                tmp_node_emb += grad2 * cur_context_emb
                tmp_context_emb += grad2 * node_emb
            # update node_emb and context_emb
            model.node_embs[source_node_index] += self.lr * tmp_node_emb
            model.context_embs[context_node_index] += self.lr * tmp_context_emb

            # normalize node embedding and context embedding
            # model.node_embs[source_node_index]=normalize(embedding=node_emb,batch=0)
            # model.context_embs[context_node_index]=normalize(embedding=context_emb,batch=0)
            count+=1
        run_time=time.time()-start_time
        log.info('train context embedding ------ DONE, run time {0:.3f} s'.format(run_time))