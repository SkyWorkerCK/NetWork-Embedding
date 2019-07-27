# -*- coding: utf-8 -*-
# @Author : chxgong
# @E-mail : chx.gong@foxmail.com

import numpy as np
import time

import logging as log

log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s', level=log.DEBUG)

class Community2Vec(object):
    """
    class that train community embedding
    """

    def __init__(self,init_lr=0.025):
        """
·
        :param init_lr: init learning rate in SGD.  <0.025 as defult>
        :return:
        """
        self.lr=init_lr
        self.lr_min=0.0001


    def train(self,model,k,num_sampling=1e6,num_process=1):
        """
        train community embedding
        :param model:
        :param k:
        :param num_sampling:
        :param num_process:
        :return:
        """
        count, last_count, cur_sample_count = 0, 0, 0
        total_sample = np.uint32(num_sampling)

        log.info('train community embedding ------ START.')
        start_time=time.time()
        while 1:
            # judge for exit
            if count > total_sample / num_process + 2:
                break

            # adaptive learing rate
            if count - last_count > 10000:
                cur_sample_count += count - last_count
                last_count = count
                self.lr = max(self.lr_min, self.lr * (1 - cur_sample_count / (np.float32(total_sample + 1))))

            # sample a node
            node_index=np.random.randint(0,model.node_number)
            node_emb=model.node_embs[node_index]

            # train community embedding
            grad = np.zeros(node_emb.shape, dtype=np.float32)
            for com in range(k):
                diff = node_emb - model.means[com]
                m = model.pi[node_index, com] * model.inv_covariance_mat[com]
                grad += np.dot(m, diff)
            grad /= k
            # update node_emb
            model.node_embs[node_index] -= self.lr * 0.1 * grad.clip(min=-0.25,max=0.25)
            # for val in model.node_embs[node_index]:
            #     assert not np.isnan(val),'count {}. \n grad:{}\n. node_emb:{}'.format(count,grad,model.node_embs[node_index])

            # normalize node embedding
            # model.node_embs[node_index]=normalize(embedding=node_emb,batch=0)
            count+=1
        run_time=time.time()-start_time

        log.info('train community embedding ------ DONE. run time {0:.3f} s'.format(run_time))