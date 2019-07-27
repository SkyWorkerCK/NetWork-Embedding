# -*- coding: utf-8 -*-
# @Author : chxgong
# @E-mail : chx.gong@foxmail.com\

import Utils.data_utils as data_utils
import Utils.io_utils as io_utils

if __name__ == '__main__':

    file_id = 3980

    input_feat_file = '{}.feat'.format(file_id)
    input_edge_file = '{}.edges'.format(file_id)
    path=r'data\facebook\{}'.format(file_id)

    # the position of feature, including a and b
    number_of_feature = 42
    feature_pos=[[32,37],[4,5],[39,42]]


    for pos in feature_pos:
        output_flie = 'feat_layer_{}_{}'.format(pos[0],pos[1])
        len_comms=0

        # data_utils.remove_duplicate_nodes(input_file=input_com_true,output_file='layer1',path=path)
        comms=data_utils.clac_community_by_feat(input_file=input_feat_file,path=path,number_of_feature=number_of_feature,feature_pos=pos)
        for com in comms:
            len_comms+=len(com)
        # print('node number {}-{}: {}'.format(feature_pos[0],feature_pos[1],len_comms))

        io_utils.save_community(communities=comms,output_file_name=output_flie,path=path)
        data_utils.union_ground_truth_and_graph(edge_file=input_edge_file, ground_truth_file=output_flie, path=path)