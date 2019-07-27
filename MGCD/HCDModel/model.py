# -*- coding: utf-8 -*-
# @Author : chxgong
# @E-mail : chx.gong@foxmail.com

import numpy as np
import time
from mpmath import math2
from os.path import join as path_join
import networkx as nx
import logging as log #日志处理模块
from sklearn import mixture
from sklearn.mixture import GMM
"""
logging基本配置
format指定文件输出的格式和内容
打印时间、日志水平、文件名、日志信息"""
log.basicConfig(format='%(asctime).19s %(levelname)s %(filename)s:  %(message)s', level=log.DEBUG)

class Model(object):
    """
    class that keep track of all the paraments used during the learning of the embedding.
    """

    def __init__(self,dim,neg_table_size=1e8,seed=2): #初始化实例
        """
        :param dim: dimension of embedding vector 向量维度
        :param neg_table_size: size of the negative table to generate 负采样表的大小
        :param seed: seed for numpy.random.seed() #随机数生成
        参数相同时使得每次生成的随机数相同；当参数不同或者无参数时，多次生成随机数且每次生成的随机数都不同。
        """
        #init param
        self.dim = dim # dimension of embedding vector
        self.neg_table_size = np.uint32(neg_table_size) #size of the negative table to generate
        self.seed = seed # seed for numpy.random.seed()
        self.node_number = 0 # number of nodes in g  导入数据的节点数
        self.edge_number = 0 # number of edges in g  导入数据的边数
        self.nodes = [] # a list that saving nodes name  e.g. ['4','2','6',...] 节点信息存储
        self.nodes_weighted = [] # a list that saving nodes name and sum of its neighbors' edges weight边权值. e.g. [['0',1.2],['1',0.8],...]
        self.edges = [] # a list that saving edges  e.g. [('0','1'),('0','2'),...] 边信息存储
        self.edges_weighted = [] # a list that saving edges and weight. e.g. [['1','2','0.5'],['2','3','1.0'],...]
        self.neg_table = None # a negative table for negative sampling. 负采样表
        self.alias = None # alias in alias table
        self.prob = None # prob in alias table 概率
        self.node_embs = None # a ndarray saves nodes' embedding 节点信息嵌入.  E.g., node_embs[0] saves nodes[0]'s embedding
        self.context_embs = None # a ndarray saves nodes' context embedding 根据上下文节点存储信息.   E.g., context_embs[0] saves nodes[0]'s context embedding
        self.g = nx.Graph() # a graph to save the input network
        self.gmm = mixture.GaussianMixture() # a Gaussian Mixture Model 高斯混合模型

        self.weights = None  # The weights of each mixture components. 混合后的成分权值
        self.means = None  # The mean of each mixture component. 混合后每个成分的均值
        self.covariance_mat = None  # The covariance matrix of GMM.  GMM的协方差
        self.inv_covariance_mat = None  # The inverse covariance matrix of GMM. GMM协方差转置矩阵
        self.pi = None # Predict posterior probability of each component given the data. 后验证概率

        if dim % 4 != 0:  #显示设置
            log.warning("consider setting layer size to a multiple of 4 for greater performance")


    def load_data(self,input_file_name,sep,weighted_graph=False,path='data'):
        """
        数据加载及预处理
        :param input_file_name: input file name 数据文件名称
        :param sep: sep in split() 划分方式
        :param weighted_graph: wether the graph is weighted or unweighted 是否具有权值，默认无
        :param path: 文件路径
        :return:
        """
        log.info('start reading data from input file.') #日志
        assert weighted_graph == False or weighted_graph == True, "param: weighted_graph should be False or True."
        start_time=time.time() #启动时间
        file_path = path_join(path,input_file_name) #文件载入
        #判断是否具有权值，分开处理数据
        if not weighted_graph:
            with open(file_path,'r') as fin:
                for line in fin.readlines():
                    cur_line = line.strip().split(sep)
                    cur_line.append('1')     #无权值在边后追加1
                    self.edges_weighted.append(cur_line)
                    self.g.add_weighted_edges_from([cur_line]) #将权加入图中

            self.node_number = self.g.number_of_nodes() #节点总数
            self.edge_number = self.g.number_of_edges() #节点的总边数
            self.nodes = list(self.g.nodes())  #存储所有节点信息
            self.edges = list(self.g.edges())  #边信息
#            print(self.edges,'\n',self.nodes)
#            print(self.node_number,self.edge_number)
#            print(self.g.nodes())

            # save node and its weight
            #统计每个节点的相邻节点的数目，并追加权值
            for node_name in self.g.nodes():
                # node_tmp = []
                # node_tmp.append(node_name)
                # node_tmp.append(np.float32(len(list(self.g.neighbors(node_name)))))
                node_tmp= [node_name, np.float32(len(list(self.g.neighbors(node_name))))]
                #统计和当前节点相邻节点数目
                self.nodes_weighted.append(node_tmp)
        #有权值时处理方式
        else:
            with open(file_path,'r') as fin:
                for line in fin.readlines():
                    cur_line = line.strip().split(sep)
                    self.edges_weighted.append(cur_line)
                    self.g.add_weighted_edges_from([cur_line])
            self.node_number=self.g.number_of_nodes()
            self.edge_number=self.g.number_of_edges()
            self.nodes = list(self.g.nodes())
            self.edges = list(self.g.edges())
            for val in self.g.nodes():
                self.nodes_weighted.append([val,0.0])
            # save node and its weight
            for u,v,w in self.edges_weighted:
                u_index=self.nodes.index(u)
                v_index=self.nodes.index(v)
                self.nodes_weighted[u_index][1] += np.float32(w)
                self.nodes_weighted[v_index][1] += np.float32(w)
        run_time=time.time()-start_time #运行时间统计
        log.info('reading data from input file ===> | %s | with %d nodes and %d edges. run time : %.3f s' % (input_file_name, self.node_number, self.edge_number,run_time))

    def init_neg_table(self): #负梯度采样表
        """
        initalize negative table for negative sampling.#初始化负采样梯度表
        :return:
        """
        log.info('start initalizing negative table.')
        start_time=time.time()
        self.neg_table = np.zeros(self.neg_table_size, dtype=np.uint32) #生成全零数组
        total_sum=cur_sum=pro=np.float32(0.0)  #取值：
        print(self.neg_table)
        #calc total_sum
        for i in range(self.node_number):
            total_sum += self.nodes_weighted[i][1]**0.75   #权重×0.75

        #初始化负采样表核心
        node_index=0 # the index of node in <type:list> self.nodes
        for i in range(self.neg_table_size): #负采样梯度表大小
            if node_index<self.node_number:#节点索引数小于节点总数
                if (i+1)/self.neg_table_size > pro:
                    # node_degree_weight = 0.0
                    # for neighbor_name in self.g.neighbors(node_name):
                    #     node_degree_weight += np.float32(self.g.get_edge_data(node_name,neighbor_name)['weight'])
                    cur_sum += self.nodes_weighted[node_index][1]**0.75
                    pro = cur_sum/total_sum
                    node_index += 1
            self.neg_table[i]=self.nodes_weighted[node_index-1][0]
        run_time=time.time()-start_time
        log.info('initalize negative table done. run time : {0:.3f} s.'.format(run_time))


    def init_alias_table(self):
        """
        initalize a alias table for alias sampling algorithm, which is used to sample an edge in O(1) time.
        :return:
        """
        start_time=time.time()
        log.info('start initalizing alias table.')
        self.alias = np.zeros(self.edge_number,dtype=np.uint32)      # alias table  存放第i列 另一个事件的标号
        self.prob = np.zeros(self.edge_number,dtype=np.float32)       # 归一化的概率 table 存放第i列 事件i占的面积百分比
        norm_prob = []    # 概率list 存第i列事件i占的面积百分比
        large_block = []  # 面积大于1的list
        small_block = []  # 面积小于1的list
        total_sum=np.float32(0)
        # cur_small_block = cur_large_block = 0
        num_small_block = num_large_block = 0 # number of small block and large block

        # calculate sum of all edges weight 统计所有边的权重和
        for (i,j,w) in self.edges_weighted:
            total_sum += np.float32(w)

        # 得到第i列中事件i本身所占的百分比每一列面积为1   总面积为1 * num_edges
        for (i,j,w) in self.edges_weighted:
            norm_prob.append(np.float32(w)*self.edge_number/total_sum)
        # 分为两组，大于1的一组，小于1的一组
        for k in range(self.edge_number):
            if norm_prob[k] < 1:
                small_block.append(k)
                num_small_block += 1
            else:
                large_block.append(k)
                num_large_block += 1

        # 直到每一列的占比都为1
        while num_small_block and num_large_block:
            num_small_block -= 1
            cur_small_block = small_block[num_small_block] # 当前小边的序号
            num_large_block -= 1
            cur_large_block = large_block[num_large_block] # 当前大边的序号
            self.prob[cur_small_block] = norm_prob[cur_small_block] #把归一化占比赋给Prob
            self.alias[cur_small_block] = cur_large_block # 用面积大于1的去填充面积小于1的  alias中存，不是事件i的序号，即用来填充的事件的序号
            norm_prob[cur_large_block] = norm_prob[cur_large_block]+norm_prob[cur_small_block]-1 # 得到large block填充的剩下的面积

            # 如果剩下的面积小于1则归到小块面积
            if norm_prob[cur_large_block]<1:
                small_block[num_small_block]=cur_large_block
                num_small_block += 1
            else:
                large_block[num_large_block]=cur_large_block
                large_block[num_large_block]=cur_large_block
                num_large_block += 1
            # print(self.prob)
        # print(self.prob)
        while num_large_block:
            num_large_block -= 1
            self.prob[large_block[num_large_block]]=1
        while num_small_block:
            num_small_block -= 1
            self.prob[small_block[num_small_block]]=1
        run_time=time.time()-start_time
        log.info('initalize alias table done. run time : {0:.3f} s.'.format(run_time))


    def init_embedding(self):#节点嵌入
        """
        initalize node embedding and context embedding.type:float32
        random init node embedding from [0,1) uniform distribution and init context embedding to zero.
        each row is a node's embedding. e.g. node_embs[0] save self.nodes[0] embedding.
        :return:
        """
        np.random.seed(seed=self.seed)
        # self.node_embs = self.xavier_normal(size=(self.node_number,self.dim), as_type=np.float32)
        # self.context_embs = self.xavier_normal(size=(self.node_number,self.dim), as_type=np.float32)
        # self.node_embs = np.random.normal(loc=0.0,scale=1.0,size=(self.node_number, self.dim)).astype(np.float32)

        #从均匀分布low到high中进行随机采样
        self.node_embs=np.random.uniform(low=-4.5,high=4.5,size=(self.node_number,self.dim)).astype(np.float32)
        #context嵌入
        self.context_embs=np.zeros(shape=(self.node_number,self.dim),dtype=np.float32)
    # def reset_embedding(self):
    #     """
    #     reset node_embs to node_embs.
    #     :return:
    #     """
    #     np.random.seed(seed=self.seed)
    #     self.node_embs=self.node_embs

    def reset_embedding_xavier(self):
        """
        #参数初始化
        :return:
        """
        np.random.seed(seed=self.seed)
        """
        Xavier初始化函数
        """
        self.node_embs=self.xavier_normal(size=(self.node_number,self.dim),as_type=np.float32)
        self.context_embs = self.xavier_normal(size=(self.node_number,self.dim), as_type=np.float32)


    def sample_an_edge(self): #边采样
        """
        sample an edge from alias table
        :return: the index of edge in g.edges()
        """
        edge_index=np.random.randint(0,self.edge_number-1)  #随机生成一个数进索引行边采样
        return edge_index if np.random.uniform(0.0,1.0) < self.prob[edge_index] else self.alias[edge_index]
        #边采样，如果采样的边概率大于随机均匀分布则输出边，否则输出别名表的值

    def sample_a_node(self,exist_node_name):
        """
        negative sampling a node fron negative table. if sample the same exist_node_name, then sampling again
        :return: the index of node in g.nodes()
        """
        node_index=np.random.randint(0,self.neg_table_size-1)
        return self.nodes.index(str(self.neg_table[node_index])) if self.neg_table[node_index]!=exist_node_name else self.sample_a_node(exist_node_name)
        #点采样

    def fit_gmm(self,k,reg_covar=1e-6,n_init=10): #高斯混合模型
        """
        Fit GMM with the current model's node embedding and save the result in self
        :param k:混合高斯模型个数
        :param reg_covar:协方差对方矩阵非负正则化，保证协方差矩阵均为正，默认为0
        :param n_init:初始化次数，用于产生最佳的初始参数
        :return:
        """
        log.info('start fitting GMM with {} components.'.format(k))
        start_time=time.time()
        self.gmm = mixture.GaussianMixture(n_components=k,reg_covar=reg_covar,n_init=n_init)
        self.gmm.fit(self.node_embs) #数据预处理，训练其固有属性，例如均值，方差
        # GaussianMixture对象执行EM(expectation-maximization)算法拟合GMM.
        # GaussianMixture.fit方法从数据学习一个GMM. 给定检验数据，
        # 使用GaussianMixture.predict方法给每个样本分派它最可能属于的高斯分布。
        for cov in self.gmm.covariances_:
            for val in np.diag(cov):
                assert val != 0,'diag of covariance matrix is 0.'
                #assert的异常参数，其实就是在断言表达式后添加字符串信息，用来解释断言并更好的知道是哪里出了问题
        # save pi, means, covariance_mat in self
        # 权重
        self.weights = self.gmm.weights_.astype(np.float32)  # The weights of each mixture components.
        # 均值
        self.means = self.gmm.means_.astype(np.float32)  # The mean of each mixture component.
        # 协方差系数
        self.covariance_mat = self.gmm.covariances_.astype(np.float32)  # The covariance matrix of GMM.
        #转置协方差矩阵
        self.inv_covariance_mat = self.gmm.precisions_.astype(np.float32)  # The inverse covariance matrix of GMM.
        #先验概率
        self.pi=self.gmm.predict_proba(self.node_embs).astype(np.float32) # Predict posterior probability of each component given the data.
        run_time=time.time()-start_time
        log.info('fitting GMM with {} components done. run time : {:.3f} s.'.format(k,run_time))


    def community_detection(self,community_number): #社区关系划分，划分社区的个数
        """
        基于高斯混合分布的社区关系划分，每一个成员list代表一个社区
        :return: a list of community detection result based on GMM. every member of the list presents a community. e.g. [['1','2','3'],['4','5'],...]
        """
        com_demo = []
        ret_commmunity = []
        for pros in self.pi: #先验概率
            com_index=int(np.where(pros == max(pros))[0])#确定其索引
            com_demo.append(com_index)

        for i in range(community_number):
            #遍历数据对象，enumerate（）函数列出数据和数据下标
            node_tmp = [self.nodes[index] for index,x in enumerate(com_demo) if x==i]
            ret_commmunity.append(node_tmp)
        return ret_commmunity


    def weaken_community(self,community, weakening_rate = 0.5):#社区关系弱化
        """
        weaken communities
        :param community:
        :param init_wr:
        :return: None
        """
        wr = weakening_rate # weaken rate弱化比例
        # adaptive weaken rate

        # weaken communities
        for com in community:
            for node_name in com:
                node_index=self.nodes.index(str(node_name))
                # label=max(self.pi[self.nodes.index(str(node_name))])
                label=self.gmm.predict(self.node_embs[node_index].reshape(1,-1))[0]
                # node_emb = self.node_embs[self.nodes.index(str(node_name))]

                diff = self.node_embs[node_index] - self.means[label]
                self.node_embs[node_index] += wr * diff
                # if self.gmm.predict_proba(self.node_embs[node_index].reshape(1,-1))[0][label]<0.05:
                #     break

    @staticmethod
    def xavier_normal(size, as_type=np.float32, gain=1):
        assert len(size) == 2
        std = gain * math2.sqrt(2.0 / sum(size))#方差
        return np.random.normal(size=size, loc=0, scale=std).astype(as_type)#采样由均值和方差识别