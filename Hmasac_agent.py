import torch
import math
import os
import pickle
from agent import Agent
import random
import psutil
import os
import time
import numpy as np
import torch.nn as nn
# from torch import _linalg_utils
import torch.nn.functional as F

"""
Model for HMASAC
"""
# import keras
# from keras import backend as K
# from keras.optimizers import Adam, RMSprop
# import tensorflow as tf
# from keras.layers import Dense, Dropout, Conv2D, Input, Lambda, Flatten, TimeDistributed, merge
# from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
# from keras.models import Model, model_from_json, load_model
# from keras.layers.core import Activation
# from keras.utils import np_utils,to_categorical
# from keras.engine.topology import Layer
# from keras.callbacks import EarlyStopping, TensorBoard

# 需要将torch的网络全部修改成keras版本的，或者直接下载torch版本。


# 超图部分
class Predictor(nn.Module):

    def __init__(self, in_dim, n_cats):
        super(Predictor, self).__init__()
        self.predict = nn.Linear(in_dim, n_cats)
        torch.nn.init.xavier_uniform_(self.predict.weight)
        self.sigma = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, node_emb):
        n_cat = self.predict(node_emb)
        return n_cat

# 这个部分肯定需要保留的
class Readout(nn.Module):

    def __init__(self, in_dim, method="mean"):
        super(Readout, self).__init__()
        self.method = method
        self.linear = nn.Linear(2*in_dim, in_dim)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, node_fea, raw):
        if self.method == "mean":
            return self.linear(torch.cat([node_fea, raw],dim=-1))  # 这里是dim(?,36,64) -> dim(?,36,32)


# 空间超边，返回构建空间超边的损失已经主节点与各个节点之间构建超边的系数矩阵
class SpatialHyperedge(nn.Module):
    # num_node是需要根据路网结构来进行更改的。
    # TODO:这里到时候还得改路口数量
    def __init__(self,in_dim,l2_lamda = 0.001,recons_lamda = 0.2,num_node = 196,lb_th = 0):
        super(SpatialHyperedge, self).__init__()
        self.l2_lamda = l2_lamda
        self.fea_dim = in_dim
        self.recons_lamda = recons_lamda
        # projection matrix
        self.num_node = num_node
        self.r_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim,in_dim)))  # 维度是(32,32)
        # reconstruction linear combination
        self.incidence_m = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(num_node, num_node-1)))  # 维度是(36,35)
        self.lb_th = lb_th  # 值为0 TODO:阈值的设置

    def forward(self, X):
        # X的维度是(?-1,36,32)，得注意这里传进来的X的维度
        self.incidence_all = torch.zeros(self.num_node, self.num_node)  # 维度是(36,36)
        self_node = torch.eye(self.num_node)
        self.recon_loss = 0
        batch = len(X)
        X = X.detach()  # 除去梯度
        for node_idx in range(self.num_node):
            master_node_fea = X[:,node_idx,:]  # 维度是32，主节点的特征,TODO:审核这里的维度是否是(?-1,32)
            master_node_proj_fea = torch.matmul(master_node_fea, self.r_proj).reshape(-1, self.fea_dim)  # master_node_proj_fea.shape(?-1,32),表示当前主节点的特征乘对应的权重矩阵
            slave_node_idx = [i for i in range(self.num_node) if i != node_idx]  # 除去master_node外的其他35个节点的序号,dim(35)
            node_linear_comb = self.incidence_m[node_idx].unsqueeze(0)  # 维度变成(1,35)，也就是当前这个节点和其他九个结点之间的
            node_linear_comb = torch.clamp(node_linear_comb, min=self.lb_th)  # 让最小值为0

            node_linear_comb_mask = node_linear_comb > self.lb_th  # 大于lb_th的位置设置为true dim(1,35)
            node_linear_comb = node_linear_comb.masked_fill(~node_linear_comb_mask, value=torch.tensor(0))  # 设置为0，跟上面的到comb的操作结果一致
            neigh_recon_fea = torch.matmul(node_linear_comb, X[:,slave_node_idx,:]).reshape(batch,-1)  # 得到的结果是(1,35)*(?-1,35,32) = (?-1,32)，得到的是每个slave节点的特征向量乘以slave节点系数，然后35个结点求和
            self.incidence_all[node_idx][slave_node_idx] = node_linear_comb  # 将主节点与剩下的结点之间的系数存到all里面
            
            linear_comb_l1 = torch.max(torch.sum(torch.abs(node_linear_comb),dim=0))

            # linear_comb_l1 = torch.linalg.norm(node_linear_comb, ord=1)  # 第一范式
            linear_comb_l2 = torch.sqrt(torch.sum(node_linear_comb.pow(2)))

            # linear_comb_l2 = torch.linalg.norm(node_linear_comb, ord=2)  # 第二范式
            recon_error = torch.sqrt(torch.sum((master_node_proj_fea - neigh_recon_fea).pow(2),dim=-1)).reshape(batch,-1) # dim(?-1,1)
            # recon_error = torch.linalg.norm(master_node_proj_fea - neigh_recon_fea, ord=2)  # (1,20)-(1,20)，计算主节点特征和其他九个结点特征和的第二范式
                # recon_error = torch.cdist(master_node_proj_fea,neigh_recon_fea,2)
                # linear_comb_l1 = torch.linalg.norm(self.incidence_m[neigh_recon_idx],1)
                # linear_comb_l2 = torch.linalg.norm(self.incidence_m[neigh_recon_idx],2)
            node_recons_loss = recon_error * self.recons_lamda + linear_comb_l1 + self.l2_lamda * linear_comb_l2  # dim(?-1,1)
                # node_recons_loss = recon_error.squeeze()
            self.recon_loss += node_recons_loss
        self.incidence_all = self.incidence_all + self_node  # 这一步相当于是主节点对主节点的系数为1，主节点对其他九个结点的系数则是node_linear_comb。
        return self.recon_loss, self.incidence_all  # 一个样本点构建空间超边的损失dim(?-1, 1)、该样本的节点之间系数dim(36,36)


# 时间超边
class TemporalHyperedge(nn.Module):

    def __init__(self,in_dim,l2_lamda = 0.001,recons_lamda = 0.2,num_node = 196,lb_th = 0):
        super(TemporalHyperedge, self).__init__()
        self.l2_lamda = l2_lamda
        self.recons_lamda = recons_lamda
        # projection matrix
        self.num_node = num_node
        self.r_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim, in_dim)))  # dim(32,32)
        # reconstruction linear combination
        self.incidence_m = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(num_node,num_node)))  # dim(36,36)
        self.lb_th = lb_th

    def forward(self, cur, pre):
        # pre/cur.shape(?-1,36,32)
        # self.incidence_all = torch.zeros(self.num_node, self.num_node)
        self_node = torch.eye(self.num_node)
        self.recon_loss = 0
        batch = len(cur)
        cur = cur.detach()
        pre = pre.detach()
        for node_idx in range(self.num_node):
            master_node_fea = cur[:,node_idx,:]  # dim(?-1,32)  TODO:记得核实一下
            master_node_proj_fea = torch.matmul(master_node_fea, self.r_proj).reshape(batch, -1)  # dim(?-1,32)
            slave_node_idx = [i for i in range(self.num_node)]  # dim(36)
            node_linear_comb = self.incidence_m[node_idx].unsqueeze(0)  # dim(1,36)
            node_linear_comb = torch.clamp(node_linear_comb, min=self.lb_th)

            node_linear_comb_mask = node_linear_comb > self.lb_th
            node_linear_comb = node_linear_comb.masked_fill(~node_linear_comb_mask, value=torch.tensor(0))
            neigh_recon_fea = torch.matmul(node_linear_comb, pre).reshape(batch,-1)  # dim(1,36)*dim(?-1,36,32) -->dim(?-1,32)
            # self.incidence_all[node_idx][slave_node_idx] = node_linear_comb
            # 改写
            linear_comb_l1 = torch.max(torch.sum(torch.abs(node_linear_comb), dim=0))
            # linear_comb_l1 = torch.linalg.norm(node_linear_comb, ord=1)

            linear_comb_l2 = torch.sqrt(torch.sum(node_linear_comb.pow(2)))
            # linear_comb_l2 = torch.linalg.norm(node_linear_comb, ord=2)

            recon_error = torch.sqrt(torch.sum((master_node_proj_fea - neigh_recon_fea).pow(2),dim=-1)).reshape(batch,-1)  # dim(?-1,1)
            # recon_error = torch.linalg.norm(master_node_proj_fea - neigh_recon_fea, ord=2)
                # recon_error = torch.cdist(master_node_proj_fea,neigh_recon_fea,2)
                # linear_comb_l1 = torch.linalg.norm(self.incidence_m[neigh_recon_idx],1)
                # linear_comb_l2 = torch.linalg.norm(self.incidence_m[neigh_recon_idx],2)
            node_recons_loss = recon_error * self.recons_lamda + linear_comb_l1 + self.l2_lamda * linear_comb_l2
                # node_recons_loss = recon_error.squeeze()
            self.recon_loss += node_recons_loss
        # self.incidence_all = self.incidence_all + self_node
        return self.recon_loss, self.incidence_m  # dim(?-1,1) dim(36,36)


# 返回空间超边的特征向量
class SpatialHyperedgeMP(nn.Module):

    def __init__(self):
        super(SpatialHyperedgeMP, self).__init__()

    def forward(self,cur,incidence_m):
        # cur.shape(?-1,36,32),incidence_m.shape(?-1,36,32)
        batch = len(cur)
        edge_fea = torch.matmul(incidence_m, cur)  # (36,36),(?-1,36,32)-->(?-1,36,32)，是36条超边的特征向量，每条超边其维度和顶点的维度一样
        edge_degree = torch.sum(incidence_m, dim=-1).reshape(-1, 1)  # 维度是(?-1,36,1)，表示是以这10个节点构建超边时的生成的超边权重
        edge_fea_normed = torch.div(edge_fea, edge_degree)
        return edge_fea_normed  # 返回超边特征向量的标准值dim(?-1,36,32)


# 返回时间超边的特征向量
class TemporalHyperedgeMP(nn.Module):

    def __init__(self):
        super(TemporalHyperedgeMP, self).__init__()

    def forward(self, cur, pre, incidence_m):
        # # cur.shape(?-1,36,32),incidence_m.shape(36,36)
        edge_fea = torch.matmul(incidence_m, pre) + cur  # dim(?-1,36,32)
        self_degree = torch.ones(incidence_m.shape[0], 1)  # 维度是(36,1)
        edge_degree = torch.sum(incidence_m, dim=1).reshape(-1, 1) + self_degree
        edge_fea_normed = torch.div(edge_fea, edge_degree)  # dim(?-1,36,32)
        return edge_fea_normed  # 返回的超边的特征向量（规范化）dim(?-1,36,32)


# 将时间超边和空间超边的特征汇聚到对应节点中。
class HHNodeMP(nn.Module):

    def __init__(self, in_dim = 32, num_node = 196, drop_rate = 0.3):
        super(HHNodeMP, self).__init__()
        self.node_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim, in_dim)))  # dim(32,32)
        self.spatial_edge_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim, in_dim)))
        self.temporal_edge_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim, in_dim)))
        self.num_node = num_node
        self.act_1 = nn.Softmax(dim = 0)
        self.in_dim = in_dim
        self.drop = nn.Dropout(drop_rate)
        self.act = nn.ReLU(inplace=True)
        self.theta = nn.Linear(in_dim, in_dim, bias=True)
        torch.nn.init.xavier_uniform_(self.theta.weight)

    def forward(self, cur, spatial_hyperedge_emb, temporal_hyperedge_emb):
        # cur：dim(?-1,36,32),emb:(?-1,36,32)
        rlt = []
        batch = len(cur)
        for node_idx in range(self.num_node):
            node_fea = cur[:,node_idx,:]  # 获得当前节点的特征向量dim(?-1,32)
            node_fea = (torch.matmul(node_fea, self.node_proj)).reshape(batch,-1,1)  # 乘上当前节点的权重矩阵dim(?-1,32,1)

            spatial_hyperedge_fea = spatial_hyperedge_emb[:,node_idx,:]  # dim(?-1,32)
            temporal_hyperedge_fea = temporal_hyperedge_emb[:,node_idx,:]  # 获得以当前节点的构建空间超边和时间超边向量特征，dim(?-1,32)
            spatial_hyperedge_fea = torch.matmul(spatial_hyperedge_fea, self.spatial_edge_proj)  # dim(?-1,32)
            temporal_hyperedge_fea = torch.matmul(temporal_hyperedge_fea, self.temporal_edge_proj)  # 分别乘上超边对应的权重矩阵dim(?-1,32)

            # hyperedge = torch.vstack([spatial_hyperedge_fea, temporal_hyperedge_fea])  # 向量垂直拼接，维度变成(2,20)
            # 改写这部分TODO:如果有错的话，可以直接将时空fea的维度先改成三维的，再进行拼接。
            hyperedge = torch.cat((spatial_hyperedge_fea[:,None,:],temporal_hyperedge_fea[:,None,:]),dim=1)  # 向量垂直拼接，维度变成(?-1,2,32)

            atten = self.act_1(torch.matmul(hyperedge, node_fea)/math.sqrt(self.in_dim)).reshape(batch,-1, 1)  # 维度是(?-1,2,1)
            rlt.append(torch.sum(torch.mul(atten, hyperedge), dim=1).reshape(batch,1, -1))  # rlt内部元素的维度是(batch,1,32)
        concatenated = torch.stack(rlt, dim=1).reshape(batch,-1,32)  # dim(?-1,36,32)
        return self.drop(self.act(self.theta(concatenated)))  # 将时空超边按照注意力得分结合起来后，使用线性输出将dim(?-1,36,32)


class TimeBlock(nn.Module):
    def __init__(self, in_dim=32, num_node=36):
        super(TimeBlock, self).__init__()
        self.spatial_hyperedge = SpatialHyperedge(in_dim)
        self.temporal_hyperedge = TemporalHyperedge(in_dim)
        self.spatial_hyperedge_MP = SpatialHyperedgeMP()
        self.temporal_hyperedge_MP = TemporalHyperedgeMP()
        self.node_mp = HHNodeMP()

    def forward(self, pre, cur):  # 台式机上也有同样的问题
        """
        :param cur: (?-1)*N * d
        :param pred: (?-1)*N * d
        :return: (?-1)*N * d
        """
        spatial_hyperedge_loss, spatial_hyperedge_incidence = self.spatial_hyperedge(cur)  # dim(?-1,1) dim(?-1,36,32)
        temporal_hyperedge_loss, temporal_hyperedge_incidence = self.temporal_hyperedge(cur, pre)
        spatial_hyperedge_emb = self.spatial_hyperedge_MP(cur, spatial_hyperedge_incidence)  # 传入(?-1,36,32)、空间超边系数矩阵(36,36)
        temporal_hyperedge_emb = self.temporal_hyperedge_MP(cur, pre, temporal_hyperedge_incidence)  # 传入(?-1,36,32),(36,36),返回的是时间超边和空间超边的特征向量。
        node_emb = self.node_mp(cur, spatial_hyperedge_emb, temporal_hyperedge_emb)  # -->dim(?-1,36,32)
        return node_emb, temporal_hyperedge_loss+spatial_hyperedge_loss  # node_emb维度是(?-1,36,32),表示十个节点的更新后的特征向量,loss:dim(?-1,1)


class HyperModule(nn.Module):
    """
    multi-timestamp training
    """
    def __init__(self, win_size, h_dim=32, n_cats=4, recons_lambda=0.1):
        super(HyperModule, self).__init__()
        self.win_size = win_size
        self.time_cursor = TimeBlock()
        self.predictor = Predictor(h_dim, n_cats=n_cats)
        self.readout = Readout(h_dim)
        self.recons_lambda = recons_lambda

    def forward(self, node_fea):
        recon_loss = 0
        # pre_node.shape(?-1,36,32），cur_node.shape(?-1,36,32)
        for i in range(self.win_size-1):  # win_size设置为2就好。
            if i == 0:
                pre_node = node_fea[:-1,:,:]  # TODO:这里看看是否是(?-1,36,20)
                cur_node = node_fea[1:,:,:]
                cur_node_emb, r_loss = self.time_cursor(pre_node, cur_node)  # cur_node_emb(?-1,36,32)、r_loss(?-1,1)
                recon_loss += r_loss
            else:
                cur_node = node_fea[i + 1].contiguous()
                pre_node = cur_node_emb.contiguous()
                cur_node_emb,r_loss = self.time_cursor(pre_node, cur_node)
                recon_loss += r_loss
        graph_emb = self.readout(cur_node_emb, node_fea[1:,:,:])  # graph_emb.shape(?-1,36,32) ,node_fea[-1].shape(32)
        logits = self.predictor(graph_emb)  # (?-1,36,32)-->(?-1,36,4)
        # 将中间结果传入critic的最后一层，或者直接让predictor最为ciritic网络的最后一层
        return logits, recon_loss*self.recons_lambda  # dim(?-1,36,4) 和 dim(?-1,1)


# 强化学习部分，创建策略网络时，
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim=20, hidden_dim=8, action_dim=4):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):  # 这里传进来的应该是dim(?,36,20)-->dim(?,36,4)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim= -1)  # 输出离散的动作概率


class QValueNet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络，其中可以尝试将dim 20--> dim 8 --> dim 4'''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 先只用超图网络
        self.HyperModule = HyperModule(win_size=2)
        # self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x_pair):  # 这里传的x_pair的dim(?,36,32)
        x = F.relu(self.fc1(x_pair))  # dim(?,36,32)
        total_logits, total_r_loss = self.HyperModule(x)  # -->dim(?-1,36,4) ，dim(?-1,1)
        print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        return total_logits, total_r_loss


class HMASACNet(nn.Module):
    def __init__(self, state_dim=20, hidden_dim=8, action_dim=4, actor_lr=1e-3, critic_lr=1e-2,
                 alpha_lr=1e-2, target_entropy=-1, tau=0.005, gamma=0.98):
        super(HMASACNet,self).__init__()
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim)
        # 第一个Q网络
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim)
        # 第二个Q网络
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim,
                                         action_dim)  # 第一个目标Q网络
        self.target_critic_2 = QValueNet(state_dim, hidden_dim,
                                         action_dim)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.alpha_hyper = 0.01  # 构建超图的损失占整个critic网络损失的占比，TODO:后期需要自己去调参


class HMaSACAgent(Agent):
    def __init__(self,
                 dic_agent_conf=None,
                 dic_traffic_env_conf=None,
                 dic_path=None,
                 cnt_round=None,
                 best_round=None,
                 intersection_id="0",
                 bar_round=None
                 ):

        super(HMaSACAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)  # 传入配置文件

        # 配置文件
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_actions = len(
            self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])  # 有4个动作
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))  # 车道数量为3
        self.memory = self.build_memory()  # 构建经验池
        self.round_cur = cnt_round
        # sac网络的参数设置
        state_dim=20
        hidden_dim=32
        action_dim=4
        actor_lr = 1e-4
        critic_lr = 1e-2
        alpha_lr = 1e-3
        target_entropy = -0.5  # TODO:这个参数可能需要自己去调参 -1-->0
        tau = 0.005
        gamma = 0.98

        if cnt_round == 0:
            # initialization
            self.HmaSACNet = HMASACNet(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma)  # 构建网络

            if os.listdir(self.dic_path["PATH_TO_MODEL"]):  # 这里的path_to_model就是在model的anon的那个路径
                self.HmaSACNet.load_state_dict(torch.load(
                    os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_{0}.pth".format(intersection_id))))  # 加载网络的权重？
            # self.HmaSACNet_bar = self.build_network_from_copy(self.HmaSACNet)  # TODO：修改build函数,bar是一个备份网络
        else:
            # 不太懂里面的bar是啥？
            try:
                if best_round:
                    pass
                else:
                    self.load_network("round_{0}_inter_{1}".format(max(cnt_round - 1,0), self.intersection_id))  # 加载前一轮的模型
            except:
                print("fail to load network, current round: {0}".format(cnt_round))

        # decay the epsilon，贪婪度的衰减
        """
        "EPSILON": 0.8,
        "EPSILON_DECAY": 0.95,
        "MIN_EPSILON": 0.2,
        """
        if os.path.exists(os.path.join(self.dic_path["PATH_TO_MODEL"],"round_-1_inter_{0}.pth".format(intersection_id))):
            # the 0-th model is pretrained model
            self.dic_agent_conf["EPSILON"] = self.dic_agent_conf["MIN_EPSILON"]
            print('round%d, EPSILON:%.4f' % (cnt_round, self.dic_agent_conf["EPSILON"]))
        else:
            decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"],
                                                                   cnt_round)  # 贪婪系数的随着轮次逐步衰减
            self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])  # 贪婪系数最低是0.2


    # 动作预测，返回动作
    def action_att_predict(self, state, total_features=[],  bar=False):
        # state:cur_phase,lane_num_v,adjacency_matrix,adjacency_matrix_lane
        batch_size = len(state)  # 从choose_action传进来的state维度是(1,36)，从prepXs传进来的是(?,36)
        # state = torch.tensor([state], dtype=torch.float)
        if total_features == []:
            total_features = list()
            for i in range(batch_size):
                feature = []  # 大小是20
                for j in range(self.num_agents):
                    observation = []  # 观察值的维度是20（包含当前相位和车辆数）
                    for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                        if 'adjacency' in feature_name:
                            continue
                        if feature_name == "cur_phase":
                            if len(state[i][j][feature_name]) == 1:
                                # choose_action
                                observation.extend(
                                    self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']]
                                    [state[i][j][feature_name][0]])
                            else:
                                observation.extend(state[i][j][feature_name])
                        elif feature_name == "lane_num_vehicle":
                            observation.extend(state[i][j][feature_name])
                    feature.append(observation)  # feature的维度是(36,20)，也就是每个路口的相位以及车辆数。
                total_features.append(feature)  # total_features就是加了batch_size的，维度变成(1,36,20)
            # feature:[agents,feature]
            total_features = np.reshape(np.array(total_features), [batch_size, self.num_agents, -1])  # 维度改成(?,36,20)
            # adj:[agent,neighbors]
        total_features = torch.tensor(total_features, dtype=torch.float)
        all_output = self.HmaSACNet.actor(total_features)  # 这里是使用q网络进行输出，输出的维度是(?,36,4)
        action = all_output

        # out: [batch,agent,action]
        if len(action) > 1:
            # # TODO:当有多个样本时，也就是从prepare_Xs函数进来时，应该要返回什么
            # q_value_1, _ =  self.HmaSACNet.target_critic_1(total_features)  # critic返回一个Q(s,a) dim(?,36,4)；还返回一个构建超边的损失
            # q_value_2, _ =  self.HmaSACNet.target_critic_2(total_features)
            # q_value =  torch.min(q_value_1,q_value_2)
            return total_features # 这里主要是在prepare_Xs_Y中执行该函数时返回。

        # [batch,agent,1]
        action = action.detach().numpy()  # 将tensor转化为numpy
        max_action = np.expand_dims(np.argmax(action, axis=-1), axis=-1)  # dim(1,36,1)
        random_action = np.reshape(np.random.randint(self.num_actions, size=1 * self.num_agents),
                                   (1, self.num_agents, 1))
        # [batch,agent,2]
        possible_action = np.concatenate([max_action, random_action], axis=-1)  # 将Q(s,a)最大的action和随机的action拼在一起
        selection = np.random.choice(
            [0, 1],
            size=batch_size * self.num_agents,
            p=[1 - self.dic_agent_conf["EPSILON"], self.dic_agent_conf["EPSILON"]])
        act = possible_action.reshape((batch_size * self.num_agents, 2))[
            np.arange(batch_size * self.num_agents), selection]
        act = np.reshape(act, (batch_size, self.num_agents))
        return act # dim(1,36,)


    # 随机执行动作,或者执行动作概率最大的那个动作
    def choose_action(self, count, state):
        """
        input: state:[batch_size,num_agent,feature]，默认这里的batch_size是1
        output: out:[batch_size,num_agent,action]
        """
        act = self.action_att_predict([state])  # 原本传进来的state维度是(36,)，经过[]之后，变成（1,36）
        return act[0]  # dim(36,1)


    # 为智能体加载样本，并更新cirtic网络。出现在updater.py文件中
    def prepare_Xs_Y(self, memory, dic_exp_conf):
        """
        memory：(1440,36)表示36个路口的state，next_state，...等
        """
        ind_end = len(memory)  # ind_end为1440
        print("memory size before forget: {0}".format(ind_end))
        # use all the samples to pretrain, i.e., without forgetting
        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            sample_slice = memory  # sample_slice表示样本，大小为1440，因为是一个路口的
        # forget
        else:
            # TODO:这里也将样本进行缩小
            ind_sta = max(0, ind_end - 2000)  # ind_sta为0，MAX_MEMORY_LEN=5000，这里应该可以改样本数量
            # ind_sta = max(0, ind_end - self.dic_agent_conf["MAX_MEMORY_LEN"])  # ind_sta为0，MAX_MEMORY_LEN=10000，这里应该可以改样本数量
            memory_after_forget = memory[ind_sta: ind_end]  # 初始大小也是1440
            print("memory size after forget:", len(memory_after_forget))

            # sample the memory
            # TODO(备注)：先暂时这里设置为200
            sample_size = min(100, len(memory_after_forget))  # SAMPLE_SIZE=1000， 样本大小为1000-->500-->200
            # sample_size = min(self.dic_agent_conf["SAMPLE_SIZE"], len(memory_after_forget))
            # TODO(solved):还有必要打乱么？没有，为了凑成两个时间步的样本，直接将这1000个样本以两两为一组的输入到critic网络中
            start = random.randint(0,len(memory_after_forget)-sample_size)
            sample_slice = memory_after_forget[start:start + sample_size]
            # sample_slice = random.sample(memory_after_forget, sample_size)  # 从memory中随机抽取1000个样本。
            print("memory samples number:", sample_size)

        # 下面的列表最后的维度为1000*36
        _state = []
        _next_state = []
        _action = []
        _reward = []

        # 将1000个样本的几个属性都分开存储
        for i in range(len(sample_slice)):
            _state.append([])
            _next_state.append([])
            _action.append([])
            _reward.append([])
            for j in range(self.num_agents):
                state, action, next_state, reward, _ = sample_slice[i][j]
                _state[i].append(state)
                _next_state[i].append(next_state)
                _action[i].append(action)
                _reward[i].append(reward)


        #target: [#agents,#samples,#num_actions]，根据状态得到特征和q值
        # 其中features的维度是(1000,36,20)
        _features = self.action_att_predict(_state)
        _next_features = self.action_att_predict(_next_state)  # 得到下一状态的特征，邻接路口、车道（基本不变）

        # #self.Xs should be: [#agents,#samples,#features+#]
        self.transition_dict = {
            'states':_features,
            'actions':_action,
            'next_states':_next_features,
            'rewards':_reward,
            'dones':None
        }
        return


    # 构建经验池
    def build_memory(self):

        return []

    # 出现在model_test文件中
    def load_network(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        self.HmaSACNet = torch.load(
            os.path.join(file_path, "%s.pth" % file_name))
        print("succeed in loading model %s"%file_name)


    def save_network(self, file_name):
        torch.save(self.HmaSACNet, os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.pth" % file_name))


    def train_network(self, dic_exp_conf):

        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            epochs = 1000
        else:
            epochs = self.dic_agent_conf["EPOCHS"]  # epchs=100
        # batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))  # batch_size为20
        #
        # TODO:这里还需要改
        for i in range(50):
            self.update(self.transition_dict,i)

            # 这里写一个每次更新完的参数，直接卸载update里面

    # 计算时序误差的
    def calc_target(self, rewards, next_states, dones=0):
        next_probs = self.HmaSACNet.actor(next_states)  # 这里得到的概率是一个离散的分布,dim(?,36,4)
        next_probs = next_probs[1:]
        rewards = rewards[1:]
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=-1, keepdim=True)  # dim(?-1,36,1)离散动作的熵正则化项的计算，就是将每个动作的概率log并与当前概率乘积之后再求和，得到当前策略的一个评估参数
        q1_value, _ = self.HmaSACNet.target_critic_1(next_states)  # dim(?-1,36,4)
        q2_value, _ = self.HmaSACNet.target_critic_2(next_states)
        # TODO:这里炸了
        # min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=-1, keepdim=True)  # dim(?-1,36,1)
        # 重写
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=-1,keepdim=True)

        next_value = min_qvalue + self.HmaSACNet.log_alpha.exp() * entropy
        td_target = rewards + self.HmaSACNet.gamma * next_value * (1 - dones)
        return td_target  # dim(?-1,36,1)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.HmaSACNet.tau) + param.data * self.HmaSACNet.tau)


    # TODO:这个部分应该写在updater.py文件中,传进来的是字典，所以这部分应该会修改。
    def update(self, transition_dict, i):
        states = torch.tensor(transition_dict['states'], dtype=torch.float)  # dim(1000,36,20)
        actions = torch.tensor(transition_dict['actions']).view(-1, self.num_agents ,1) # 动作不再是float类型dim(1000,36,1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, self.num_agents, 1)  # dim(1000,36,1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)  # dim(1000,36,20)
        # dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        # 更新两个Q网络,TODO:现在有个很严重的问题，就是网络的更新是哪个网络的更新，是单个网络的更新还是整体网络的更新
        td_target = self.calc_target(rewards, next_states)  # dim(?-1,36,1)
        actions = actions[1:]
        # TODO:这里应该是要分开，等下在试验一下gather的用法。
        critic_1_q_values, critic_1_recon_loss = self.HmaSACNet.critic_1(states)  # dim(?-1,36,4) and dim(?-1,1)注意这里的寿面那个1是否单独是一个1
        critic_q_1_a = critic_1_q_values.gather(-1, actions) # dim(?-1,36,1)
        critic_1_loss = ((1-self.HmaSACNet.alpha_hyper)*torch.mean(F.mse_loss(critic_q_1_a, td_target.detach())) +
                         self.HmaSACNet.alpha_hyper * torch.mean(critic_1_recon_loss)) # TODO：将构建超边的损失加入到这个loss里面，然后使用一个超参数进行训练
        critic_2_q_values, critic_2_recon_loss = self.HmaSACNet.critic_2(states)
        critic_q_2_a = critic_2_q_values.gather(-1, actions)
        critic_2_loss = ((1-self.HmaSACNet.alpha_hyper) * torch.mean(F.mse_loss(critic_q_2_a, td_target.detach())) +
                         self.HmaSACNet.alpha_hyper * torch.mean(critic_2_recon_loss))  # 这里是直接计算所有路口，所有样本的损失。
        self.HmaSACNet.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.HmaSACNet.critic_1_optimizer.step()
        self.HmaSACNet.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.HmaSACNet.critic_2_optimizer.step()
        print('round:'+ str(self.round_cur) + ' Epoch:' + str(i+1) + ' critic_1_loss:' + str(critic_1_loss))
        print('round:'+ str(self.round_cur) + ' Epoch:' + str(i+1) + ' critic_2_loss:' + str(critic_2_loss))

        # 更新策略网络
        probs = self.HmaSACNet.actor(states)
        probs = probs[1:]  # 改变维度，舍去第一个样本
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵，这里算是与连续动作不同的点
        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)  # dim(?-1,36,1)
        q1_value, _ = self.HmaSACNet.critic_1(states)
        q2_value, _ = self.HmaSACNet.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=-1, keepdim=True)  # 直接根据概率计算期望dim(?-1,36,1)
        actor_loss = torch.mean(-self.HmaSACNet.log_alpha.exp() * entropy - min_qvalue)
        self.HmaSACNet.actor_optimizer.zero_grad()
        actor_loss.backward()
        print('round:'+ str(self.round_cur) +' Epoch:' + str(i+1) +  ' actor_loss:' + str(actor_loss))

        self.HmaSACNet.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean((entropy - self.HmaSACNet.target_entropy).detach() * self.HmaSACNet.log_alpha.exp())
        self.HmaSACNet.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.HmaSACNet.log_alpha_optimizer.step()

        self.soft_update(self.HmaSACNet.critic_1, self.HmaSACNet.target_critic_1)
        self.soft_update(self.HmaSACNet.critic_2, self.HmaSACNet.target_critic_2)


