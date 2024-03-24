import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import to_dense_adj


class GCNConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNConv, self).__init__()
        # 输入特征的维度
        self.in_features = in_features
        # 输出特征的维度
        self.out_features = out_features
        # 定义可学习的权重矩阵
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # 定义可学习的偏置向量
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        # 使用Xavier初始化权重矩阵
        nn.init.xavier_uniform_(self.weight)
        # 将偏置向量初始化为0
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        # x: 节点特征矩阵，大小为[num_nodes, in_features]
        # edge_index: 边索引，大小为[2, num_edges]

        # 计算规范化的邻接矩阵
        num_nodes = x.size(0)
        # 创建一个全零矩阵作为邻接矩阵的初始状态
        adj = torch.zeros(num_nodes, num_nodes).cuda()
        # 根据edge_index填充邻接矩阵，无向图因此两个方向都要填充
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1
        # 计算每个节点的度
        deg = torch.sum(adj, dim=1)
        # 计算度矩阵的逆平方根，用于后续的归一化
        deg_inv_sqrt = 1.0 / torch.sqrt(deg)  # 加上一个小的常数避免除零错误
        # 计算对称归一化的邻接矩阵
        norm_adj = adj * deg_inv_sqrt.unsqueeze(1)
        norm_adj = norm_adj * deg_inv_sqrt.unsqueeze(0)

        # 支撑传播：线性变换节点特征
        support = torch.matmul(x, self.weight)  # X * W
        # 消息传递：通过归一化的邻接矩阵传播特征
        output = torch.matmul(norm_adj, support) + self.bias  # D^{-1/2} * A * D^{-1/2} * (X * W) + b
        return output  # 返回输出，可以选择使用ReLU等激活函数进行非线性变换

class GATConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2, concat=True):
        super(GATConv, self).__init__()
        # 定义dropout率，用于在注意力系数上进行dropout操作以防止过拟合
        self.dropout = dropout
        # 输入特征的维度
        self.in_features = in_features
        # 输出特征的维度
        self.out_features = out_features
        # LeakyReLU非线性激活函数中的负斜率alpha
        self.alpha = alpha
        # 是否在多头注意力中进行拼接，对于最后一层通常设为False，使用平均
        self.concat = concat
        # 定义可学习的权重矩阵W，用于线性变换输入特征
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # 定义注意力机制中可学习的参数a
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # 定义LeakyReLU激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_index):
        # 将边索引转换为稠密邻接矩阵，并去除多余的维度
        adj = to_dense_adj(edge_index).squeeze(0)
        # 应用线性变换
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        # 准备注意力机制的输入
        e = self._prepare_attentional_mechanism_input(Wh)
        # 创建一个足够小的向量用于掩盖不存在的边
        zero_vec = -9e15*torch.ones_like(e)
        # 只有当adj中存在边时，才保留e中的值，否则用zero_vec中的极小值代替
        attention = torch.where(adj > 0, e, zero_vec)
        # 对注意力系数进行softmax操作，使得每个节点的注意力系数和为1
        attention = F.softmax(attention, dim=1)
        # 对注意力系数进行dropout
        attention = F.dropout(attention, self.dropout, training=self.training)
        # 应用注意力机制更新节点特征
        h_prime = torch.matmul(attention, Wh)

        # 如果concat为真，则对输出使用ELU激活函数；否则直接返回结果
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # 这个函数负责计算注意力系数
        # 首先通过与a的前半部分做矩阵乘法计算得到每个节点的影响力分数Wh1
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        # 通过与a的后半部分做矩阵乘法计算得到每个节点被影响的分数Wh2
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # 将Wh1加上Wh2的转置，得到每一对节点的非归一化注意力分数e
        e = Wh1 + Wh2.T
        # 使用LeakyReLU激活函数处理e，增加非线性
        return self.leakyrelu(e)
