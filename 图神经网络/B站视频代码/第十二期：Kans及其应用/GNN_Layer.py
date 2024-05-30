import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils import to_dense_adj
from KAN import KANLinear
from FastKAN import FastKANLayer

class GCNConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.FloatTensor(out_features))

        # self.mlp = nn.Linear(in_features, out_features)
        self.FastKAN = FastKANLayer(in_features, out_features)


    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        adj = torch.zeros(num_nodes, num_nodes).cuda()
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = 1.0 / torch.sqrt(deg)
        norm_adj = adj * deg_inv_sqrt.unsqueeze(1)
        norm_adj = norm_adj * deg_inv_sqrt.unsqueeze(0)
        support = self.FastKAN(x)
        output = torch.matmul(norm_adj, support) + self.bias
        return output
