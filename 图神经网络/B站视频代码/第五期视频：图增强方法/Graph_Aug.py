import random

import numpy as np
import torch
from torch import cosine_similarity


# 节点特征扰动
def perturb_node_features(x, noise_level=0.01):
    """
    对节点特征进行轻微扰动。
    参数:
    - data: 图数据。
    - noise_level: 噪声水平。
    """
    x = x + noise_level * torch.randn_like(x)
    return x

# 节点特征掩码
def mask_node_features(x, mask_rate=0.1):
    """
    随机掩码节点特征。
    参数:
    - data: 图数据。
    - mask_rate: 掩码率。
    """
    num_nodes, num_features = x.size()
    mask = np.random.binomial(1, mask_rate, (num_nodes, num_features))
    mask = torch.FloatTensor(mask).to(x.device)
    x = x * (1 - mask)
    return x

# 边添加
def add_edges(edge_index, num_nodes, add_rate=0.01):
    """
    随机添加边。
    参数:
    - data: 图数据。
    - add_rate: 添加边的比例。
    """
    num_nodes = num_nodes
    num_add = int(num_nodes * num_nodes * add_rate)
    edge_index = edge_index.t().tolist()
    all_possible_edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if
                          i != j and [i, j] not in edge_index]
    added_edges = random.sample(all_possible_edges, k=min(num_add, len(all_possible_edges)))
    added_edges = torch.tensor(added_edges, dtype=torch.long).t()
    edge_index = torch.cat([edge_index, added_edges], dim=1)
    return edge_index

# 边删除
def remove_edges(edge_index, remove_rate=0.01):
    """
    随机删除边。
    参数:
    - data: 图数据。
    - remove_rate: 删除边的比例。
    """
    num_edges = edge_index.size(1)
    num_remove = int(num_edges * remove_rate)
    edge_index = edge_index.t().tolist()
    removed_edge_idxs = random.sample(range(num_edges), k=num_remove)
    edge_index = [edge for i, edge in enumerate(edge_index) if i not in removed_edge_idxs]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index.cuda()


def adjust_edge_weights_by_similarity(x, edge_index, edge_attr):
    """
    根据节点特征的相似度调整边权重。
    参数:
    - data: 图数据对象，需要有边索引`edge_index`和节点特征`x`。

    返回:
    - 修改权重后的图数据对象。
    """
    # 确保data对象有边权重，如果没有，则初始化为1。
    if edge_attr is None:
        edge_attr = torch.ones((edge_index.size(1),), dtype=torch.float)
    # 计算所有边的节点特征的相似度
    edge_features_src = x[edge_index[0]]
    edge_features_dst = x[edge_index[1]]
    similarities = cosine_similarity(edge_features_src, edge_features_dst, dim=1)
    # 以相似度作为新的边权重
    edge_attr = similarities
    return edge_attr


from torch_geometric.utils import k_hop_subgraph


def extract_subgraph(data, node_idx, num_hops):
    """
    从给定的图数据中抽取以node_idx为中心的num_hops跳的子图。

    参数:
    - data: 图数据对象。
    - node_idx: 子图中心节点的索引。
    - num_hops: 子图的跳数。

    返回:
    - 子图数据对象。
    """
    # 获取子图的节点和边索引
    sub_nodes, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, num_hops, edge_index=data.edge_index, relabel_nodes=True
    )
    # 创建子图数据对象
    sub_data = data.__class__()
    sub_data.edge_index = sub_edge_index
    sub_data.x = data.x[sub_nodes]
    # 如果有边属性，也进行抽取
    if data.edge_attr is not None:
        sub_data.edge_attr = data.edge_attr[edge_mask]
    # 如：sub_data.y = data.y[sub_nodes]
    return sub_data


