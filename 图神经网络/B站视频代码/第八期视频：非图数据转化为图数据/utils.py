from sklearn.neighbors import kneighbors_graph
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def nearest_neighbors_sparse(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    edge_index = np.vstack(np.nonzero(adj)).astype(np.int64)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index

def nearest_neighbors_dense(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    return torch.tensor(adj)

def cosine_similarity_adj(X, threshold):
    similarity_matrix = cosine_similarity(X)
    adj = similarity_matrix > threshold
    edge_index = np.array(np.where(adj)).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return edge_index.T