import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, add_self_loops
from torch_geometric.nn.conv import MessagePassing
from torch import ones_like, sparse_coo_tensor, svd_lowrank
from sklearn.manifold import LocallyLinearEmbedding

class MP(MessagePassing):
    def __init__(self):
        super(MP, self).__init__()

    def forward(self, x, edge_index, norm=None):
        return self.propagate(edge_index=edge_index, x=x, norm=None)

    def message(self, x_j, norm=None):
        if norm != None:
            return norm.view(-1, 1) * x_j # 广播计算
        else:
            return x_j


class Basis_Generator(nn.Module):
    def __init__(self, nx, nlx, nl, k, operator, low_x=False, low_lx=False, norm1=False):
        super(Basis_Generator, self).__init__()

        self.nx = nx
        self.nlx = nlx
        self.nl = nl
        self.norm1 = norm1
        self.k = k
        self.operator = operator
        self.low_x = low_x
        self.low_lx = low_lx
        self.mp = MP()

    def get_x_basis(self, x):
        x = F.normalize(x, dim=1) # 对于每个节点，对所有维特征进行规范化
        x = F.normalize(x, dim=0) # 对于每维特征，对所有节点进行规范化

        if self.low_x:
            # 是否对节点特征矩阵x进行有损压缩(基于奇异值分解)
            U, S, V = svd_lowrank(x, q=self.nx)
            low_x = U @ torch.diag(S)
            return low_x
        else:
            return x

    def get_lx_basis(self, x, edge_index):
        lxs = []
        num_nodes = x.shape[0]

        # L = I - D^(-1/2) A D^(-1/2) edge_index再添加自环
        edge_index_lap, edge_weight_lap = get_laplacian(edge_index=edge_index, normalization='sym', num_nodes=num_nodes)
        h = F.normalize(x, dim=1)

        if self.operator == 'gcn':
            lxs = [h]
            edge_index, edge_weight = add_self_loops(edge_index=edge_index_lap,
                                                     edge_attr=-edge_weight_lap,
                                                     fill_value=2.0,
                                                     num_nodes=num_nodes)
            edge_index, edge_weight = get_laplacian(edge_index=edge_index,
                                                    edge_weight=edge_weight,
                                                    normalization='sym',
                                                    num_nodes=num_nodes)
            edge_index, edge_weight = add_self_loops(edge_index=edge_index,
                                                    edge_attr=-edge_weight,
                                                    fill_value=1.,
                                                    num_nodes=num_nodes)
            for k in range(self.k + 1):
                h = self.mp.propagate(edge_index=edge_index, x=h, norm=edge_weight)
                if self.norm1:
                    h = F.normalize(h, dim=1)
                lxs.append(h)

        elif self.operator == 'gpr':
            lxs = [h]
            edge_index, edge_weight = add_self_loops(edge_index=edge_index_lap,
                                                     edge_attr=-edge_weight_lap,
                                                     fill_value=1.0,
                                                     num_nodes=num_nodes)
            for k in range(self.k):
                h = self.mp.propagate(edge_index=edge_index, x=h, norm=edge_weight)
                if self.norm1:
                    h = F.normalize(h, dim=1)
                lxs.append(h)

        elif self.operator == 'cheb':
            edge_index, edge_weight = add_self_loops(edge_index=edge_index_lap,
                                                     edge_attr=edge_weight_lap,
                                                     fill_value=-1.0,
                                                     num_nodes=num_nodes)
            for k in range(self.k + 1):
                if k == 0:
                    pass
                elif k == 1:
                    h = self.mp.propagate(edge_index=edge_index, x=h, norm=edge_weight)
                else:
                    h = self.mp.propagate(edge_index=edge_index, x=h, norm=edge_weight) * 2
                    h = h - lxs[-1]
                if self.norm1:
                    h = F.normalize(h, dim=1)
                lxs.append(h)

        elif self.operator == 'ours':
            lxs = [h]
            edge_index, edge_weight = add_self_loops(edge_index=edge_index_lap,
                                                     edge_attr=edge_weight_lap,
                                                     fill_value=-1.0,
                                                     num_nodes=num_nodes)
            for k in range(self.k):
                h = self.mp.propagate(edge_index=edge_index, x=h, norm=edge_weight)
                h = h - lxs[-1]
                if self.norm1:
                    h = F.normalize(h, dim=1)
                lxs.append(h)

        norm_lxs = []
        low_lxs = []
        for lx in lxs:
            if self.low_lx:
                U, S, V = svd_lowrank(lx, q=self.nlx)
                low_lx = U @ torch.diag(S)
                low_lxs.append(low_lx)
                norm_lxs.append(F.normalize(low_lx, dim=1))
            else:
                norm_lxs.append(F.normalize(lx, dim=1))

        final_lxs = [F.normalize(lx, dim=0) for lx in lxs]
        return final_lxs

    def get_l_basis(self, edge_index, num_nodes):
        """对图结构(邻接)矩阵进行有损压缩(基于奇异值分解)"""
        # get_laplacian先计算 L = I - D^(-1/2) A D^(-1/2) edge_index再添加自环
        edge_index, edge_weight = get_laplacian(edge_index=edge_index, normalization='sym', num_nodes=num_nodes)
        adj = sparse_coo_tensor(indices=edge_index,
                                values=ones_like(edge_index[0]),
                                size=(num_nodes, num_nodes),
                                device=edge_index.device,
                                dtype=torch.float32).to_dense()
        adj = F.normalize(adj, dim=1)  # 对二维矩阵, 沿着列dim=1 对行 进行规范化
        U, S, V = svd_lowrank(adj, q=self.nl, niter=2)  # 奇异值分解 adj ≈ U diag(S) V^T
        adj = U @ torch.diag(S)  # 矩阵近似 adj ≈ U diag(S)
        adj = F.normalize(adj, dim=0)  # 对二维矩阵, 沿着行dim=0 对列 进行规范化
        return adj
    
    def get_l_basis_nmf(self, edge_index, num_nodes, nmf_iters=100, epsilon=1e-10):
        #. 对图结构(邻接)矩阵进行有损压缩（基于非负矩阵分解）
        edge_index, edge_weight = get_laplacian(edge_index=edge_index, normalization='sym', num_nodes=num_nodes)
        adj = sparse_coo_tensor(indices=edge_index,
                                values=ones_like(edge_index[0]),
                                size=(num_nodes, num_nodes),
                                device=edge_index.device,
                                dtype=torch.float32).to_dense()
        adj = F.normalize(adj, dim=1)  # 对二维矩阵, 沿着列dim=1 对行 进行规范化
        adj = torch.clamp(adj, min=0)
        W = torch.rand(num_nodes, self.nl, device=adj.device, dtype=adj.dtype)
        H = torch.rand(self.nl, num_nodes, device=adj.device, dtype=adj.dtype)
        for _ in range(nmf_iters):
            WH = torch.matmul(W, H)
            numerator = torch.matmul(adj, H.t())
            denominator = torch.matmul(WH, H.t()) + epsilon
            W *= numerator / denominator
            W = torch.clamp(W, min=epsilon)  
            WH = torch.matmul(W, H)
            numerator = torch.matmul(W.t(), adj)
            denominator = torch.matmul(W.t(), WH) + epsilon
            H *= numerator / denominator
            H = torch.clamp(H, min=epsilon)  
        W = F.normalize(W, p=2, dim=0)
        return W
    
    def get_l_basis_lle(self, edge_index, num_nodes, n_neighbors=10, n_components=None, eigen_solver='auto', tol=1e-6, max_iter=200):
        edge_index, edge_weight = get_laplacian(edge_index=edge_index, normalization='sym', num_nodes=num_nodes)
        adj = sparse_coo_tensor(indices=edge_index,
                                values=torch.ones(edge_index.size(1), device=edge_index.device),
                                size=(num_nodes, num_nodes),
                                device=edge_index.device,
                                dtype=torch.float32).to_dense()
        adj = F.normalize(adj, p=1, dim=1)
        adj_np = adj.cpu().numpy()
        if n_components is None:
            n_components = self.nl
        lle = LocallyLinearEmbedding(n_neighbors=n_neighbors,
                                     n_components=n_components,
                                     eigen_solver=eigen_solver,
                                     tol=tol,
                                     max_iter=max_iter)
        lle_embedding = lle.fit_transform(adj_np)  # 形状为 [num_nodes, n_components]
        adj_lle = torch.tensor(lle_embedding, device=edge_index.device, dtype=torch.float32)
        adj_lle = F.normalize(adj_lle, dim=0)
        return adj_lle



class FE_GNN(nn.Module):
    def __init__(self, args, ninput, nclass):
        super(FE_GNN, self).__init__()

        self.nx = ninput if args.nx < 0 else args.nx
        self.nlx = ninput if args.nlx < 0 else args.nlx
        self.nl = args.nl
        self.k = args.k
        self.operator = args.operator

        self.basis_generator = Basis_Generator(nx=self.nx, nlx=self.nlx, nl=self.nl, k=self.k, operator=args.operator,
                                               low_x=False, low_lx=False, norm1=False)

        self.share_lx = args.share_lx
        self.thetas = nn.Parameter(torch.ones(args.k + 1), requires_grad=True)

        self.lin_lxs = nn.ModuleList()
        for i in range(self.k + 1):
            self.lin_lxs.append(nn.Linear(self.nlx, args.nhid, bias=True))

        self.lin_x = nn.Linear(self.nx, args.nhid, bias=True)
        self.lin_lx = nn.Linear(self.nlx, args.nhid, bias=True)
        self.lin_l = nn.Linear(self.nl, args.nhid, bias=True)

        self.cls = nn.Linear(args.nhid, nclass, bias=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x_basis = self.basis_generator.get_x_basis(x)
        lx_basis = self.basis_generator.get_lx_basis(x, edge_index)
        # l_basis = self.basis_generator.get_l_basis(edge_index, x.shape[0])
        # l_basis = self.basis_generator.get_l_basis_nmf(edge_index, x.shape[0])
        l_basis = self.basis_generator.get_l_basis_lle(edge_index, x.shape[0])
        feature_mat = 0

        if self.nx > 0:
            x_mat = self.lin_x(x_basis)
            feature_mat += x_mat

        if self.nlx > 0:
            lxs_mat = 0
            for k in range(self.k + 1):
                if self.share_lx:
                    lx_mat = self.lin_lx(lx_basis[k]) * self.thetas[k] # share W_lx across each layer/order
                else:
                    lx_mat = self.lin_lxs[k](lx_basis[k]) # do not share the W_lx parameters
                lxs_mat = lxs_mat + lx_mat
            feature_mat += lxs_mat

        if self.nl > 0:
            l_mat = self.lin_l(l_basis)
            feature_mat += l_mat

        output = self.cls(feature_mat)

        return F.log_softmax(output, dim=1)
