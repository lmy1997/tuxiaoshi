import argparse
import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, TransformerConv
import torch.optim as optim

from model import FE_GNN


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
setup_seed(42)

# 命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description='Base-Graph Neural Network')
    parser.add_argument('--dataset', choices=['Cora', 'Citeseer', 'Pubmed'], default='Cora',
                        help="Dataset selection")
    parser.add_argument('--lr', default=0.01, help="Learning Rate selection")
    parser.add_argument('--wd', default=0.0001, help="weight_decay selection")
    parser.add_argument('--epochs', default=1000, help="train epochs selection")
    parser.add_argument('--tsne_drawing', choices=[True, False], default=False,
                        help="Whether to use tsne drawing")
    parser.add_argument('--tsne_colors', default=['#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'], help="colors")
    # FE-GNN:
    parser.add_argument('--nx', type=int, default=-1,
                        help='Rank of singular value decomposition of node feature matrix, defaulting to -1: Use the node feature dimension')
    parser.add_argument('--nlx', type=int, default=-1,
                        help='Rank of singular value decomposition of feature matrix, defaulting to -1: Use the node feature dimension')
    parser.add_argument('--nl', type=int, default=50,
                        help='Rank of singular value decomposition of graph structure matrix, defaulting to 0: Do Not Use')
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument("--operator", type=str, default='gpr', choices=['gcn', 'gpr', 'cheb', 'ours'])
    parser.add_argument('--nhid', type=int, default=64, help='Hidden dimension of feature transformation')
    parser.add_argument('--share_lx', action='store_true', default=False,
                        help='Share the same W for different hops of lx')

    return parser.parse_args()


# 加载数据集
def load_dataset(name):
    dataset = Planetoid(root='dataset/' + name, name=name, transform=T.NormalizeFeatures())
    return dataset

# 使用Tsne绘图
def plot_points(z, y):
    z = TSNE(n_components=2).fit_transform(z.detach().cpu().numpy())
    classes = len(torch.unique(y))
    y = y.cpu().numpy()
    plt.figure(figsize=(8, 8))
    for i in range(classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=args.tsne_colors[i])
    plt.axis('off')
    plt.savefig('{} embeddings ues tnse to plt figure.png'.format(args.model))
    plt.show()


def train(model, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs, logits

if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(args.dataset)
    data = dataset[0].to(device)
    print(data)
    model = FE_GNN(args, data.x.shape[1],data.y.max()+1).to(device)
    print(model)
    print(f"Loaded {args.dataset} dataset with {data.num_nodes} nodes and {data.num_edges} edges.")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    Best_Acc = []
    for epoch in range(1, args.epochs):
        loss = train(model, data)
        accs, log= test(model, data)
        train_acc, val_acc, test_acc = accs
        print(f'Epoch: [{epoch:03d}/{args.epochs:04d}], Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
        Best_Acc.append(test_acc)
    if args.tsne_drawing == True:
        plot_points(log, data.y)
    print('---------------------------')
    print('Best Acc: {:.4f}'.format(max(Best_Acc)))
    print('---------------------------')
