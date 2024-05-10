import argparse
import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import nn
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils import nearest_neighbors_sparse, nearest_neighbors_dense, cosine_similarity_adj
from torch_geometric.utils import dense_to_sparse, to_dense_adj
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
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden layer dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', default=0.01, help="Learning Rate selection")
    parser.add_argument('--wd', default=0, help="weight_decay selection")
    parser.add_argument('--epochs', default=200, help="train epochs selection")
    parser.add_argument('--tsne_drawing', choices=[True, False], default=False,
                        help="Whether to use tsne drawing")
    parser.add_argument('--tsne_colors', default=['#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'], help="colors")
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

# 定义模型
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super(GNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def train(model, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    model.eval()
    logits, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs, logits

if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(args.dataset)
    data = dataset[0]
    # knn_adj = nearest_neighbors_sparse(data.x.numpy(), k=7, metric='minkowski')
    cons_adj = cosine_similarity_adj(data.x.numpy(), 0.5)
    data.edge_index = cons_adj
    data = data.to(device)
    model = GNN(in_channels=data.x.shape[1], hidden_channels=args.hidden_dim,
                out_channels=dataset.num_classes, dropout_rate=args.dropout_rate).to(device)
    print(f"Loaded {args.dataset} dataset with {data.num_nodes} nodes and {data.num_edges} edges.")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    Best_Acc = []
    for epoch in range(1, args.epochs):
        loss = train(model, data)
        accs, log= test(model, data)
        train_acc, val_acc, test_acc = accs
        print(f'Epoch: [{epoch:03d}/200], Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
        Best_Acc.append(test_acc)
    if args.tsne_drawing == True:
        plot_points(log, data.y)
    print('---------------------------')
    print('Best Acc: {:.4f}'.format(max(Best_Acc)))
    print('---------------------------')