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
import torch.nn.functional as F
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

setup_seed(42)
# 命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description='Dual-Channels Graph Neural Network')
    parser.add_argument('--dataset', choices=['Cora', 'Citeseer', 'Pubmed'], default='Cora',
                        help="Dataset selection")
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden layer dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--lr', default=0.01, help="Learning Rate selection")
    parser.add_argument('--wd', default=5e-4, help="weight_decay selection")
    parser.add_argument('--epochs', default=200, help="train epochs selection")
    parser.add_argument('--model', choices=['GCN', 'GAT', 'SAGE', 'ChebNet', 'TransformerConv', 'Others'],
                        default='GCN', help="Model selection")
    parser.add_argument('--FusionMethod', choices=['Sum', 'Dot', 'Concat', 'Average', 'Max', 'Attention'],
                        default='Attention', help="Select Fusion Method")
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
    def __init__(self, in_channels, hidden_channels, out_channels, model_type, dropout_rate):
        super(GNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.lin = torch.nn.Linear(out_channels *2, out_channels)

        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, out_channels)
        elif model_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        elif model_type == 'ChebNet':
            self.conv1 = ChebConv(in_channels, hidden_channels, K=2)
            self.conv2 = ChebConv(hidden_channels, out_channels, K=2)
        elif model_type == 'TransformerConv':
            self.conv1 = TransformerConv(in_channels, hidden_channels)
            self.conv2 = TransformerConv(hidden_channels, out_channels)
        else:
            raise NotImplementedError


    def forward(self, x, edge_index):
        # first channel
        x1 = self.conv1(x, edge_index).relu()
        x1 = F.dropout(x1, p=self.dropout_rate, training=self.training)
        x1 = self.conv2(x1, edge_index)
        # second channel
        x2 = self.conv1(x, edge_index).relu()
        x2 = F.dropout(x2, p=self.dropout_rate, training=self.training)
        x2 = self.conv2(x2, edge_index)
        # Apply fusion method
        if args.FusionMethod == 'Sum':
            x = x1 + x2
        elif args.FusionMethod == 'Dot':
            x = x1 * x2
        elif args.FusionMethod == 'Concat':
            x = torch.cat([x1, x2], dim=1)
            x = self.lin(x)
        elif args.FusionMethod == 'Average':
            x = (x1 + x2) / 2
        elif args.FusionMethod == 'Max':
            x = torch.max(x1, x2)
        elif args.FusionMethod == 'Attention':
            attention_weights = F.softmax(torch.randn(x1.size(0), 1), dim=0).cuda()
            x = attention_weights * x1 + (1 - attention_weights) * x2
        else:
            raise NotImplementedError
        return F.log_softmax(x)

def train(model, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 新增测试函数
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
    data = dataset[0].to(device)
    model = GNN(in_channels=dataset.num_node_features, hidden_channels=args.hidden_dim,
                out_channels=dataset.num_classes, model_type=args.model, dropout_rate=args.dropout_rate).to(device)
    print(model)
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