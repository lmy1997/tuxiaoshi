import argparse
import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, ChebConv, TransformerConv, GCNConv, GATConv
import torch.optim as optim
import torch.nn.functional as F
from Graph_Aug import *
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
# setup_seed(42)

# 命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description='Base-Graph Neural Network')
    parser.add_argument('--dataset', choices=['Cora', 'Citeseer', 'Pubmed'], default='Cora',
                        help="Dataset selection")
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--model', choices=['GCN', 'GAT', 'SAGE', 'ChebNet', 'TransformerConv'],
                        default='GAT',
                        help="Model selection")
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
    def __init__(self, in_channels, hidden_channels, out_channels, model_type, dropout_rate):
        super(GNN, self).__init__()
        self.dropout_rate = dropout_rate

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

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr).relu()
        # edge_attr = adjust_edge_weights_by_similarity(x, edge_index, edge_attr)
        # x = mask_node_features(x, mask_rate=0.1)
        # x = perturb_node_features(x, noise_level=0.01)
        # edge_index = remove_edges(edge_index, remove_rate=0.01)
        # edge_index = add_edges(edge_index, x.shape[0])

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x

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
    data = dataset[0]
    data = data.to(device)
    print(data)
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