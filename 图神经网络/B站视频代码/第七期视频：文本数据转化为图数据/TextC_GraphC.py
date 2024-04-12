

'''
案例数据集使用的是 ChnSentiCorp_htl_all数据集
7000 多条酒店评论数据，5000 多条正向评论，2000 多条负向评论
地址：https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv
 TextC_GraphC.py ----------- 将文本分类任务转化为图分类任务：每条评论中出现的词作为一个节点。
'''


import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
import jieba
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import train_test_split
# 读取数据
data = pd.read_csv('data.csv')[:100]
# 处理空值
data['review'].fillna('', inplace=True)
# 分词
data['tokens'] = data['review'].apply(lambda x: list(jieba.cut(x)))

# 创建图数据对象列表
graph_data_list = []
for idx, row in data.iterrows():
    tokens = row['tokens']
    vocab_indices = {token: idx for idx, token in enumerate(tokens)}
    nodes = list(range(len(tokens)))
    # 使用滑动窗口构建边关系
    window_size = 3  # 滑动窗口大小
    edges = []
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + window_size, len(tokens))):
            edges.append((vocab_indices[tokens[i]], vocab_indices[tokens[j]]))
    # 创建图数据对象
    graph_data = Data(
        x=torch.tensor([[1] for _ in nodes], dtype=torch.float),  # 节点特征，这里用简单的标记表示节点
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),  # 边索引
        y=torch.tensor([row['label']], dtype=torch.long)  # 图分类的标签，即每行文本的分类
    )

    graph_data_list.append(graph_data)
print(graph_data_list)
# 划分训练、验证和测试集
train_data, test_data = train_test_split(graph_data_list, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 定义图神经网络模型
class GNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.out = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.out(x)
        return F.log_softmax(x, dim=1)


model = GNN(num_node_features=1, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(200):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")

# 验证模型
model.eval()
val_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch in val_loader:
        out = model(batch)
        val_loss += F.nll_loss(out, batch.y, reduction='sum').item()
        pred = out.argmax(dim=1, keepdim=True)
        correct += pred.eq(batch.y.view_as(pred)).sum().item()
        total += batch.y.size(0)

print(f"Validation Loss: {val_loss / total}, Accuracy: {100 * correct / total}%")

# 测试模型
model.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        out = model(batch)
        test_loss += F.nll_loss(out, batch.y, reduction='sum').item()
        pred = out.argmax(dim=1, keepdim=True)
        correct += pred.eq(batch.y.view_as(pred)).sum().item()
        total += batch.y.size(0)
print(f"Test Loss: {test_loss / total}, Accuracy: {100 * correct / total}%")
