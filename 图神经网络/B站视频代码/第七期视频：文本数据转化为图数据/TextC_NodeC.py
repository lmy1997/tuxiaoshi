

'''
案例数据集使用的是 ChnSentiCorp_htl_all数据集
7000 多条酒店评论数据，5000 多条正向评论，2000 多条负向评论
地址：https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv
 TextC_NodeC.py ----------- 将文本分类任务转化为节点分类任务：将每个评论作为一个节点。
'''

import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import Data
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# 读取数据文件
df = pd.read_csv('data.csv')[:10]
df.dropna(inplace=True)
# 打乱数据顺序
df = df.sample(frac=1).reset_index(drop=True)

# 去除评论内容中的标点符号
def remove_punctuation(text):
    punctuation = '，。！？；：“”‘’（）【】《》【】'
    return ''.join([c for c in text if c not in punctuation])

df['review_cleaned'] = df['review'].apply(remove_punctuation)

# 分词
def tokenize(text):
    return jieba.lcut(text)

df['tokens'] = df['review_cleaned'].apply(tokenize)

# 计算TF-IDF编码作为节点特征
corpus = [' '.join(tokens) for tokens in df['tokens']]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
tfidf_features = torch.from_numpy(tfidf_matrix.toarray()).float()
# 构建节点编号映射
node_to_idx = {node: idx for idx, node in enumerate(df.index)}
# 构建边列表
edges = []
total_iterations = len(df['tokens']) * len(df['tokens'][0])  # 计算总的迭代次数，用于进度条显示

with tqdm(total=total_iterations, desc='Processing edges') as pbar:  # 创建进度条对象
    for idx, tokens in enumerate(df['tokens']):
        for token in tokens:
            edges.extend([(node_to_idx[idx], node_to_idx[other_idx]) for other_idx, other_tokens in enumerate(df['tokens']) if token in other_tokens and idx != other_idx])
            pbar.update(1)

# 去除重复的边
edges = list(set(edges))
# 构建PyG Data对象
edge_index = torch.tensor(edges).t().contiguous()
labels = torch.tensor(df['label'].values, dtype=torch.long)
data = Data(x=tfidf_features, edge_index=edge_index, y=labels)
print(data)
# 划分数据集为训练集、验证集和测试集
num_nodes = data.num_nodes
indices = np.arange(num_nodes)
np.random.shuffle(indices)
# 划分数据集大小
train_size = int(0.8 * num_nodes)
val_size = int(0.1 * num_nodes)
# 创建训练集、验证集和测试集的 mask
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[indices[:train_size]] = True
val_mask[indices[train_size:train_size + val_size]] = True
test_mask[indices[train_size + val_size:]] = True
# 更新 Data 对象中的 train_mask, val_mask, test_mask
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

# 创建 GCN 模型
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


model = GCN(in_channels=tfidf_features.shape[1], hidden_channels=64, out_channels=2)

# 定义损失函数和优化器
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# 模型训练和评估
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    val_out = model(data.x, data.edge_index)
    val_pred = out.argmax(dim=1)
    val_correct = val_pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
    val_acc = val_correct / data.val_mask.sum().item()

    test_out = model(data.x, data.edge_index)
    test_pred = test_out.argmax(dim=1)
    test_correct = test_pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    test_acc = test_correct / data.test_mask.sum().item()
    print(f'Epoch: [{epoch:03d}/{num_epochs}], Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')



