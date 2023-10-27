import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv, GATv2Conv, GAT
from torch_geometric.data import Data


# 就决定是你了，HGAT。
# 1024，没有明白如何加入GRU

# 定义NodeModel
# class NodeModel(nn.Module):
#     def __init__(self):
#         super(NodeModel, self).__init__()


# 定义GraphModel, 用于训练
# 边存在权重，节点存在特征
class GATa(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=num_heads, dropout=0.2)
        self.conv2 = GATConv(hidden_dim * num_heads, 1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


x = torch.rand((5, 16))  # 5个节点，每个节点有16维特征
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 4],
                           [1, 0, 2, 1, 3, 2, 3, 2]], dtype=torch.long)
edge_attr = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 4],
                           [1, 0, 2, 1, 3, 2, 3, 2]], dtype=torch.long)
y = torch.tensor([0, 1, 0, 1, 1], dtype=torch.float)  # 二分类标签

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr ,y=y)
model = GATa(in_dim=16, hidden_dim=8, num_heads=2)
output = model(data.x, data.edge_index)

print(output)
print(nn.Softmax(output))
