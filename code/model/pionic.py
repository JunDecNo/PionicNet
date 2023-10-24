import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import Transformer
# 定义pionic Model
class Pionic(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, layers=4, num_heads=4, dropout=0.1):
        super(Pionic, self).__init__()
        # 必须对输入的ligand+protein进行嵌入，以便适合encoder
        self.input = nn.Sequential(
            nn.LayerNorm(feature_dim, eps=1e-6),
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU()
        )
        # 定义了transformer encoder层
        self.encoder = nn.ModuleList([
            Transformer(hidden_dim, num_heads, dropout) 
            for _ in range(layers)
            ])
        # mlp用于接收transformer的输出，并且只对除第一个向量以外的向量进行处理
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim, eps=1e-6),
            nn.Linear(hidden_dim,1,bias=True)
        )
    
    def forward(self, x, mask=None):
        # 对数据进行input处理
        x = self.input(x)
        # encoder处理
        for layer in self.encoder:
            x = layer(x, mask)
        # mlp完成节点分类
        x = self.mlp(x)
        return x