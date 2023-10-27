

# pytorch训练模块
import torch
from torch import nn
from model import PnModel

if __name__ == "__main__":
    seq = 50
    feature_dim = 128
    tensor = torch.randn(1, seq, feature_dim)
    model = PnModel(feature_dim)
    output = model(tensor)
