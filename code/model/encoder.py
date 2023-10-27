# 我觉得将离子直接加入序列中应该是可行的
import torch
import torch.nn as nn
import torch.nn.functional as F


# self attention module
class SelfAttention(nn.Module):
    def __init__(self, num_hidden, num_heads=4):  # num_hidden表示隐藏层，num_heads表示头的数目
        super().__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.attention_head_size = int(num_hidden / num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size

    def transpose_for_scores(self, x):  # 作用：将输入转换为attention的输入形式，将隐藏层转化为头数目的形式
        # 原始输入x的形状为[batch_size, seq_len, hidden_size]
        # 修改后的x的形状为[batch_size, seq_len, num_heads, attention_head_size]
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)  # 这里的size()[:-1]是指除了最后一维的其他维度
        x = x.view(*new_x_shape)  # view()函数作用是将一个多行的Tensor,拼接成一行
        return x.transpose(-1,-2)  # permute(0, 2, 1)将第二维和第三维交换位置

    def forward(self, q, k, v, mask=None):
        # q, k, v分别表示transformer中的Q,K,V用于计算两两之间的attention
        print('before q size:', q.size())
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        print('q size:', q.size())
        attention_scores = torch.matmul(q, k.transpose(-1, -2))  # 计算attention的分数
        print('k transpose size:', k.transpose(-1,-2).size())
        print('attention:',attention_scores.size())
        if mask is not None:  # 带入mask，表示只对mask部分计算attention
            attention_mask = (1.0 - mask) * -10000
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(1)

        attention_scores = nn.Softmax(dim=-1)(attention_scores)  # 对attention的分数进行softmax操作

        outputs = torch.matmul(attention_scores, v)  # 计算attention的输出
        print(outputs)
        outputs = outputs.transpose(-1, -2).contiguous()
        print(outputs)
        print('permute output:', outputs.size())
        new_output_shape = outputs.size()[:-2] + (self.all_head_size,)
        print('new_output:', outputs.size()[:-1])
        outputs = outputs.view(*new_output_shape)  # 将输出的形状转换为[batch_size, seq_len, hidden_size]
        print('last output', outputs.size())
        return outputs


class PositionWiseFeedForward(nn.Module):  # transformer中的position FNN
    def __init__(self, input_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.PF_inL = nn.Linear(input_dim, hidden_dim, bias=True)
        self.PF_outL = nn.Linear(hidden_dim, input_dim, bias=True)

    def forward(self, h):
        h = self.PF_inL(h)
        h = F.leaky_relu(h)
        h = self.PF_outL(h)
        return h


# 定义Transformer层
class Transformer(nn.Module):
    def __init__(self, num_hidden=64, num_heads=4, dropout=0.1):
        super(Transformer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(num_hidden, eps=1e-6)
        self.ln2 = nn.LayerNorm(num_hidden, eps=1e-6)
        self.attention = SelfAttention(num_hidden, num_heads)
        self.pff = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, seq, mask=None):
        # self attention
        dh = self.attention(seq, seq, seq, mask)
        seq = self.ln1(seq + self.dropout(dh))  # 残差连接+layer norm

        # position-wise feed forward
        dh = self.pff(seq)
        seq = self.ln2(seq + self.dropout(dh))

        # mask
        if mask is not None:
            seq = mask.unsqueeze(-1) * seq
        return seq
