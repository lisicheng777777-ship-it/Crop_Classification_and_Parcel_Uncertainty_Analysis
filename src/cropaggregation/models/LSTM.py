import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os

__all__ = ['LSTM']

class LSTM(torch.nn.Module):
    def __init__(self, input_dim=13, num_classes=9, hidden_dims=32, num_layers=2, dropout=0.5, # 0.5713020228087161   # origin code = hidden_dims=128, num_layers=4
                 bidirectional=True, use_layernorm=True):
        self.modelname = f"LSTM_input-dim={input_dim}_num-classes={num_classes}_hidden-dims={hidden_dims}_" f"num-layers={num_layers}_bidirectional={bidirectional}_use-layernorm={use_layernorm}" f"_dropout={dropout}"

        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.use_layernorm = use_layernorm
        # 如果是双向LSTM，隐藏维度需要加倍
        self.hidden_dims = hidden_dims * 2 if bidirectional else hidden_dims
        self.d_model = num_layers * hidden_dims # 定义模型的特征维度大小，基于隐藏层的维度和 LSTM 层数

        if use_layernorm:
            self.inlayernorm = nn.LayerNorm(input_dim)
            self.clayernorm = nn.LayerNorm(self.hidden_dims)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, num_layers=num_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        self.linear_class = nn.Linear(self.hidden_dims, num_classes, bias=True) # 最后的全连接层

    def logits(self, x): # 前向传播的主要方法，生成模型的输出
        if self.use_layernorm:
            x = self.inlayernorm(x)
        outputs, last_state_list = self.lstm.forward(x)
        h, c = last_state_list
        # outputs: (batch_size, seq_len, num_directions * hidden_size) 128 7 256
        # outputs包含最后一层所有时间步的隐藏状态（每个时间步的前向和后向隐藏状态的拼接）。
        # last_state_list包含所有层最后一个时间步的隐藏状态和细胞状态。
        # last_state_list：(num_layers * num_directions, batch_size, hidden_size) 8 128 128
        # h包括了所有层的最后一个时间步的隐藏状态，c则是最后一个时间步的细胞状态。这里h=bi2*layer4=8 h的结构与c相同。

        h_forward = h[-2, :, :] if self.lstm.bidirectional else h[-1, :, :]
        h_backward = h[-1, :, :] if self.lstm.bidirectional else None
        # 拼接正向和反向隐藏状态
        if self.lstm.bidirectional:
            h_final = torch.cat((h_forward, h_backward), dim=-1)  # 拼接最后一层的正反向隐藏状态
        else:
            h_final = h_forward  # 单向LSTM直接使用正向隐藏状态

        # LayerNorm（如果启用）
        if self.use_layernorm:
            h_final = self.clayernorm(h_final)

        logits = self.linear_class.forward(h_final) #  # 通过线性分类器生成 logits

        return logits

    def forward(self, x):
        logprobabilities = F.log_softmax(self.logits(x), dim=-1)
        return logprobabilities

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to " + path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state, **kwargs), path)

    def load(self, path):
        print("loading model from " + path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot

# # 创建一个实例
# model = LSTM()
#
# # 打印模型结构
# print(model)

# 如果希望同时捕捉多个时间步的特征，可以使用卷积层：
# 在时间序列中使用卷积神经网络（如TempCNN）可以实现这种跨时间步捕捉特征的需求。
# 卷积核的 kernel_size 可以直接控制跨越几个时间步来提取特征，
# 而LSTM则在内部记忆单元中逐步累积记忆，生成长短期依赖。

class LSTM1(torch.nn.Module):
    def __init__(self, input_dim=13, num_classes=9, hidden_dims=128, num_layers=4, dropout=0.5, # 0.5713020228087161
                 bidirectional=True, use_layernorm=True):
        self.modelname = f"LSTM_input-dim={input_dim}_num-classes={num_classes}_hidden-dims={hidden_dims}_" f"num-layers={num_layers}_bidirectional={bidirectional}_use-layernorm={use_layernorm}" f"_dropout={dropout}"

        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.use_layernorm = use_layernorm
        # 如果是双向LSTM，隐藏维度需要加倍
        self.hidden_dims = hidden_dims * 2 if bidirectional else hidden_dims
        self.d_model = num_layers * hidden_dims # 定义模型的特征维度大小，基于隐藏层的维度和 LSTM 层数

        if use_layernorm:
            self.inlayernorm = nn.LayerNorm(input_dim)
            self.clayernorm = nn.LayerNorm(self.hidden_dims)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, num_layers=num_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        self.linear_class = nn.Linear(self.hidden_dims, num_classes, bias=True) # 最后的全连接层

    def logits(self, x): # 前向传播的主要方法，生成模型的输出
        if self.use_layernorm:
            x = self.inlayernorm(x)
        attention = Attention(self.hidden_dims)
        outputs, last_state_list = self.lstm.forward(x)
        h_attention = attention(outputs)

        # LayerNorm（如果启用）
        if self.use_layernorm:
            h_attention = self.clayernorm(h_attention)

        logits = self.linear_class.forward(h_attention) #  # 通过线性分类器生成 logits

        return logits

    def forward(self, x):
        logprobabilities = F.log_softmax(self.logits(x), dim=-1)
        return logprobabilities

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to " + path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state, **kwargs), path)

    def load(self, path):
        print("loading model from " + path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, outputs):
        # outputs shape: (batch_size, seq_len, hidden_size)
        scores = self.attention_weights(outputs).squeeze(-1)  # shape: (batch_size, seq_len)
        weights = F.softmax(scores, dim=1)  # 归一化得到权重
        context = torch.sum(outputs * weights.unsqueeze(-1), dim=1)  # 加权求和
        return context