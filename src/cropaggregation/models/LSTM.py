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
        # Double the hidden dimension for a bidirectional LSTM.
        self.hidden_dims = hidden_dims * 2 if bidirectional else hidden_dims
        self.d_model = num_layers * hidden_dims # Define the feature dimension from hidden size and layer count.

        if use_layernorm:
            self.inlayernorm = nn.LayerNorm(input_dim)
            self.clayernorm = nn.LayerNorm(self.hidden_dims)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, num_layers=num_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        self.linear_class = nn.Linear(self.hidden_dims, num_classes, bias=True) # Final fully connected layer.

    def logits(self, x): # Compute model logits.
        if self.use_layernorm:
            x = self.inlayernorm(x)
        outputs, last_state_list = self.lstm.forward(x)
        h, c = last_state_list
        # outputs: (batch_size, seq_len, num_directions * hidden_size) 128 7 256
        # outputs contains final-layer hidden states at every time step, concatenated across directions.
        # last_state_list contains final hidden and cell states for all layers.
        # last_state_list：(num_layers * num_directions, batch_size, hidden_size) 8 128 128
        # h and c contain final states for each layer and direction.

        h_forward = h[-2, :, :] if self.lstm.bidirectional else h[-1, :, :]
        h_backward = h[-1, :, :] if self.lstm.bidirectional else None
        # Concatenate forward and backward hidden states.
        if self.lstm.bidirectional:
            h_final = torch.cat((h_forward, h_backward), dim=-1)  # Concatenate final-layer states from both directions.
        else:
            h_final = h_forward  # Use the forward state for a unidirectional LSTM.

        # Apply LayerNorm if enabled.
        if self.use_layernorm:
            h_final = self.clayernorm(h_final)

        logits = self.linear_class.forward(h_final) # Generate logits with the linear classifier.

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

# Create a model instance.
# model = LSTM()
#
# Print the model architecture.
# print(model)

# Convolutional layers can capture features across multiple time steps.
# Temporal convolutional networks such as TempCNN capture these temporal patterns.
# kernel_size controls the temporal span of each convolution.
# LSTMs accumulate information in memory cells to capture temporal dependencies.

class LSTM1(torch.nn.Module):
    def __init__(self, input_dim=13, num_classes=9, hidden_dims=128, num_layers=4, dropout=0.5, # 0.5713020228087161
                 bidirectional=True, use_layernorm=True):
        self.modelname = f"LSTM_input-dim={input_dim}_num-classes={num_classes}_hidden-dims={hidden_dims}_" f"num-layers={num_layers}_bidirectional={bidirectional}_use-layernorm={use_layernorm}" f"_dropout={dropout}"

        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.use_layernorm = use_layernorm
        # Double the hidden dimension for a bidirectional LSTM.
        self.hidden_dims = hidden_dims * 2 if bidirectional else hidden_dims
        self.d_model = num_layers * hidden_dims # Define the feature dimension from hidden size and layer count.

        if use_layernorm:
            self.inlayernorm = nn.LayerNorm(input_dim)
            self.clayernorm = nn.LayerNorm(self.hidden_dims)

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims, num_layers=num_layers,
                            bias=False, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        self.linear_class = nn.Linear(self.hidden_dims, num_classes, bias=True) # Final fully connected layer.

    def logits(self, x): # Compute model logits.
        if self.use_layernorm:
            x = self.inlayernorm(x)
        attention = Attention(self.hidden_dims)
        outputs, last_state_list = self.lstm.forward(x)
        h_attention = attention(outputs)

        # Apply LayerNorm if enabled.
        if self.use_layernorm:
            h_attention = self.clayernorm(h_attention)

        logits = self.linear_class.forward(h_attention) # Generate logits with the linear classifier.

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
        weights = F.softmax(scores, dim=1)  # Normalize attention scores into weights.
        context = torch.sum(outputs * weights.unsqueeze(-1), dim=1)  # Compute the weighted sum.
        return context