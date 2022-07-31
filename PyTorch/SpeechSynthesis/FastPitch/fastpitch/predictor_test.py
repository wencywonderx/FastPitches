import torch
from torch import rand
import torch.nn as nn
import torch.nn.functional as F

class ConvReLUNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0):
        super(ConvReLUNorm, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    padding=(kernel_size // 2))
        self.norm = torch.nn.LayerNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, signal):
        out = F.relu(self.conv(signal))
        out = self.norm(out.transpose(1, 2)).transpose(1, 2).to(signal.dtype)
        return self.dropout(out)


class TemporalPredictor(nn.Module):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, filter_size, kernel_size, dropout,
                 n_layers=2, n_predictions=1):
        super(TemporalPredictor, self).__init__()

        self.layers = nn.Sequential(*[
            ConvReLUNorm(input_size if i == 0 else filter_size, filter_size,
                         kernel_size=kernel_size, dropout=dropout)
            for i in range(n_layers)]
        )
        self.n_predictions = n_predictions
        self.fc = nn.Linear(filter_size, self.n_predictions, bias=True)

    def forward(self, enc_out, enc_out_mask): # mask is to ignore 0s when predicting
        out = enc_out * enc_out_mask # [16, 148, 384]
        # print(out.shape)
        out = self.layers(out.transpose(1, 2)).transpose(1, 2) # [16, 148, 256]
        # print(out.shape)
        # print(self.fc(out).shape) # [16, 148, 1]
        out = self.fc(out) * enc_out_mask # [16, 148, 1], after embedding [16, 148, 384]
        # print(out.shape)
        return out


class MeanPredictor(nn.Module):
    """Predicts a single float per sample"""

    def __init__(self, input_size, hidden_size, n_predictions=2):
        super(MeanPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, n_predictions) 
        self.hidden_size = hidden_size
    def forward(self, input):       
        # out = enc_out * enc_out_mask
        # print(out.shape) # [16, 148, 384]
        h0 = torch.zeros(1, input.size(0), self.hidden_size, device=input.device)
        c0 = torch.zeros(1, input.size(0), self.hidden_size, device=input.device)
        lstm_out, _ = self.lstm(input.permute(1, 0, 2), (h0,c0))
        print(lstm_out.shape) # [148, 16, 256]
        # print(self.hidden[0].shape) # [1, 16, 256]
        # print(self.hidden[1].shape) # [1, 16, 256]
        print(lstm_out[-1, :, :].shape) # [16, 256]
        y_pred = self.fc(lstm_out[-1, :, :])
        print(y_pred.shape) # [16, 1]
        return y_pred

# model = TemporalPredictor(384, 256, 3, 0.1)
# enc_out = rand(16, 148, 384)
# # print(enc_out)
# enc_mask = rand(16, 148, 1)
# # print(enc_mask)
# outputs = model.forward(enc_out, enc_mask)

model = MeanPredictor(384, 256)
enc_out = rand(16, 148, 384) # input size: [batch_size, max_input_length, hidden]
enc_mask = rand(16, 148, 1) 
input = enc_out * enc_mask
outputs = model.forward(input)
print(outputs)
# reshape = outputs.unsqueeze(1) # expected [batch_size]




# input_mean = rand(3, 1)
# input_delta = rand(3, 4, 10)
# enc_out = rand(3, 5, 384)
# mean_f0_emb = nn.Linear(1, 384)
# emb = mean_f0_emb(input)
# print(emb)
# output = emb.view(16, 1, 384) + enc_out
# print(output)
# # print(input.view(16, 10, 1))

# delta_f0_pred = rand(16, 1, 148)
# print(delta_f0_pred)
# mean_f0_pred = rand(16, 1)
# print(mean_f0_pred)
# mean_and_delta_pred = delta_f0_pred + mean_f0_pred.view(16, 1, 1)
# print(mean_and_delta_pred.shape)

