import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()

        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        squeeze_x = self.GAP(x)
        squeeze_x = squeeze_x.view(x.size(0), x.size(1))
        squeeze_x = self.relu(self.fc1(squeeze_x))
        squeeze_x = self.sigmoid(self.fc2(squeeze_x))
        squeeze_x = squeeze_x.view(x.size(0), x.size(1), 1)

        return squeeze_x * x

class CNNBlock(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, padding):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()
        self.se = SEBlock(out_layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        
        return x


# AttentionBlock Dense(384, 768)과 input_dim용 dense까지 3개를 사용하였다.
class AttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim * 2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        Y = self.fc1(x)
        Y = self.fc2(Y)
        Y = self.gap(Y)
        
        W = self.softmax(Y)

        # print('output shape: ', K.shape)
        
        return W * x

class BiLSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.0):
        super(BiLSTMBlock, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,batch_first=True, 
                            dropout=dropout, bidirectional=True)#, batch_first=True

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super(DenseBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        return x

class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return x