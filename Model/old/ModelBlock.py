import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride):
        super(CNNBlock, self).__init__()
        # Loss를 줄이기위해 BatchNorm1d를 사용함
        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, padding='valid')

    def forward(self, x):
        x = self.conv1(x)

        return x

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()

        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        batch_size, num_channels, H = x.size()

        # squeeze_x = self.GAP(x).view(batch_size, num_channels)
        squeeze_x = self.GAP(x)
        squeeze_x = squeeze_x.squeeze(dim=2)
        # print('squeeze_shape: ', squeeze_x.shape)

        squeeze_x = F.relu(self.fc1(squeeze_x))
        squeeze_x = F.sigmoid(self.fc2(squeeze_x))
        squeeze_x = squeeze_x.unsqueeze(dim=2)
        # print('last shape: ', squeeze_x.shape)

        return x * squeeze_x


# class SEBlock(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(SEBlock, self).__init__()
#
#         self.GAP = nn.AdaptiveAvgPool1d(1)
#         self.fc1 = nn.Linear(in_channels, in_channels // reduction)
#         self.fc2 = nn.Linear(in_channels // reduction, in_channels)
#
#     def forward(self, x):
#         batch_size, num_channels, H = x.size()
#
#         squeeze_x = self.GAP(x).view(batch_size, num_channels)
#
#         squeeze_x = F.relu(self.fc1(squeeze_x))
#         squeeze_x = F.sigmoid(self.fc2(squeeze_x))
#
#
#         return x * squeeze_x.view(batch_size, num_channels, 1)

'''
class SEBlock(nn.Module):
    def __init__(self, in_layer, out_layer):
        super(SEBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer // 16, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer // 16, in_layer, kernel_size=1, padding=0)

    def forward(self, x):
        # 논문에 있는데로 GAP를 사용하여 squeeze를 진행하였음
        x_squeeze = nn.functional.adaptive_avg_pool1d(x, 1)
        x_squeeze = self.conv1(x_squeeze)
        x_squeeze = F.relu(x_squeeze)
        x_squeeze = self.conv2(x_squeeze)
        x_squeeze = F.sigmoid(x_squeeze)
        return torch.add(x, x_squeeze)
'''
# AttentionBlock Dense(384, 768)과 input_dim용 dense까지 3개를 사용하였다.
class AttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim * 2)
        # 논문에는 dense layer가 2개 밖에 없으나 이상태로 원본 x와 multiply가 안되어 다시 원래 input_dim으로 변환
        self.fc3 = nn.Linear(output_dim * 2, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        attention = self.fc1(x)
        attention = self.fc2(attention)
        # attention = self.fc3(attention)
        attention = self.softmax(attention)
        print('x transposed shape: ', x.T.shape)
        print('attention shape: ', attention.shape)
        K = x.T.matmul(attention)
        return K


class BiLSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.4):
        super(BiLSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True,
                            bidirectional=True)

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