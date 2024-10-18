import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import DataLoader
from ModelBlock_v2 import CNNBlock, AttentionBlock, BiLSTMBlock, DenseBlock, Transpose
import ModelTrain as mtrain

'''
NSR: 0 / AFR: 1 / OTH: 2

MuDANet.py => Main
ModelTrain.py => Model Train Code -> mtrain
ModelBlock.py => Model을 구성하는 블럭 -> nblock
'''

# MuDANet 모델 설계
class MuDANet(nn.Module):
    def __init__(self, num_classes):
        super(MuDANet, self).__init__()

        # 1. CNN block Streams 
        self.cnn_stream1 = self.create_stream()
        self.cnn_stream2 = self.create_stream()

        # 2. BiLSTM blocks
        self.bilstm_stream1 = self.create_bilstm_block()
        self.bilstm_stream2 = self.create_bilstm_block()
    
        # 3. Fully Cannected 
        self.fc1 = DenseBlock(256, 1024, dropout=0.2)
        self.fc2 = DenseBlock(1024, 1024, dropout=0.2)
        self.fc3 = DenseBlock(1024, 256, dropout=0.0)
        self.final_gap = nn.AdaptiveAvgPool1d(1)
   
        # 4. Classification 
        self.fc4 = DenseBlock(33, num_classes, dropout=0.0)
        self.softmax = nn.Softmax(dim=1)

    def create_stream(self):
        return nn.Sequential(
            CNNBlock(1, 128, 3, 1, 1),
            CNNBlock(128, 256, 9, 3, 3),
            CNNBlock(256, 256, 9, 3, 3),
            Transpose(),
            DenseBlock(256, 32),
            DenseBlock(32, 64),
            DenseBlock(64, 128),
            Transpose(),
            CNNBlock(128, 128, 3, 1, 1),
            CNNBlock(128, 256, 9, 3, 3),
            CNNBlock(256, 256, 9, 3, 3),
            Transpose(),
            DenseBlock(256, 32),
            DenseBlock(32, 64),
            DenseBlock(64, 128),
            AttentionBlock(128, 384)
        )

    def create_bilstm_block(self):
        return nn.Sequential(
            BiLSTMBlock(128, 128, 4),
            nn.Dropout(0.4)
        )

    def forward(self, x):
        
        # CNN Streams
        x1 = self.cnn_stream1(x)
        
        # First Fusion
        x_fused = x1

        # x_fused = x_fused.permute(0, 2, 1)
        
        # BiLSTM Blocks
        x_fused1 = self.bilstm_stream1(x_fused)
        
        # Second Fusion
        x_final_fusion = x_fused1 
        
        x_final_fusion = self.fc1(x_final_fusion)
        x_final_fusion = self.fc2(x_final_fusion)
        x_final_fusion = self.fc3(x_final_fusion)

        # Classification
        x_final_fusion = self.final_gap(x_final_fusion)
        x_final_fusion = x_final_fusion.squeeze(2)
        output = self.fc4(x_final_fusion)
        output = self.softmax(output)

        return output
