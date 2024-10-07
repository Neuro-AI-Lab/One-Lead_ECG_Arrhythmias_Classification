import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import DataLoader
import ModelBlock as mblock
import ModelTrain as mtrain

'''
NSR: 0 / AFR: 1 / OTH: 2

MuDANet.py => Main
ModelTrain.py => Model Train Code -> mtrain
ModelBlock.py => Model을 구성하는 블럭 -> nblock
'''

# numpy data path
data_path = 'C:/Users/LJY/Desktop/ECG/Dataset/training_9s.npy'
label_path = 'C:/Users/LJY/Desktop/ECG/Dataset/training_9s_label.npy'
learning_late = 0.0001


# MuDANet 모델 설계
class MuDANet(nn.Module):
    def __init__(self, num_classes):
        super(MuDANet, self).__init__()

        # 1_1. CNNBlock1 Stream-1 (input_layer, output_layer_kernel_size, stride)
        self.cnn_block1_stream1 = mblock.CNNBlock(1, 128, 3, 1)
        self.se_block1_stream1 = mblock.SEBlock(128, 128)

        self.cnn_block2_stream1 = mblock.CNNBlock(128, 256, 9, 3)
        self.se_block2_stream1 = mblock.SEBlock(256, 256)

        self.cnn_block3_stream1 = mblock.CNNBlock(256, 256, 9, 3)
        self.se_block3_stream1 = mblock.SEBlock(256, 256)
        # 2_1. DenseBlock (input_dim, output_dim)
        self.dense_block1_stream1 = mblock.DenseBlock(76032, 32)
        self.dense_block2_stream1 = mblock.DenseBlock(32, 64)
        self.dense_block3_stream1 = mblock.DenseBlock(64, 128)

        # 3_1. CNNBlock2 Stream-1 (input_layer, output_layer_kernel_size, stride)
        self.cnn_block4_stream1 = mblock.CNNBlock(1, 128, 3, 1)
        self.se_block4_stream1 = mblock.SEBlock(128, 128)

        self.cnn_block5_stream1 = mblock.CNNBlock(128, 256, 9, 3)
        self.se_block5_stream1 = mblock.SEBlock(256, 256)

        self.cnn_block6_stream1 = mblock.CNNBlock(256, 256, 9, 3)
        self.se_block6_stream1 = mblock.SEBlock(256, 256)
        # 4_1. DenseBlock (input_dim, output_dim)
        self.dense_block4_stream1 = mblock.DenseBlock(2816, 32)
        self.dense_block5_stream1 = mblock.DenseBlock(32, 64)
        self.dense_block6_stream1 = mblock.DenseBlock(64, 128)
        # 5_1. AttentionBlock(input_dim, output_dim)
        self.attention_block_stream1 = mblock.AttentionBlock(128, 384)


        # 1_2. CNNBlock1 Stream-2 (input_layer, output_layer_kernel_size, stride)
        self.cnn_block1_stream2 = mblock.CNNBlock(1, 128, 3, 1)
        self.se_block1_stream2 = mblock.SEBlock(128, 128)

        self.cnn_block2_stream2 = mblock.CNNBlock(128, 256, 9, 3)
        self.se_block2_stream2 = mblock.SEBlock(256, 256)

        self.cnn_block3_stream2 = mblock.CNNBlock(256, 256, 9, 3)
        self.se_block3_stream2 = mblock.SEBlock(256, 256)
        # 2_2. DenseBlock (input_dim, output_dim)
        self.dense_block1_stream2 = mblock.DenseBlock(76032, 32)
        self.dense_block2_stream2 = mblock.DenseBlock(32, 64)
        self.dense_block3_stream2 = mblock.DenseBlock(64, 128)

        # 3_2. CNNBlock2 Stream-2 (input_layer, output_layer_kernel_size, stride)
        self.cnn_block4_stream2 = mblock.CNNBlock(1, 128, 3, 1)
        self.se_block4_stream2 = mblock.SEBlock(128, 128)

        self.cnn_block5_stream2 = mblock.CNNBlock(128, 256, 9, 3)
        self.se_block5_stream2 = mblock.SEBlock(256, 256)

        self.cnn_block6_stream2 = mblock.CNNBlock(256, 256, 9, 3)
        self.se_block6_stream2 = mblock.SEBlock(256, 256)
        # 4_2. DenseBlock (input_dim, output_dim)
        self.dense_block4_stream2 = mblock.DenseBlock(2816, 32)
        self.dense_block5_stream2 = mblock.DenseBlock(32, 64)
        self.dense_block6_stream2 = mblock.DenseBlock(64, 128)
        # 5_2. AttentionBlock (input_dim, output_dim)
        self.attention_block_stream2 = mblock.AttentionBlock(128, 384)


        # 6_1. Bi-LSTM Block Stream-1 (input_dim, output_dim)
        self.bilstm1_block = mblock.BiLSTMBlock(128, 256)
        # 6_2. Bi-LSTM Block Stream-2 (input_dim, output_dim)
        self.bilstm2_block = mblock.BiLSTMBlock(128, 256)

        # 7. Fully Cannected (input_dim, output_dim)
        self.fc1 = mblock.DenseBlock(512, 1024)
        self.fc2 = mblock.DenseBlock(1024, 1024)
        self.fc3 = mblock.DenseBlock(1024, 256, dropout=0.0)

        # 8. output (input_dim, output_dim)
        self.fc4 = mblock.DenseBlock(256, num_classes, dropout=0.0)

    def forward(self, x):
        # Stream-1
            # CNNBlock-1
        x1 = self.cnn_block1_stream1(x)
        print('cnn1: ' + str(str(x1.shape)))
        x1 = self.se_block1_stream1(x1)
        print('se1: ' + str(str(x1.shape)))
        x1 = self.cnn_block2_stream1(x1)
        print('cnn2: ' + str(str(x1.shape)))
        x1 = self.se_block2_stream1(x1)
        print('se2: ' + str(x1.shape))
        x1 = self.cnn_block3_stream1(x1)
        print('cnn3: ' + str(x1.shape))
        x1 = self.se_block3_stream1(x1)
        print('se3: ' + str(x1.shape))

        x1 = torch.flatten(x1, start_dim=1)
        print('flatten1: ' + str(x1.shape))
        x1 = self.dense_block1_stream1(x1)
        print('dnn1: ' + str(x1.shape))
        x1 = self.dense_block2_stream1(x1)
        print('dnn2: ' + str(x1.shape))
        x1 = self.dense_block3_stream1(x1)
        print('dnn3: ' + str(x1.shape))

            # CNNBlock-2
        x1 = x1.view(x1.size(0), 1, -1)
        print('reshape1: ' + str(x1.shape))
        x1 = self.cnn_block4_stream1(x1)
        print('cnn4: ' + str(x1.shape))
        x1 = self.se_block4_stream1(x1)
        print('se4: ' + str(x1.shape))
        x1 = self.cnn_block5_stream1(x1)
        print('cnn5: ' + str(x1.shape))
        x1 = self.se_block5_stream1(x1)
        print('se5: ' + str(x1.shape))
        x1 = self.cnn_block6_stream1(x1)
        print('cnn6: ' + str(x1.shape))
        x1 = self.se_block6_stream1(x1)
        print('se6: ' + str(x1.shape))

        x1 = torch.flatten(x1, start_dim=1)
        print('flatten2: ' + str(x1.shape))
        x1 = self.dense_block4_stream1(x1)
        print('dnn4: ' + str(x1.shape))
        x1 = self.dense_block5_stream1(x1)
        print('dnn5: ' + str(x1.shape))
        x1 = self.dense_block6_stream1(x1)
        print('dnn6: ' + str(x1.shape))
            # AttentionBlock
        x1 = self.attention_block_stream1(x1)
        print('attention: ' + str(x1.shape))


        # Stream-2
            # CNNBlock-1
        x2 = self.cnn_block1_stream2(x)
        x2 = self.se_block1_stream2(x2)

        x2 = self.cnn_block2_stream2(x2)
        x2 = self.se_block2_stream2(x2)

        x2 = self.cnn_block3_stream2(x2)
        x2 = self.se_block3_stream2(x2)

        x2 = torch.flatten(x2, start_dim=1)
        x2 = self.dense_block1_stream2(x2)
        x2 = self.dense_block2_stream2(x2)
        x2 = self.dense_block3_stream2(x2)

            # CNNBlock-2
        x2 = x2.view(x2.size(0), 1, -1)
        x2 = self.cnn_block4_stream2(x2)
        x2 = self.se_block4_stream2(x2)

        x2 = self.cnn_block5_stream2(x2)
        x2 = self.se_block5_stream2(x2)

        x2 = self.cnn_block6_stream2(x2)
        x2 = self.se_block6_stream2(x2)

        x2 = torch.flatten(x2, start_dim=1)
        x2 = self.dense_block4_stream2(x2)
        x2 = self.dense_block5_stream2(x2)
        x2 = self.dense_block6_stream2(x2)
            # AttentionBlock
        x2 = self.attention_block_stream2(x2)

        # model_add
        x_fused = torch.add(x1, x2)
        print('fused model: ' + str(x_fused.shape))
        x_fused = torch.flatten(x_fused, start_dim=1)
        print('flatten_fused1: ' + str(x_fused.shape))
        x_fused = x_fused.view(x_fused.size(0), 1, -1)
        print('reshape_fused1: ' + str(x_fused.shape))

        # Bi-LSTM-Stream-1
        x_fused1 = self.bilstm1_block(x_fused)
        print('biLstm_fused1: ' + str(x_fused1.shape))
        # Bi-LSTM-Stream-2
        x_fused2 = self.bilstm2_block(x_fused)

        # model_add
        x_fused = torch.add(x_fused1, x_fused2)
        print('fused: ' + str(x_fused.shape))
        # Fully_Connected
        x_fused = F.relu(self.fc1(x_fused))
        print('fc1: ' + str(x_fused.shape))
        x_fused = F.relu(self.fc2(x_fused))
        print('fc2: ' + str(x_fused.shape))
        x_fused = F.relu(self.fc3(x_fused))
        print('fc3: ' + str(x_fused.shape))

        # Output
        x_fused = self.fc4(x_fused)
        print('output: ' + str(x_fused.shape))
        return x_fused



# MuDANet 실행 코드
def main():
    train_loader, val_loader = dataload()

    model = MuDANet(num_classes=3)
    model.apply(mtrain.weights_init)

    # CrossEntropyLoss(), Adam 사용
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model train
    mtrain.train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=40, device=device)

# 데이터 불러올때 사용하는 코드
def dataload():
    np_data = np.load(data_path)
    label = np.load(label_path)

    # 데이터 크기 변환  n x 1 x 2700으로 변환 (n은 데이터 개수)
    data = torch.tensor(np_data.reshape(np_data.shape[0], 1, np_data.shape[2]), dtype=torch.float32)
    labels = torch.tensor(label, dtype=torch.long)

    # 데이터 나누기
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.1)

    # ModelTrain에 있는 ECGDataset class
    train_dataset = mtrain.ECGDataset(train_data, train_labels)
    val_dataset = mtrain.ECGDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader


if __name__ == '__main__':
    main()
