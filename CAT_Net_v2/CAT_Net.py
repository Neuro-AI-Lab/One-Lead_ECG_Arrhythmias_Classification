import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel Attention Mechanism
class ChannelAttention(nn.Module):
    def __init__(self, filters, ratio):
        super(ChannelAttention, self).__init__()
        self.shared_layer_one = nn.Sequential(
            nn.Linear(filters, filters // ratio),
            nn.ReLU()
        )
        self.shared_layer_two = nn.Linear(filters // ratio, filters)
        
    
    def forward(self, x):
        # AvgPool
        avg_pool = torch.mean(x, dim=-1, keepdim=False)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        # MaxPool
        max_pool = torch.max(x, dim=-1, keepdim=False).values
        #print('max_pool shape: ', max_pool.shape)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        # Combine
        attention = torch.sigmoid(avg_pool + max_pool)
        attention = attention.unsqueeze(-1)
        return x * attention


# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, d_model, dff, dropout_rate):
        super(TransformerEncoder, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )

    def forward(self, x):
        attn_output, _ = self.multi_head_attention(x, x, x)
        attn_output = self.dropout(attn_output)
        out1 = self.layer_norm(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output)
        out2 = self.layer_norm(out1 + ffn_output)

        return out2


# Positional Encoding
def positional_encoding(seq_len, d, n=10000):
    P = torch.zeros(d, seq_len)
    for k in range(seq_len):
        for i in range(0, d // 2):
            denominator = torch.tensor(n ** (2 * i / d))
            P[2 * i, k] = torch.sin(k / denominator)
            P[2 * i + 1, k] = torch.cos(k / denominator)
    return P


# Main Model
class CAT_Net(nn.Module):
    def __init__(self, max_sequence_length, num_channels, d_model, num_heads, dff, dropout_rate, num_classes):
        super(CAT_Net, self).__init__()
        
        self.cnn_block1 = self.cnn_block(num_channels, 16, 21, 1, 10)
        self.cnn_block2 = self.cnn_block(16, 32, 23, 1, 11)
        self.cnn_block3 = self.cnn_block(32, 64, 25, 1, 12)
        self.cnn_block4 = self.cnn_block(64, 128, 27, 1, 13)
        
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Positional Encoding
        self.pos_encoding = positional_encoding(32, 128).unsqueeze(0)

        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(num_heads=num_heads, d_model=d_model, dff=dff, dropout_rate=dropout_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.2)
        
    def cnn_block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=k, stride=s, padding=p),
            nn.ReLU(),
            nn.BatchNorm1d(out_c),
            ChannelAttention(out_c, 8),
        )

    def forward(self, x):

        x = self.cnn_block1(x)
        x = self.pool1(x)

        #print('first conv layer: ', x.shape)
        
        x = self.cnn_block2(x)
        x = self.pool2(x)

        #print('second1 conv layer: ', x.shape)

        x = self.cnn_block3(x)
        x = self.pool3(x)

        #print('third conv layer: ', x.shape)

        x = self.cnn_block4(x)
        

        #print('fourth conv layer: ', x.shape)

        # Add positional encoding
        
        PE = self.pos_encoding.to(x.device)
        x = x + PE

        # print('position embedding: ', PE.shape)

        # Transformer Encoder
        x = x.transpose(1, 2).contiguous()  # To match the input shape (seq_len, batch_size, features)
        x = self.transformer_encoder(x)

        #print('Transformer: ', x.shape)

        x = x.flatten(start_dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        
        # Output
        output = self.softmax(self.fc2(x))
        return output