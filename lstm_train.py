"""train_312121"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

df = pd.read_csv('train_312121.csv')


def create_dense_matrix(df, rows=1024, cols=30, offset=0):
    """根据给定的DataFrame创建一个密集矩阵"""
    matrix = np.zeros((rows, cols), dtype=np.uint8)
    num = 0
    for _, row in df.iterrows():
        x = int(row['X']) - 1
        y = int(row['Y']) - 1 - offset
        if 0 <= y < cols:
            matrix[x, y] = row['Activity']
            if row['Activity'] == 1:
                num += 1
    return matrix.T, num


class DeepLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate):
        super(DeepLSTMModel, self).__init__()
        self.hidden_dims = hidden_dims

        self.lstm1 = nn.LSTM(input_dim, hidden_dims[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        h0_1 = torch.zeros(1, x.size(0), self.hidden_dims[0]).to(x.device)
        c0_1 = torch.zeros(1, x.size(0), self.hidden_dims[0]).to(x.device)

        out1, _ = self.lstm1(x, (h0_1, c0_1))

        h0_2 = torch.zeros(1, x.size(0), self.hidden_dims[1]).to(x.device)
        c0_2 = torch.zeros(1, x.size(0), self.hidden_dims[1]).to(x.device)

        out2, _ = self.lstm2(out1, (h0_2, c0_2))

        out = self.dropout(out2)
        predictions = self.fc(out)
        return predictions


def batch_generator(df, batch_size=64):
    grouped = df.groupby('Group')
    groups = [group for _, group in grouped]
    np.random.shuffle(groups)
    for i in range(0, len(groups), batch_size):
        yield groups[i:i + batch_size]


def train_model(model, df, optimizer, criterion, scheduler, epochs=50, batch_size=128):
    """训练模型，支持批量处理"""
    for epoch in range(epochs):
        print(f'Starting Epoch {epoch + 1}/{epochs}')
        df_shuffled = df.sample(frac=1).reset_index(drop=True)

        batch_counter = 0
        for batch_groups in batch_generator(df_shuffled, batch_size):
            batch_counter += 1
            print(f'  Processing batch {batch_counter}')
            datasets = []
            for group in batch_groups:
                df_input = group[group['Y'] < 30]
                df_output = group[(group['Y'] >= 30) & (group['Y'] < 60)]

                dense_matrix_input, input_num = create_dense_matrix(df_input)
                dense_matrix_output, output_num = create_dense_matrix(df_output, cols=30, offset=30)

                # print(f'    Input Num: {input_num}, Group Num: {output_num}')

                X = torch.tensor(np.expand_dims(dense_matrix_input, axis=0), dtype=torch.float32)
                y = torch.tensor(np.expand_dims(dense_matrix_output, axis=0), dtype=torch.float32)

                dataset = TensorDataset(X, y)
                datasets.append(dataset)

            model.train()
            for dataset in datasets:
                data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
                for X, y in data_loader:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    output = model(X)
                    loss = criterion(output.view(-1), y.view(-1))
                    loss.backward()
                    optimizer.step()
            print(f'    Finished batch {batch_counter}, Loss: {loss.item()}')
        scheduler.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()}')

        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'LSTM/model_checkpoint_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model saved to {checkpoint_path} at epoch {epoch + 1}')

        if device.type == 'cuda':
            torch.cuda.empty_cache()


# 模型参数配置
input_dim = 1024
hidden_dims = [1024, 512]
output_dim = 1024
dropout_rate = 0.4

model = DeepLSTMModel(input_dim, hidden_dims, output_dim, dropout_rate).to(device)

# 正样本的权重
adjusted_pos_weight = 30
pos_weight = torch.tensor([adjusted_pos_weight]).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=2, gamma=0.9)

# 训练模型
train_model(model, df, optimizer, criterion, scheduler, epochs=200)
