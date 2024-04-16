import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from spikingjelly.activation_based import functional
from spikingjelly.clock_driven import neuron, functional, surrogate, layer

torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 假设train_312121.csv已经准备好，此处不再重复
df = pd.read_csv('train_312121.csv')

def create_dense_matrix(df, rows=1024, cols=30, offset=0):
    matrix = np.zeros((rows, cols), dtype=np.float32)
    for _, row in df.iterrows():
        x = int(row['X']) - 1
        y = int(row['Y']) - 1 - offset
        if 0 <= y < cols:
            matrix[x, y] = 1
    return matrix.T

def rate_encode(matrix, time_steps=30):
    spike_sequences = np.zeros((time_steps, *matrix.shape), dtype=np.float32)
    for t in range(time_steps):
        spike_sequences[t] = (np.random.rand(*matrix.shape) < matrix).astype(np.float32)
    return spike_sequences

def batch_generator(df, batch_size=64, time_steps=30):
    grouped = df.groupby('Group')
    groups = [group for _, group in grouped]
    np.random.shuffle(groups)
    for i in range(0, len(groups), batch_size):
        batch_groups = groups[i:i + batch_size]
        inputs, targets = [], []
        for group in batch_groups:
            df_input = group[group['Y'] < 30]
            df_output = group[(group['Y'] >= 30) & (group['Y'] < 60)]

            dense_input = create_dense_matrix(df_input)
            dense_output = create_dense_matrix(df_output, offset=30)

            encoded_input = rate_encode(dense_input, time_steps)
            encoded_output = rate_encode(dense_output, time_steps)

            inputs.append(torch.tensor(encoded_input, dtype=torch.float32))
            targets.append(torch.tensor(encoded_output, dtype=torch.float32))

        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        yield TensorDataset(inputs, targets)

class DeepSNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, time_steps, dropout_rate=0.4):
        super(DeepSNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), tau=2.0)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate.ATan(), tau=2.0)

        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.lif3 = neuron.LIFNode(surrogate_function=surrogate.ATan(), tau=2.0)
        self.time_steps = time_steps

    def forward(self, x):
        outputs = []
        for t in range(self.time_steps):
            x_t = x[:, t]
            x = self.lif1(self.dropout1(self.fc1(x_t)))
            x = self.lif2(self.dropout2(self.fc2(x)))
            x = self.lif3(self.dropout3(self.fc3(x)))
            outputs.append(x.unsqueeze(1))
        out = torch.cat(outputs, dim=1)
        return out

def train_model(model, df, optimizer, criterion, scheduler, epochs=10, batch_size=128, time_steps=30):
    model.train()
    for epoch in range(epochs):
        print(f'Starting Epoch {epoch + 1}/{epochs}')
        total_loss = 0
        batch_counter = 0
        for dataset in batch_generator(df, batch_size, time_steps):
            batch_counter += 1
            data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward(retain_graph=True)
                optimizer.step()
                total_loss += loss.item()
                functional.reset_net(model)
            print(f'  Batch {batch_counter}, Loss: {loss.item()}')
        scheduler.step()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / batch_counter}')

        checkpoint_path = f'SNN/model_checkpoint_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Model saved to {checkpoint_path} at epoch {epoch + 1}')

        if device.type == 'cuda':
            torch.cuda.empty_cache()

time_steps = 30
input_dim = 1024
hidden_dim1 = 1024
hidden_dim2 = 512
output_dim = 1024

model = DeepSNNModel(input_dim, hidden_dim1, hidden_dim2, output_dim, time_steps).to(device)

adjusted_pos_weight = 30
pos_weight = torch.tensor([adjusted_pos_weight]).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=2, gamma=0.9)

train_model(model, df, optimizer, criterion, scheduler, epochs=50, batch_size=128, time_steps=time_steps)
