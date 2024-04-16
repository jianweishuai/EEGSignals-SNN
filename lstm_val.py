import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def create_dense_matrix(df, rows=1024, cols=30, offset=0):
    matrix = np.zeros((rows, cols), dtype=np.uint8)
    for _, row in df.iterrows():
        x = int(row['X']) - 1
        y = int(row['Y']) - 1 - offset
        if 0 <= y < cols:
            matrix[x, y] = row['Activity']
    return matrix.T


def calculate_metrics(predictions, targets, threshold=0.4):
    predictions = predictions.squeeze().cpu().numpy()
    targets = targets.squeeze().cpu().numpy()

    predictions_binary = (predictions >= threshold).astype(np.int32)
    targets_binary = (targets >= threshold).astype(np.int32)

    TP = np.sum((predictions_binary == 1) & (targets_binary == 1))
    FP = np.sum((predictions_binary == 1) & (targets_binary == 0))
    TN = np.sum((predictions_binary == 0) & (targets_binary == 0))
    FN = np.sum((predictions_binary == 0) & (targets_binary == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, accuracy, f1_score


def load_and_predict(test_df, model, device, rows=1024, cols=30):
    grouped = test_df.groupby('Group')
    metrics = []
    best_metrics = {'Precision': (-1, None, None),
                    'Recall': (-1, None, None),
                    'Accuracy': (-1, None, None),
                    'F1 Score': (-1, None, None)}

    for idx, (group_name, group) in enumerate(grouped):
        print(f'Processing group: {group_name}')
        group_input = group[group['Y'] <= 30]
        group_output = group[(group['Y'] > 30) & (group['Y'] <= 60)]

        dense_matrix_input = create_dense_matrix(group_input, rows, cols)
        dense_matrix_output = create_dense_matrix(group_output, rows, cols, offset=30)

        X_test = torch.tensor(np.expand_dims(dense_matrix_input, axis=0), dtype=torch.float32).to(device)
        y_test = torch.tensor(np.expand_dims(dense_matrix_output, axis=0), dtype=torch.float32).to(device)

        with torch.no_grad():
            predictions = model(X_test)

        predictions = torch.sigmoid(predictions)
        predictions_binary = (predictions >= 0.4).float()

        precision, recall, accuracy, f1_score = calculate_metrics(predictions_binary, y_test, threshold=0.4)
        metrics.append((precision, recall, accuracy, f1_score))

        for metric_name, value in zip(['Precision', 'Recall', 'Accuracy', 'F1 Score'],
                                      [precision, recall, accuracy, f1_score]):
            if value > best_metrics[metric_name][0]:
                original_matrix = y_test.squeeze().cpu().numpy()
                predicted_matrix = predictions.squeeze().cpu().numpy()
                best_metrics[metric_name] = (value, group_name, (original_matrix, predicted_matrix))

        print(
            f'Group: {group_name}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1_score:.4f}')

    metrics = np.array(metrics)
    average_metrics = metrics.mean(axis=0)
    print(
        f'Average Metrics - Precision: {average_metrics[0]:.4f}, Recall: {average_metrics[1]:.4f}, Accuracy: {average_metrics[2]:.4f}, F1 Score: {average_metrics[3]:.4f}')

    for metric_name, (metric_value, group_name, matrices) in best_metrics.items():
        if group_name:
            print(
                f'Plotting matrices for the best group by {metric_name}: {group_name} with {metric_name}: {metric_value:.4f}')
            plot_matrices(*matrices, metric_name, f'{group_name} ({metric_name})')


def plot_matrices(original, predicted, metric_name, group_name):
    save_path = 'lstm_predictions'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    original_binary = (original >= 0.4).astype(int)
    predicted_binary = (predicted >= 0.4).astype(int)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    cmap = 'gray_r'
    ax[0].imshow(original_binary, aspect='auto', cmap=cmap)
    ax[0].set_title(f'Original Matrix for Group {group_name}')
    ax[1].imshow(predicted_binary, aspect='auto', cmap=cmap)
    ax[1].set_title(f'Predicted Matrix for Group {group_name}')

    # 构造保存文件的名称
    filename = os.path.join(save_path, f"{group_name}_{metric_name}_comparison.png")
    plt.savefig(filename)
    plt.close()


test_df = pd.read_csv('test_312121.csv')

input_dim = 1024
hidden_dims = [1024, 512]
output_dim = 1024
dropout_rate = 0.4

model = DeepLSTMModel(input_dim, hidden_dims, output_dim, dropout_rate).to(device)
model.load_state_dict(torch.load('LSTM/model_checkpoint_epoch_30.pth', map_location=device))
model.eval()

load_and_predict(test_df, model, device)
