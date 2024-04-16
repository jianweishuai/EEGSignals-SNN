import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, surrogate

# 设定使用的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


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


def rate_encode(matrix, time_steps=30):
    spike_sequences = np.zeros((time_steps, *matrix.shape), dtype=np.float32)
    for t in range(time_steps):
        spike_sequences[t] = (np.random.rand(*matrix.shape) < matrix).astype(np.float32)
    return spike_sequences


def create_dense_matrix(df, rows=1024, cols=30, offset=0):
    matrix = np.zeros((rows, cols), dtype=np.float32)
    for _, row in df.iterrows():
        x = int(row['X']) - 1
        y = int(row['Y']) - 1 - offset
        if 0 <= y < cols:
            matrix[x, y] = 1
    return matrix.T


# 读取测试数据集
test_df = pd.read_csv('test_312121.csv')

# 初始化模型
time_steps = 30
input_dim = 1024
hidden_dim1 = 1024
hidden_dim2 = 512
output_dim = 1024

model = DeepSNNModel(input_dim, hidden_dim1, hidden_dim2, output_dim, time_steps).to(device)

# 加载模型权重
model.load_state_dict(torch.load('SNN/model_checkpoint_epoch_1.pth', map_location=device))
model.eval()


def load_and_predict_snn(test_df, model, device, rows=1024, cols=30, time_steps=30):
    grouped = test_df.groupby('Group')
    metrics = []
    best_scores = {'precision': (0, None), 'recall': (0, None), 'accuracy': (0, None), 'f1_score': (0, None)}

    for idx, (group_name, group) in enumerate(grouped):
        print(f'Processing group: {group_name}')
        group_input = group[group['Y'] < 30]
        group_output = group[(group['Y'] >= 30) & (group['Y'] < 60)]

        dense_matrix_input = create_dense_matrix(group_input, rows, cols)
        dense_matrix_output = create_dense_matrix(group_output, rows, cols, offset=30)

        encoded_input = rate_encode(dense_matrix_input, time_steps)
        encoded_output = rate_encode(dense_matrix_output, time_steps)

        X_test = torch.tensor(encoded_input, dtype=torch.float32).unsqueeze(0).to(device)
        y_test = torch.tensor(encoded_output, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(X_test)

        predictions_last_step = predictions[:, -1, :, :]
        y_test_last_step = y_test[:, -1, :, :]

        precision, recall, accuracy, f1_score = calculate_metrics(predictions_last_step, y_test_last_step)
        metrics.append((precision, recall, accuracy, f1_score))

        scores = {'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1_score': f1_score}
        for metric in scores:
            if scores[metric] > best_scores[metric][0]:
                best_scores[metric] = (scores[metric], (
                    predictions_last_step.squeeze().cpu().numpy(), y_test_last_step.squeeze().cpu().numpy()))

        print(
            f'Group: {group_name}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1_score:.4f}')

    metrics = np.array(metrics)
    average_metrics = metrics.mean(axis=0)
    print(
        f'Average Metrics - Precision: {average_metrics[0]:.4f}, Recall: {average_metrics[1]:.4f}, '
        f'Accuracy: {average_metrics[2]:.4f}, F1 Score: {average_metrics[3]:.4f}')

    # 保存所有最佳得分的预测和目标到CSV
    for metric, (score, data) in best_scores.items():
        if data is not None:
            predictions, targets = data
            save_to_csv(predictions, targets, metric, score)

    for metric, (score, _) in best_scores.items():
        visualize_from_csv(metric, score)


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

def save_to_csv(predictions, targets, metric, score, output_folder='snn_predictions_csv'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 创建DataFrame
    predictions_df = pd.DataFrame(predictions)
    targets_df = pd.DataFrame(targets)

    # 保存为CSV文件
    predictions_csv_path = os.path.join(output_folder, f'predictions_{metric}_{score:.4f}.csv')
    targets_csv_path = os.path.join(output_folder, f'targets_{metric}_{score:.4f}.csv')
    predictions_df.to_csv(predictions_csv_path, index=False)
    targets_df.to_csv(targets_csv_path, index=False)
    print(f'{metric.capitalize()} predictions and targets saved to CSV.')

def visualize_from_csv(metric, score, output_folder='snn_predictions_csv', save_folder='snn_predictions'):
    predictions_csv_path = os.path.join(output_folder, f'predictions_{metric}_{score:.4f}.csv')
    targets_csv_path = os.path.join(output_folder, f'targets_{metric}_{score:.4f}.csv')

    predictions_df = pd.read_csv(predictions_csv_path)
    targets_df = pd.read_csv(targets_csv_path)

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(predictions_df.values, cmap='hot', interpolation='nearest', aspect='auto')
    plt.title(f'Predictions - {metric}: {score:.4f}')
    
    plt.subplot(1, 2, 2)
    plt.imshow(targets_df.values, cmap='hot', interpolation='nearest', aspect='auto')
    plt.title(f'Targets - {metric}: {score:.4f}')
    
    plt.suptitle(f'Best {metric.capitalize()}')
    
    # 确保保存目录存在
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # 保存图像到文件
    plt.savefig(os.path.join(save_folder, f'{metric}_best_{score:.4f}.png'))
    # 关闭图像以释放内存
    plt.close()


# 运行预测和评估
load_and_predict_snn(test_df, model, device, time_steps=time_steps)


