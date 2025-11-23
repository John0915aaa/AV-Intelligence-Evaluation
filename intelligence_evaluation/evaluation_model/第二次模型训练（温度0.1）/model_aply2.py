import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

nn_layers = 4
layer_1 = 40
layer_2 = 40
layer_3 = 20
layer_4 = 10
active_fun = 'ReLU'
layer_4_fun = 'Sigmoid'
learning_rate = 0.0001
batch_size = 128
epochs = 100

# 1. 定义MLP模型类（与训练时一致）
class MLPModel(torch.nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, layer_1)
        self.fc2 = torch.nn.Linear(layer_1, layer_2)
        self.fc3 = torch.nn.Linear(layer_2, layer_3)
        self.fc4 = torch.nn.Linear(layer_3, layer_4)
        self.fc5 = torch.nn.Linear(layer_4, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x

# 2. 加载模型
model = MLPModel()
model_save_path = '训练模型/mlp_model_2.pth'  # 模型路径

# 加载训练好的模型权重
model.load_state_dict(torch.load(model_save_path))
model.eval()  # 将模型设置为评估模式

# 3. 加载 train_data_modified 数据，用于标准化参数
train_data_modified = pd.read_csv('train_data.csv')  # 使用 train_data_modified 来标准化
X_train_modified = train_data_modified[['TTC', 'PET', 'a_p', 'a_l', 'v_l', 'yaw_rate', 'task_time', 'avg_delay', 'IO', 'impact']].values

# 4. 创建一个 StandardScaler，并用 train_data_modified 的均值和标准差进行标准化
scaler = StandardScaler()
scaler.fit(X_train_modified)  # 用 train_data_modified 来拟合 scaler

# 5. 加载待预测的 metric_score 数据
metric_score_data = pd.read_csv('train_data.csv')  # 读取待预测的数据
X_metric_score = metric_score_data[['TTC', 'PET', 'a_p', 'a_l', 'v_l', 'yaw_rate', 'task_time', 'avg_delay', 'IO', 'impact']].values

# 6. 使用 train_data_modified 数据的参数对 metric_score 数据进行标准化
X_metric_score_scaled = scaler.transform(X_metric_score)  # 使用 train_data_modified 的参数进行标准化

# 7. 将标准化后的数据转换为 PyTorch 张量
X_metric_score_tensor = torch.tensor(X_metric_score_scaled, dtype=torch.float32)

# 8. 使用模型进行预测
with torch.no_grad():  # 不计算梯度
    y_pred = model(X_metric_score_tensor)

# 9. 将预测结果与输入数据一起保存到结果文件
metric_score_data['predicted_score'] = y_pred.numpy()  # 将预测结果作为新列添加到 metric_score 数据中

# 10. 保存到 CSV 文件
results_file = f'训练结果/evaluation_score_2.csv'
metric_score_data.to_csv(results_file, index=False)  # 将结果保存到 CSV 文件中
print(f'Results saved to {results_file}')
