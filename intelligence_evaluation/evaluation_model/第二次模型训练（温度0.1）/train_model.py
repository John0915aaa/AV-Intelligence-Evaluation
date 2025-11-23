import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt  # 导入matplotlib库用于可视化
import os
import shap


# 0. 参数定义
nn_layers = 4
layer_1 = 40
layer_2 = 40
layer_3 = 20
layer_4 = 10
active_fun = 'ReLU'
layer_4_fun = 'Sigmoid'
lerning_rate = 0.0001
batch_size = 128
epochs = 125

num = 9
model_num = 9

# 1. 加载数据
data = pd.read_csv('train_data.csv')

# 2. 提取输入特征和输出标签
X = data[['TTC', 'PET', 'a_p', 'a_l', 'v_l', 'yaw_rate', 'task_time', 'avg_delay', 'IO', 'impact']].values
y = data['score'].values

# 3. 数据标准化（先进行标准化处理）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 对X进行标准化

# 4. 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# 5. 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # 将y转换为列向量
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# 6. 定义MLP模型
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(10, layer_1)  # 输入层到第一层（64个神经元）
        self.fc2 = nn.Linear(layer_1, layer_2)  # 第一层到第二层（32个神经元）
        self.fc3 = nn.Linear(layer_2, layer_3)  # 第二层到第三层（16个神经元）
        self.fc4 = nn.Linear(layer_3, layer_4)   # 第三层到第四层（8个神经元）
        self.fc5 = nn.Linear(layer_4, 1)    # 第四层到输出层（1个神经元，回归任务）
        
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数，用于输出层
    
    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一层
        x = self.relu(self.fc2(x))  # 第二层
        x = self.relu(self.fc3(x))  # 第三层
        x = self.relu(self.fc4(x))  # 第四层
        x = self.sigmoid(self.fc5(x))  # 输出层使用Sigmoid
        return x


# 7. 初始化模型
model = MLPModel()

# 8. 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.AdamW(model.parameters(), lr=lerning_rate)  # Adam优化器

# 用于存储每个epoch的训练损失和验证损失
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清零梯度

    # 随机选择一个批次的数据
    indices = torch.randperm(X_train_tensor.size(0))  # 打乱数据
    for i in range(0, X_train_tensor.size(0), batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = X_train_tensor[batch_indices]
        y_batch = y_train_tensor[batch_indices]
        
        # 前向传播
        y_pred = model(X_batch)
        
        # 计算损失
        loss = criterion(y_pred, y_batch)
        
        # 反向传播
        loss.backward()
        
        # 更新权重
        optimizer.step()
    
    # 计算训练损失并记录
    train_losses.append(loss.item())

    # 计算验证损失
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():
        y_val_pred = model(X_val_tensor)
        val_loss = mean_squared_error(y_val_tensor.numpy(), y_val_pred.numpy())
        val_losses.append(val_loss)

    # 每50轮打印一次训练损失
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

# 保存训练过程和效果的损失文件
train_loss_df = pd.DataFrame({
    'Epoch': range(1, epochs+1),
    'Train Loss': train_losses,
    'Validation Loss': val_losses
})

# 保存训练损失和验证损失到CSV文件
train_loss_file = f'训练效果/csv文件/training_process_{num}.csv'
train_loss_df.to_csv(train_loss_file, index=False)
print(f'Training and validation losses saved to {train_loss_file}')

# 1. 训练结束后，保存模型
model_save_path = f'训练模型/mlp_model_{model_num}.pth'  # 模型保存路径
torch.save(model.state_dict(), model_save_path)  # 保存模型权重

print(f'Model saved to {model_save_path}')

# 10. 创建保存图片的文件夹（如果不存在的话）
output_dir = '训练效果'
train_loss_dir = os.path.join(output_dir, '训练过程')
effect_dir = os.path.join(output_dir, '效果')

# 创建文件夹
os.makedirs(train_loss_dir, exist_ok=True)
os.makedirs(effect_dir, exist_ok=True)

# 11. 可视化训练集和验证集损失并保存到图片文件
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss', color='blue')
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss', color='red')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 添加神经网络参数到图形上
network_info = (
    f"Layers: 5 (10 -> {layer_1} -> {layer_2} -> {layer_3} -> {layer_4} -> 1)\n"
    f"Activation Functions: {active_fun} (except output: {layer_4_fun})\n"
    f"Learning Rate: {lerning_rate}\n"
    f"Batch Size: {batch_size}"
)
plt.figtext(0.5, 0.95, network_info, ha="center", va="top", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "pad": 10})

# 保存图形到“训练过程”文件夹
train_loss_img_file = os.path.join(train_loss_dir, f'training_loss_plot_{num}.png')
plt.savefig(train_loss_img_file, dpi=300)  # 以300 dpi保存为高清图像
plt.close()  # 关闭图形，释放内存

# 12. 可视化预测结果与真实值的对比并保存到图片文件
plt.figure(figsize=(10, 5))
plt.scatter(y_val, y_val_pred.numpy(), alpha=0.5, color='blue')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')  # 理想情况下的y = x线
plt.title('Validation Predictions vs True Values')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.grid(True)

# 添加神经网络参数到图形上
plt.figtext(0.5, 0.95, network_info, ha="center", va="top", fontsize=10, bbox={"facecolor": "white", "alpha": 0.8, "pad": 10})

# 保存图形到“效果”文件夹
effect_file = os.path.join(effect_dir, f'validation_predictions_vs_true_values_{num}.png')
plt.savefig(effect_file, dpi=300)  # 以300 dpi保存为高清图像
plt.close()  # 关闭图形，释放内存

# 保存预测结果与真实值对比的CSV文件
predictions_df = pd.DataFrame({
    'True Values': y_val,
    'Predictions': y_val_pred.numpy().flatten()
})

predictions_file = f'训练效果/csv文件/predictions_vs_true_values_{num}.csv'
predictions_df.to_csv(predictions_file, index=False)
print(f'Predictions vs True Values saved to {predictions_file}')
