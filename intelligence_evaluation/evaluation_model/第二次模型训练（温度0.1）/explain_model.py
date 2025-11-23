import torch
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer

nn_layers = 4
layer_1 = 40
layer_2 = 40
layer_3 = 20
layer_4 = 10
active_fun = 'ReLU'
layer_4_fun = 'Sigmoid'
lerning_rate = 0.0001
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

# 3. 加载训练数据并标准化
train_data = pd.read_csv('train_data.csv')  # 训练数据路径
X_train = train_data[['TTC', 'PET', 'a_p', 'a_l', 'v_l', 'yaw_rate', 'task_time', 'avg_delay', 'IO', 'impact']].values

# 创建StandardScaler，并用训练数据的均值和标准差进行标准化
scaler = StandardScaler()
scaler.fit(X_train)

# 标准化训练数据
X_train_scaled = scaler.transform(X_train)

# 4. 查看模型的权重和偏置
print("\nModel's weights and biases:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name} | Shape: {param.shape} | Values: {param.data[:5]}")  # 打印部分权重和偏置的值

# 5. 使用SHAP来查看特征重要性
# 将X_train_scaled转换为 NumPy 数组（必要步骤）
X_train_numpy = X_train_scaled.astype(np.float32)

# 创建一个包装函数，将 numpy 数据转换为 torch.Tensor
def model_forward_numpy(input_data):
    input_tensor = torch.tensor(input_data, dtype=torch.float32)  # 转换为 tensor
    return model(input_tensor).detach().numpy()  # 调用模型并返回 NumPy 数组

# 使用 SHAP 进行解释
explainer = shap.KernelExplainer(model_forward_numpy, X_train_numpy)  # 使用转换后的模型
shap_values = explainer.shap_values(X_train_numpy[:10])  # 计算前10个样本的SHAP值

# 可视化SHAP值
shap.summary_plot(shap_values, X_train_numpy[:10], feature_names=['TTC', 'PET', 'a_p', 'a_l', 'v_l', 'yaw_rate', 'task_time', 'avg_delay', 'IO', 'impact'])

# 6. 使用LIME来查看特征对某个样本的贡献
# Wrap the model to allow predictions for LIME
def model_predict(input_data):
    input_tensor = torch.tensor(input_data, dtype=torch.float32)  # Convert to tensor
    with torch.no_grad():  # Disable gradient computation
        prediction = model(input_tensor).numpy()  # Get model prediction
    return prediction

# Use LIME to explain one sample
explainer = LimeTabularExplainer(X_train_scaled, training_labels=train_data['score'].values, mode="regression", feature_names=['TTC', 'PET', 'a_p', 'a_l', 'v_l', 'yaw_rate', 'task_time', 'avg_delay', 'IO', 'impact'])

# 解释一个样本
explanation = explainer.explain_instance(X_train_scaled[0], model_predict)
explanation.show_in_notebook()  # 在Jupyter中可视化LIME解释

# 7. 可视化神经网络的激活
def plot_activations(model, X):
    # 获取每一层的输出激活
    activations = []
    def hook_fn(module, input, output):
        activations.append(output)
    
    # 注册hook以捕获每一层的激活
    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    # 前向传播数据
    model(X)
    
    # 绘制激活图
    for i, activation in enumerate(activations):
        plt.figure(figsize=(10, 5))
        plt.title(f"Activation of Layer {i+1}")
        plt.imshow(activation.detach().numpy(), cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.show()

    # 移除hook
    for hook in hooks:
        hook.remove()

# 可视化模型的激活
plot_activations(model, torch.tensor(X_train_scaled[:5], dtype=torch.float32))

# 8. 输出每个输入特征的贡献
# SHAP特征贡献可通过SHAP值进行查看，已通过summary_plot展示。
