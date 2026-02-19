import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt
import numpy as np

# 添加数据可视化
# 原始数据
x = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [[3], [7], [11], [15]]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# 转换为张量
X = torch.tensor(x, dtype=torch.float32).to(device)
Y = torch.tensor(y, dtype=torch.float32).to(device)

class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(2, 8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8, 1)
    
    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x

# 实例化模型
mynet = MyNeuralNet().to(device)
print("Model architecture:")
print(mynet)
print("\nInitial weights of input_to_hidden_layer:")
print(mynet.input_to_hidden_layer.weight)
print(f"Weight shape: {mynet.input_to_hidden_layer.weight.shape}")

# 损失函数和优化器
loss_function = nn.MSELoss()
opt = SGD(mynet.parameters(), lr=0.001)  # 降低了学习率以获得更好的训练稳定性

# 初始预测
Y_pred = mynet(X)
initial_loss = loss_function(Y_pred, Y)
print(f'\nInitial Loss: {initial_loss.item():.6f}')

# 添加训练循环
num_epochs = 1000
loss_history = []

print("\nTraining started...")
for epoch in range(num_epochs):
    # 前向传播
    Y_pred = mynet(X)
    
    # 计算损失
    loss = loss_function(Y_pred, Y)
    
    # 反向传播
    opt.zero_grad()  # 清除之前的梯度
    loss.backward()  # 计算梯度
    opt.step()  # 更新参数
    
    # 记录损失
    loss_history.append(loss.item())
    
    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# 最终评估
mynet.eval()
with torch.no_grad():
    final_predictions = mynet(X)
    final_loss = loss_function(final_predictions, Y)
    
print(f'\nFinal Loss: {final_loss.item():.6f}')
print("\nPredictions vs Actual:")
for i in range(len(X)):
    print(f"Input: {X[i].cpu().numpy()}, Predicted: {final_predictions[i].item():.4f}, Actual: {Y[i].item()}")

# 可视化训练过程
plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# 预测 vs 实际
plt.subplot(1, 2, 2)
actual_values = Y.cpu().numpy().flatten()
predicted_values = final_predictions.cpu().numpy().flatten()
x_indices = np.arange(len(actual_values))

plt.scatter(x_indices, actual_values, label='Actual', s=100, alpha=0.7)
plt.scatter(x_indices, predicted_values, label='Predicted', s=100, alpha=0.7)
for i in range(len(actual_values)):
    plt.plot([i, i], [actual_values[i], predicted_values[i]], 'r--', alpha=0.5)

plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Predictions vs Actual Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(x_indices)

plt.tight_layout()
plt.show()

# 保存模型
torch.save(mynet.state_dict(), 'mynet_model.pth')
print("\nModel saved as 'mynet_model.pth'")