import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 目标检测模型
class SimpleObjectDetector(nn.Module):
    def __init__(self):
        super(SimpleObjectDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 3 -> 16 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 16 -> 32 channels
        # 修改这里的输入大小，确保与前面的卷积输出匹配
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)  # 假设有10个类别
        self.fc3 = nn.Linear(128, 2)    # 输出二维坐标

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 64x64 -> 32x32
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 32x32 -> 16x16
        x = x.view(x.size(0), -1)  # Flatten
        class_out = self.fc2(F.relu(self.fc1(x)))
        coord_out = self.fc3(F.relu(self.fc1(x)))
        return class_out, coord_out

# RNN模型用于定向步骤
class SimpleRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=2):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        last_time_step = out[:, -1, :]
        output = self.fc(last_time_step)
        return output

# 决策模型
class DecisionMaker(nn.Module):
    def __init__(self, input_size=5, hidden_size=16, output_size=3):
        super(DecisionMaker, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # 1. 观察步骤
    start_observe_time = time.time()
    
    # 创建一个随机图像 (3通道，64x64)
    random_image = torch.rand((1, 3, 64, 64))  # batch size = 1
    detector = SimpleObjectDetector()
    
    # 目标检测
    class_scores, coordinates = detector(random_image)
    predicted_classes = torch.argmax(class_scores, dim=1)
    
    observe_time = time.time() - start_observe_time

    print("Observed Classes:", predicted_classes.numpy())
    print("Observed Coordinates:", coordinates.detach().numpy())
    print("Time taken for Observation step: {:.6f} seconds".format(observe_time))

    # 2. 定向步骤
    start_orient_time = time.time()

    # 模拟输入序列 (batch_size=1, seq_length=10, input_size=2)
    input_sequence = torch.rand((1, 10, 2))  # 模拟前10秒的二维坐标
    rnn_model = SimpleRNN()
    
    # 定向
    next_position = rnn_model(input_sequence)
    
    orient_time = time.time() - start_orient_time

    print("Predicted Next Position:", next_position.detach().numpy())
    print("Time taken for Orientation step: {:.6f} seconds".format(orient_time))

    # 3. 决策步骤
    start_decision_time = time.time()
    
    enemy_info = torch.rand((1, 5))  # 模拟敌方单位信息 (batch_size=1, input_size=5)
    decision_model = DecisionMaker()
    
    # 决策
    action_scores = decision_model(enemy_info)
    action_probabilities = F.softmax(action_scores, dim=1)
    decision = torch.argmax(action_probabilities, dim=1).numpy()
    
    decision_time = time.time() - start_decision_time

    actions = ["Attack", "Defend", "Retreat"]  # 可能的行动决策
    print("Predicted Action Decision:", actions[decision[0]])
    print("Action Probabilities:", action_probabilities.detach().numpy())
    print("Time taken for Decision step: {:.6f} seconds".format(decision_time))

    # 4. 总时间记录
    total_time = observe_time + orient_time + decision_time
    print("Total Time taken for OODA Loop: {:.6f} seconds".format(total_time))

if __name__ == "__main__":
    main()
