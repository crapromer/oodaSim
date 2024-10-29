import pygame
import sys
import math
import random
from sklearn.linear_model import LinearRegression
import numpy as np

# 初始化Pygame
pygame.init()

# 设置窗口大小
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("场景")

# 加载并调整图像大小
boat_image = pygame.image.load('boat.png')
boat_image = pygame.transform.scale(boat_image, (80, 80))

arctic_image = pygame.image.load('arctic.png')
arctic_image = pygame.transform.scale(arctic_image, (80, 80))

drone_image = pygame.image.load('drone.png')
drone_image = pygame.transform.scale(drone_image, (30, 30))

missile_image = pygame.image.load('missile.png')
missile_image = pygame.transform.scale(missile_image, (30, 30))

# 定义物体的位置
boat_pos = [100, 300]
arctic_pos = [400, 250]
drone_pos = [random.randint(50, WIDTH-50), random.randint(50, HEIGHT-50)]
drone_angle = random.uniform(0, 2 * math.pi)  # 初始角度
drone_speed = 2

# 导弹列表
missiles = []
missile_launched = [False, False]  # 分别记录船和舰的发射状态

# 发射导弹的距离
launch_distance = 150

# 预测时间
prediction_time = 1.0  # 预测1秒后的位置

# 标记列表
hit_marks = []

# 记录无人机历史位置
drone_history = []

clock = pygame.time.Clock()

def predict(drone_history):
    """使用线性回归预测无人机未来的位置"""
    if len(drone_history) < 2:
        return None

    # 提取x和y坐标
    x_coords = np.array([pos[0] for pos in drone_history]).reshape(-1, 1)
    y_coords = np.array([pos[1] for pos in drone_history]).reshape(-1, 1)

    # 使用线性回归拟合
    model_x = LinearRegression().fit(np.arange(len(drone_history)).reshape(-1, 1), x_coords)
    model_y = LinearRegression().fit(np.arange(len(drone_history)).reshape(-1, 1), y_coords)

    # 预测未来位置
    future_index = len(drone_history) + prediction_time / (1000 / 60)  # 根据帧率预测
    predicted_x = model_x.predict([[future_index]])[0][0]
    predicted_y = model_y.predict([[future_index]])[0][0]

    return predicted_x, predicted_y

# 主循环
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # 只有在无人机存在时才更新位置
    if drone_pos is not None:
        drone_pos[0] += drone_speed * math.cos(drone_angle)
        drone_pos[1] += drone_speed * math.sin(drone_angle)

        # 随机改变方向
        if random.random() < 0.1:  # 10%的概率改变方向
            drone_angle += random.uniform(-math.pi / 4, math.pi / 4)  # 在-45度到45度之间改变

        # 边界检测
        if drone_pos[0] <= 0 or drone_pos[0] >= WIDTH - 30:
            drone_angle = math.pi - drone_angle  # 反转x方向
        if drone_pos[1] <= 0 or drone_pos[1] >= HEIGHT - 30:
            drone_angle = -drone_angle  # 反转y方向

        # 记录无人机位置
        drone_history.append(drone_pos.copy())
        if len(drone_history) > 10:  # 限制历史记录的长度
            drone_history.pop(0)

    # 绘制场景
    screen.fill((135, 206, 250))  # 天空蓝
    screen.blit(boat_image, boat_pos)
    screen.blit(arctic_image, arctic_pos)

    if drone_pos is not None:
        screen.blit(drone_image, drone_pos)

    pygame.draw.circle(screen, (255, 0, 0, 50), (boat_pos[0] + 40, boat_pos[1] + 40), launch_distance, 1)  # 船的探测范围
    pygame.draw.circle(screen, (0, 0, 255, 50), (arctic_pos[0] + 40, arctic_pos[1] + 40), launch_distance, 1)  # 舰的探测范围

    # 预测无人机位置
    predicted_drone_pos = predict(drone_history)

    # 检查距离并发射导弹
    for i, position in enumerate([boat_pos, arctic_pos]):
        if predicted_drone_pos is not None and not missile_launched[i]:  # 如果未发射导弹
            distance = math.sqrt((predicted_drone_pos[0] - position[0]) ** 2 + (predicted_drone_pos[1] - position[1]) ** 2)
            if distance < launch_distance:
                # 计算导弹发射方向
                angle = math.atan2(predicted_drone_pos[1] - position[1], predicted_drone_pos[0] - position[0])
                missile_dx = 5 * math.cos(angle)
                missile_dy = 5 * math.sin(angle)
                missiles.append([position[0], position[1], missile_dx, missile_dy, angle])
                missile_launched[i] = True  # 设置为已发射

    # 更新导弹位置
    for missile in missiles[:]:
        missile[0] += missile[2]
        missile[1] += missile[3]
        
        if drone_pos is not None:
            drone_rect = pygame.Rect(drone_pos[0], drone_pos[1], 30, 30)
            missile_rect = pygame.Rect(missile[0], missile[1], 30, 30)
            
            if missile_rect.colliderect(drone_rect):
                # 导弹击中无人机
                hit_marks.append((drone_pos[0] + 15, drone_pos[1] + 15))  # 记录击中位置
                drone_pos = None  # 移除无人机
                missiles.remove(missile)  # 移除导弹
                
                # 重置发射状态
                missile_launched = [False, False]  # 重置船和舰的发射状态
                break  # 退出循环，避免修改列表时出现问题

        # 如果导弹飞出屏幕，移除它
        if not (0 <= missile[0] <= WIDTH and 0 <= missile[1] <= HEIGHT):
            missiles.remove(missile)
            missile_launched = [False, False]  # 重置发射状态

    # 绘制导弹
    for missile in missiles:
        missile_rotated = pygame.transform.rotate(missile_image, -math.degrees(missile[4]) - 90)  # 旋转导弹图像
        missile_rect = missile_rotated.get_rect(center=(missile[0], missile[1]))  # 计算中心位置
        screen.blit(missile_rotated, missile_rect.center)  # 绘制旋转后的导弹

    # 绘制击中标记
    for mark in hit_marks:
        pygame.draw.circle(screen, (255, 0, 0), mark, 10)  # 用红色标记击中位置

    # 更新显示
    pygame.display.flip()
    clock.tick(60)
