import matplotlib.pyplot as plt

# 数据
x = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220]
x = [20, 40, 60, 80, 100, 120, 140, 160]
# y1 = [0, 0, 0, 6.2, 6.71, 6.55, 6.85, 6.66, 6.54, 6.91, 6.86]
# y2 = [0, 0, 0, 55.82, 56.35, 56.82, 57.13, 57.53, 57.97, 58.91, 59.62]
y = [87.07, 87.59, 87.51, 87.3, 87.17, 86.99, 86.92, 86.6]
# y = [24.87, 24.04, 21.15, 18.95, 17.45, 15.89, 15.95, 13.93]

# 创建折线图
plt.figure(figsize=(8, 5))
plt.plot(x, y, marker='o', linestyle='-', color='b', label='Comet')

# 添加标题和标签
plt.title('Comet')
plt.xlabel('Training Step')
plt.ylabel('Comet Score')

# 添加网格和图例
plt.grid(True)
plt.legend()

# 显示图形
plt.show()
plt.savefig('comet_val.png')
