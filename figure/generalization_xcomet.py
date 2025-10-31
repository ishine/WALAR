import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置全局样式 - 学术论文优化
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 使用更简洁的样式
sns.set_style("white")

# 模型和语言方向
models = ['Baseline', 'En-X', 'Ar-X', 'Tr-X', 'Hi-X', 'Mixed']
language_directions = ['En-X', 'Ar-X', 'Tr-X', 'Hi-X']

# 显著增大字体尺寸
label_size = 32  # 从15增加到18
tick_size = 30  # 从15增加到16
legend_size = 18 # 从15调整到14（图例稍小以节省空间）
title_size = 32 # 保持20
bar_text_size=18

# 数据
data = np.array([
    [44.58, 37.87, 38.42, 40.96],     # Baseline
    [49.1, 40.78, 41.33, 44.07],   # En-X
    [48.76, 42.05, 41.85, 44.3],   # Ar-X
    [47.93, 40.6, 41.67, 43.75],    # Tr-X
    [48.36, 40.71, 41.52, 44.9],    # Hi-X
    [53.02, 45.2, 45.98, 47.83],   # Mixed
])

# 创建图形 - 调整尺寸比例，减少高度以去除多余留白
fig, ax = plt.subplots(figsize=(13, 7), dpi=300)  # 宽度增加，高度减少

# 优化颜色方案 - 提高对比度
# colors = ['#8B7355', '#2E5C8A', '#D9534F', '#5CB85C', '#F0AD4E', '#5E4FA2']
colors = ['#937860', '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

# 调整柱状图宽度
bar_width = 0.13
x_pos = np.arange(len(language_directions))

# 绘制柱状图
bars = []
for i, model in enumerate(models):
    bar = ax.bar(x_pos + i * bar_width, data[i], bar_width, 
                 label=model, color=colors[i], edgecolor='black', 
                 linewidth=0.8, alpha=0.95, zorder=3)
    bars.append(bar)

# 优化数值标签 - 增大字体并调整位置
for i, model_data in enumerate(data):
    for j, value in enumerate(model_data):
        vertical_offset = 0.3
        # ax.text(j + i * bar_width, value + vertical_offset, f'{value:.1f}', 
        #         ha='center', va='bottom', fontsize=bar_text_size, fontweight='bold',
        #         color='black')  # 改为黑色提高可读性

# 设置坐标轴和标签
ax.set_xlabel('Language Directions', fontsize=label_size, fontweight='bold', labelpad=12)
ax.set_ylabel('XComet', fontsize=label_size, fontweight='bold', labelpad=12)
ax.set_title('Generalization of QE-RL', 
             fontsize=title_size, fontweight='bold', pad=15)

# 设置x轴刻度
ax.set_xticks(x_pos + bar_width * (len(models) - 1) / 2)
ax.set_xticklabels(language_directions, fontsize=tick_size, fontweight='bold')

# 设置y轴 - 从4开始，增大刻度标签
ax.set_yticks(np.arange(30, 60, 5))
ax.tick_params(axis='y', labelsize=tick_size)

# 优化图例 - 放在图形内部右上角以节省空间
# ax.legend(loc='upper right', fontsize=legend_size, 
#           frameon=True, fancybox=True, shadow=False, framealpha=0.95,
#           edgecolor='black')

# 优化网格线 - 只保留y轴网格线
ax.grid(True, axis='y', alpha=0.4, linestyle='-', linewidth=0.6)
ax.grid(False, axis='x')

# 设置y轴范围
ax.set_ylim(30, 60)

# 设置背景色
ax.set_facecolor('#FFFFFF')
fig.patch.set_facecolor('white')

# 美化边框
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
    spine.set_color('#2C3E50')

# 关键改进：去除边界留白，使用紧凑布局
plt.tight_layout(pad=0.0)  # 减少padding

# 保存为高分辨率图片，去除所有多余留白
plt.savefig('/mnt/gemini/data1/yifengliu/qe-lr/figure/generalization_xcomet.png', 
            dpi=300, 
            bbox_inches='tight',  # 紧密边界
            pad_inches=0.0)       # 减少内边距

# 显示图形
# plt.show()