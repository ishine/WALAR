import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

# steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
# qe = [44.58, 55.95, 76.83, 77.18, 77.7, 72.75, 73.01, 85.31, 74.53, 74.39, 74.33, 74.47]
# qe_lang = [44.58, 47.58, 50.64, 51.95, 52.69, 53.57, 54.09, 54.51, 54.77, 55.08, 55.61, 55.51]
# qe_word_align_lang = [44.58, 47.16, 48.66, 49.02, 49.38, 50.37, 51.15, 51.88, 52.03, 54.59, 55.23, 53.38]


steps = [0, 200, 400, 600, 800, 1000]
qe = [71.71, 90.49, 90.36, 89.98, 89.33, 88.76]
qe_lang = [71.71, 78.39, 0, 80.33, 80.67]
qe_word_align_lang = [71.71, 77.13, 77.18, 76.92, 77.31, 77.23]
# 增大图形尺寸
fig, ax = plt.subplots(figsize=(17, 10))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
markers = ['o', 's', '^']
title_size=45
axis_label_size=42
tick_size=40
legend_size=29
linewidth=10
# 进一步加粗线条和标记
ax.plot(steps, qe, marker=markers[0], linewidth=linewidth, label='QE', 
         markersize=12, color=colors[0], markerfacecolor='white', markeredgewidth=2.5)
ax.plot(steps, qe_lang, marker=markers[1], linewidth=linewidth, label='QE + Lang', 
         markersize=12, color=colors[1], markerfacecolor='white', markeredgewidth=2.5)
ax.plot(steps, qe_word_align_lang, marker=markers[2], linewidth=linewidth, 
         label='QE + Word-alignment + Lang', markersize=12, color=colors[2], 
         markerfacecolor='white', markeredgewidth=2.5)

# 标题和标签 - 进一步增大字体
ax.set_title('XComet', fontsize=title_size, fontweight='bold', pad=20)
ax.set_xlabel('Training Steps', fontsize=axis_label_size, fontweight="bold", labelpad=15)
ax.set_ylabel('EN-X', fontsize=axis_label_size, fontweight="bold", labelpad=15)

# 坐标轴范围和刻度
ax.set_xlim(0, 1050)
ax.set_ylim(70, 100)
ax.set_xticks(np.arange(0, 1100, 200))
ax.set_xticklabels(np.arange(0, 1100, 200), fontsize=tick_size, fontweight='bold')  # 增大坐标轴数字
ax.set_yticks(np.arange(70, 100, 10))
ax.set_yticklabels(np.arange(70, 100, 10), fontsize=tick_size, fontweight='bold')  # 增大坐标轴数字

# 设置刻度线粗细和长度
ax.tick_params(axis='both', which='major', width=2.5, length=8)

# 加粗边框
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(3)  # 进一步加粗边框
    spine.set_color('black')

# 网格 - 稍微加粗网格线
ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2)

# 图例 - 调整位置避免与折线重合
# 使用 bbox_to_anchor 精确定位图例位置
# ax.legend(fontsize=legend_size, loc='upper left', bbox_to_anchor=(0.02, 0.98),
#           frameon=True, framealpha=0.95, edgecolor='black', facecolor='white',
#           borderpad=1.2, handlelength=1.5, handletextpad=0.8)

# 调整布局
plt.tight_layout(pad=0)  # 增加内边距，为图例留出空间

# 保存高分辨率图像
plt.savefig('ablation_study_xcomet_clear.png', 
            dpi=300, 
            bbox_inches='tight', 
            pad_inches=0,  # 增加保存时的边距，确保图例完整保存
            facecolor='white',
            edgecolor='black')

# plt.show()