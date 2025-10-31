import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

steps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
qe = [10.28, 8.65, 1.04, 1.06, 1.06, 1.15, 2.78, 1.00, 1.14, 1.15, 1.16, 1.13]
qe_lang = [10.28, 11.95, 11.62, 11.46, 10.81, 10.91, 10.67, 10.53, 10.02, 9.91, 10.43, 9.81]
qe_word_align_lang = [10.28, 12.54, 12.59, 12.54, 12.31, 12.04, 12.12, 12.13, 12.16, 12.40, 12.34, 12.45]

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
ax.set_title('spBLEU', fontsize=title_size, fontweight='bold', pad=20)
ax.set_xlabel('Training Steps', fontsize=axis_label_size, fontweight="bold", labelpad=15)
ax.set_ylabel('EN-X', fontsize=axis_label_size, fontweight="bold", labelpad=15)

# 坐标轴范围和刻度
ax.set_xlim(0, 1150)
ax.set_ylim(-1, 21)
ax.set_xticks(np.arange(0, 1200, 200))
ax.set_xticklabels(np.arange(0, 1200, 200), fontsize=tick_size, fontweight='bold')  # 增大坐标轴数字
ax.set_yticks(np.arange(0, 22, 5))
ax.set_yticklabels(np.arange(0, 22, 5), fontsize=tick_size, fontweight='bold')  # 增大坐标轴数字

# 设置刻度线粗细和长度
ax.tick_params(axis='both', which='major', width=2.5, length=8)

# 加粗边框
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(3)  # 进一步加粗边框
    spine.set_color('black')

# 网格 - 稍微加粗网格线
ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.2)

# 图例 - 增大字体和边框
ax.legend(fontsize=legend_size, loc='upper right', frameon=True, 
           framealpha=0.95, edgecolor='black', facecolor='white',
           borderpad=1.2, handlelength=1.5, handletextpad=0.8)

# 调整布局
plt.tight_layout(pad=0.0)  # 增加内边距

# 保存高分辨率图像
# plt.savefig('ablation_study_spbleu_clear.png', 
#             dpi=300, 
#             bbox_inches='tight', 
#             pad_inches=0.0,  # 增加保存时的边距
#             facecolor='white',
#             edgecolor='black')
plt.savefig('ablation_study_spbleu_clear.pdf',
            format='pdf',          # explicitly specify PDF format
            bbox_inches='tight',   # trim extra whitespace
            pad_inches=0.0,        # no padding
            facecolor='white',     # background color
            edgecolor='black')     # edge color if applicable

plt.show()