import matplotlib.pyplot as plt
import numpy as np

# 设置专业字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.linewidth'] = 1.2

# 数据
steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]

# spBLEU 数据
spbleu_qe = [8.65, 1.04, 1.06, 1.06, 1.15, 2.78, 1.00, 1.14, 1.15, 1.16, 1.13]
spbleu_qe_lang = [11.95, 11.62, 11.46, 10.81, 10.91, 10.67, 10.53, 10.02, 9.91, 10.43, 9.81]
spbleu_qe_wa_lang = [12.54, 12.59, 12.54, 12.31, 12.04, 12.12, 12.13, 12.16, 12.40, 12.34, 12.45]

# XComet 数据
xcomet_qe = [55.95, 76.83, 77.18, 77.70, 72.75, 73.01, 85.31, 74.53, 74.39, 74.33, 74.47]
xcomet_qe_lang = [47.58, 50.64, 51.95, 52.69, 53.57, 54.09, 54.51, 54.77, 55.08, 55.61, 55.51]
xcomet_qe_wa_lang = [47.16, 48.66, 49.02, 49.38, 50.37, 51.15, 51.88, 52.03, 54.59, 55.23, 53.38]

# 创建图形
plt.figure(figsize=(14, 9))
fig = plt.gcf()
fig.patch.set_facecolor('white')

# 使用更美观的颜色方案
colors = {
    'spbleu_qe': '#E63946',        # 红色
    'spbleu_qe_lang': '#457B9D',   # 蓝色
    'spbleu_qe_wa_lang': '#2A9D8F', # 青绿色
    'xcomet_qe': '#E76F51',        # 珊瑚色
    'xcomet_qe_lang': '#9B5DE5',   # 紫色
    'xcomet_qe_wa_lang': '#F4A261'  # 橙色
}

# 绘制 spBLEU 曲线（实线，带标记）
plt.plot(steps, spbleu_qe, color=colors['spbleu_qe'], linewidth=3, marker='o', 
         markersize=7, markevery=1, label='spBLEU - QE', alpha=0.9)
plt.plot(steps, spbleu_qe_lang, color=colors['spbleu_qe_lang'], linewidth=3, marker='s', 
         markersize=7, markevery=1, label='spBLEU - QE + Lang', alpha=0.9)
plt.plot(steps, spbleu_qe_wa_lang, color=colors['spbleu_qe_wa_lang'], linewidth=3, marker='^', 
         markersize=7, markevery=1, label='spBLEU - QE + Word-alignment + Lang', alpha=0.9)

# 绘制 XComet 曲线（虚线，带不同标记）
plt.plot(steps, xcomet_qe, color=colors['xcomet_qe'], linewidth=3, linestyle='--', 
         marker='D', markersize=6, markevery=1, label='XComet - QE', alpha=0.9)
plt.plot(steps, xcomet_qe_lang, color=colors['xcomet_qe_lang'], linewidth=3, linestyle='--', 
         marker='v', markersize=6, markevery=1, label='XComet - QE + Lang', alpha=0.9)
plt.plot(steps, xcomet_qe_wa_lang, color=colors['xcomet_qe_wa_lang'], linewidth=3, linestyle='--', 
         marker='*', markersize=8, markevery=1, label='XComet - QE + Word-alignment + Lang', alpha=0.9)

# 设置图表属性
plt.xlabel('Training Steps', fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel('Score', fontsize=14, fontweight='bold', labelpad=10)
plt.title('Performance Comparison: spBLEU vs XComet Metrics', 
          fontsize=16, fontweight='bold', pad=20)

# 设置图例
legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                   frameon=True, fancybox=True, shadow=True, 
                   fontsize=11, ncol=1)

# 设置网格和刻度
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
plt.xticks(steps, rotation=45)
plt.tick_params(axis='both', which='major', labelsize=11)

# 设置y轴范围，使图表更紧凑
plt.ylim(0, 90)

# 添加背景色
plt.gca().set_facecolor('#f8f9fa')

# 调整布局
plt.tight_layout()

# 显示图形
plt.savefig("ablation_study_enhanced.png", dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
# plt.show()