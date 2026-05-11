import matplotlib.pyplot as plt
import numpy as np

# ====== Global Serif Font Configuration ======
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Times New Roman', 'Georgia'],
})
# ==============================================

N = [1, 2, 3, 4, 5, 6]
avg_mrr = [0.2178, 0.2636, 0.2758, 0.3530, 0.3562, 0.3566]
avg_h1 = [0.1494, 0.1742, 0.1828, 0.2490, 0.2524, 0.2536]
avg_h10 = [0.3392, 0.4136, 0.4232, 0.4760, 0.4820, 0.4848]

colors = ['#EA8379', '#7DAEE0', '#B395BD']

fig = plt.figure(figsize=(10, 6))
fig.patch.set_facecolor('white')   # Figure background color
ax = fig.add_subplot(111)
ax.set_facecolor('#f9f9f9')        # Axes background color

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.plot(N, avg_mrr, marker='s', markersize=16, linewidth=5, color=colors[0], label='MRR')
plt.plot(N, avg_h1, marker='^', markersize=16, linewidth=5, color=colors[1], label='H@1')
plt.plot(N, avg_h10, marker='o', markersize=16, linewidth=5, color=colors[2], label='H@10')

plt.xlabel('number of pre-trained datasets', fontsize=28)
plt.ylabel('Evaluation Index', fontsize=28)

ax.grid(True, linestyle='--', alpha=0.6)
plt.xticks(N, fontsize=24)
plt.yticks(fontsize=24)
plt.ylim(0, 0.6)

legend = ax.legend(fontsize=24, frameon=True)
legend.get_frame().set_facecolor('white')    # Legend background color
legend.get_frame().set_edgecolor('#dddddd')   # Legend border color

plt.tight_layout()
plt.savefig('ablation_figure.pdf', dpi=300, bbox_inches='tight')
