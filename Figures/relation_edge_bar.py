# relation_edge_bar.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# Set font to a more modern sans-serif
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams.update({'font.size': 20})  # unify base font size

# Data from the table
categories = ['[0,50]', '(50,150]', '>150']
ingram = [0.239, 0.186, 0.117]
ultra = [0.411, 0.334, 0.321]
graph_oracle = [0.569, 0.582, 0.585]

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 9), dpi=300)

# Plot bars
bar1 = ax.bar(x - width, ingram, width, label='INGRAM', color='#36a2eb', alpha=0.7,
              edgecolor='#36a2eb', linewidth=1.5, zorder=1)
bar2 = ax.bar(x,        ultra, width, label='ULTRA',  color='#4bc0c0', alpha=0.7,
              edgecolor='#4bc0c0', linewidth=1.5, zorder=1)
bar3 = ax.bar(x + width,graph_oracle, width, label='GraphOracle', color='#ff6384', alpha=0.7,
              edgecolor='#ff6384', linewidth=1.5, zorder=1)

# Plot smooth lines connecting bar tops
ax.plot(x - width, ingram,        color='#36a2eb', linewidth=4, solid_capstyle='round', zorder=10)
ax.plot(x,          ultra,        color='#4bc0c0', linewidth=4, solid_capstyle='round', zorder=10)
ax.plot(x + width,  graph_oracle, color='#ff6384', linewidth=4, solid_capstyle='round', zorder=10)

# Scatter points
ax.scatter(x - width, ingram,        color='#36a2eb', s=120, zorder=15,
           edgecolor='white', linewidth=2)
ax.scatter(x,          ultra,        color='#4bc0c0', s=120, zorder=15,
           edgecolor='white', linewidth=2)
ax.scatter(x + width,  graph_oracle, color='#ff6384', s=120, zorder=15,
           edgecolor='white', linewidth=2)

# Labels
plt.xlabel('Number of Relations', fontsize=30, labelpad=15, fontweight='bold', color='#505050')
plt.ylabel('Average MRR',    fontsize=35, labelpad=15, fontweight='bold', color='#505050')

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=30, fontweight='bold')
ax.set_ylim(0, 0.7)
ax.tick_params(axis='y', labelsize=40)

# Grid and spines
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#CCCCCC')
ax.spines['left'].set_color('#CCCCCC')

# Legend styling
legend = ax.legend([bar3, bar2, bar1],
                   ['GraphOracle', 'ULTRA', 'INGRAM'],
                   fontsize=22, frameon=True, loc='upper left',
                   facecolor='white', edgecolor='#CCCCCC', framealpha=0.9)
plt.setp(legend.get_title(), fontweight='bold')

plt.tight_layout()
plt.savefig('relation_edge_bar.pdf', dpi=300, bbox_inches='tight')
