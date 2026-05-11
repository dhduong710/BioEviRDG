
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams

# Update font settings
plt.style.use('seaborn-v0_8-whitegrid')
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Times New Roman', 'Georgia'],
    'font.size': 26,
})

# Data for the bar chart
datasets = ['WN18RR', 'YAGO3-10']
columns = ['Original', 'Off top1', 'Off top5', 'Off top10']

# Values for each dataset
wn18rr_values = [0.675, 0.613, 0.564, 0.499]
yago3_10_values = [0.683, 0.625, 0.598, 0.549]

# Set up the plot with a larger figure size for better visualization
plt.figure(figsize=(18, 12), dpi=300)

# Set positions for bars (grouped by dataset)
bar_width = 0.2
x = np.arange(2)  # Two groups: WN18RR and YAGO3-10
offset = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width

# Define color gradients from lightest to darkest
blue_colors = ['#C5E5FD', '#A8D8F8', '#8BC9F0', '#6AAFE6']  # Light to dark blue
pink_colors = ['#F8BBD0', '#F48FB1', '#F06292', '#EC407A']  # Light to dark pink

# Map colors based on value rank (higher value gets darker color)
wn18rr_sorted_indices = np.argsort(wn18rr_values)
yago3_sorted_indices = np.argsort(yago3_10_values)

wn18rr_color_map = {}
yago3_color_map = {}

for i, idx in enumerate(wn18rr_sorted_indices):
    wn18rr_color_map[idx] = blue_colors[i]

for i, idx in enumerate(yago3_sorted_indices):
    yago3_color_map[idx] = pink_colors[i]

# Create grouped bars with color mapping
for i in range(len(columns)):
    plt.bar(x[0] + offset[i], wn18rr_values[i], width=bar_width,
            color=wn18rr_color_map[i], edgecolor='white', linewidth=1.5,
            label=f'{columns[i]} (WN18RR)')
    plt.bar(x[1] + offset[i], yago3_10_values[i], width=bar_width,
            color=yago3_color_map[i], edgecolor='white', linewidth=1.5,
            label=f'{columns[i]} (YAGO3-10)')

# Add labels, title and custom x-axis tick labels
plt.xticks(x, datasets, fontsize=56, fontweight='bold')  # Doubled from 28 to 56
plt.yticks(fontsize=48)  # Doubled from 24 to 48

# Set y-axis limits with a bit of padding
plt.ylim(0.45, 0.70)

# Add a light grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.3, color='#CCCCCC')

# Custom legend with grouped items
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
legend_order = [f'{col} (WN18RR)' for col in columns] + [f'{col} (YAGO3-10)' for col in columns]
ordered_handles = [by_label[label] for label in legend_order if label in by_label]
ordered_labels = [label for label in legend_order if label in by_label]

# Create a more organized legend with doubled font sizes
plt.legend(ordered_handles[:4] + ordered_handles[4:],
           [col for col in columns] + [col for col in columns],
           ncol=2, title_fontsize=25, fontsize=25, loc='upper right',
           frameon=True, edgecolor='lightgrey')

# Add a box around the plot with a slightly thicker line
ax = plt.gca()
ax.set_facecolor('#f8f9fa')
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_color('lightgrey')

# Adjust layout
plt.tight_layout()

# Create a PDF file with high quality
with PdfPages('bar_chart_comparison.pdf') as pdf:
    plt.savefig(pdf, format='pdf', bbox_inches='tight', dpi=300)
