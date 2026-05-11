import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Set up a refined, elegant style
plt.style.use('seaborn-v0_8-whitegrid')
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Times New Roman', 'Georgia'],
    'font.size': 28,
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#333333',
})

# Data from the table
datasets = ['WN18RR', 'YAGO3-10']
columns = ['Original', 'Off top 5', 'Off top 10', 'Off tail 5', 'Off tail 10', 'Off random 5', 'Off random 10']

# Values for each dataset (directly from the table)
wn18rr_values = [0.675, 0.564, 0.499, 0.682, 0.659, 0.645, 0.598]
yago3_10_values = [0.683, 0.598, 0.549, 0.691, 0.676, 0.658, 0.639]

# Set up the plot with a larger figure size for better visualization
fig, ax = plt.subplots(figsize=(18, 14), dpi=300, facecolor='#f9f9f9')
ax.set_facecolor('#f9f9f9')

# Set positions for bars (grouped by dataset)
bar_width = 0.11
x = np.arange(2)  # Two groups: WN18RR and YAGO3-10

# Calculate offsets for 7 categories
positions = np.linspace(-(len(columns)-1)/2 * bar_width, (len(columns)-1)/2 * bar_width, len(columns))

# Define a sophisticated color palette (bright but soft and elegant)
# 1. Original - soft teal/mint
original_color = "#8ECDC9"  # Elegant mint

# 2. Top - soft coral/peach variations
top_colors = ["#FF9A76", "#FFC7A2"]  # Soft coral/peach

# 3. Tail - soft lavender/lilac variations
tail_colors = ["#B0A6EA", "#D6CDFE"]  # Elegant purples

# 4. Random - soft blue variations
random_colors = ["#7AA6DA", "#ABC9F0"]  # Elegant blues

# Compile all colors in order
color_map = {
    'Original': original_color,
    'Off top 5': top_colors[0],
    'Off top 10': top_colors[1],
    'Off tail 5': tail_colors[0],
    'Off tail 10': tail_colors[1],
    'Off random 5': random_colors[0],
    'Off random 10': random_colors[1]
}

# Create grouped bars with the new color system
for i, column in enumerate(columns):
    # WN18RR (left)
    bar1 = plt.bar(x[0] + positions[i], wn18rr_values[i], width=bar_width,
            color=color_map[column], edgecolor='white', linewidth=1.5,
            alpha=0.92, zorder=3)
    
    # YAGO3-10 (right)
    bar2 = plt.bar(x[1] + positions[i], yago3_10_values[i], width=bar_width,
            color=color_map[column], edgecolor='white', linewidth=1.5,
            alpha=0.92, zorder=3)

# Add refined labels
plt.xticks(x, datasets, fontsize=52, fontweight='bold')
plt.yticks(fontsize=38)

# Set y-axis limits with proper padding
plt.ylim(0.45, 0.725)

# Refine grid appearance for elegance
plt.grid(axis='y', linestyle='--', alpha=0.2, color='#aaaaaa', zorder=0)

# Customize spines for a cleaner look
for spine in plt.gca().spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_color('#666666')

# Add y-axis label
plt.ylabel('MRR', fontsize=44, labelpad=20, color='#333333', fontweight='bold')

# Leave space at the top for the legend
plt.subplots_adjust(top=0.85)

# Create a single unified legend with all items
legend_elements = [
    mpatches.Patch(color=original_color, alpha=0.92, label='Original'),
    mpatches.Patch(color=top_colors[0], alpha=0.92, label='Top 5'),
    mpatches.Patch(color=top_colors[1], alpha=0.92, label='Top 10'),
    mpatches.Patch(color=tail_colors[0], alpha=0.92, label='Tail 5'),
    mpatches.Patch(color=tail_colors[1], alpha=0.92, label='Tail 10'),
    mpatches.Patch(color=random_colors[0], alpha=0.92, label='Random 5'),
    mpatches.Patch(color=random_colors[1], alpha=0.92, label='Random 10'),
]

# Create a single unified legend with 4 columns to create exactly 2 rows
legend = plt.legend(handles=legend_elements, 
           ncol=4, fontsize=30, loc='upper center', bbox_to_anchor=(0.5, 1.05),
           frameon=True, fancybox=True, shadow=True, framealpha=0.95)
legend.get_frame().set_linewidth(0.8)
legend.get_frame().set_edgecolor('#888888')

# Adjust layout with proper margins
plt.tight_layout(rect=[0, 0, 1, 0.90])  # Leave room at the top for the legend

# Add a subtle background gradient for elegance
ax.patch.set_facecolor('#f9f9f9')
fig.patch.set_facecolor('#f9f9f9')

# Display the plot with tight bounding box to capture everything
plt.savefig('relation_figure.pdf', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), 
            edgecolor='none', pad_inches=0.1)