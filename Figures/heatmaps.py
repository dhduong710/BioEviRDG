import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Read the CSV file
df = pd.read_csv('', index_col=0)
data = df.values

# Create a custom pink-focused colormap
pink_cmap = mcolors.LinearSegmentedColormap.from_list(
    'pink_shades',
    ['#FFF0F5', '#FFC0CB', '#FF69B4', '#FF1493']  # lavender blush → light pink → hot pink → deep pink
)

# Get the default font size and increase it by 2
default_fontsize = plt.rcParams['font.size']
increased_fontsize = default_fontsize + 2

# Plot with your preferred colormap choice
fig, ax = plt.subplots()
cax = ax.imshow(
    data,
    cmap=pink_cmap,
    origin='lower'
)

# Set ticks from 0 to N-1
n_rows, n_cols = data.shape
ax.set_xticks(np.arange(n_cols))
ax.set_yticks(np.arange(n_rows))
ax.set_xticklabels(np.arange(n_cols), fontsize=increased_fontsize)
ax.set_yticklabels(np.arange(n_rows), fontsize=increased_fontsize)

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=90, ha='center', va='top')

# Add colorbar and labels
cbar = fig.colorbar(cax, ax=ax)

# Increase colorbar tick labels font size
cbar.ax.tick_params(labelsize=increased_fontsize)

plt.tight_layout()
# plt.show()
plt.savefig('pink_heatmap.pdf')  # Save as PDF