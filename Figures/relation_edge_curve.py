import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# Set font to a more modern sans-serif
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams.update({'font.size': 20})  # unify base font size

# Create DataFrame with MODIFIED data values
data = {
    'Dataset': [
        'NL-25', 'NL-50', 'NL-75', 'NL-100',
        'WK-25', 'WK-50', 'WK-75', 'WK-100',
        'FB-25', 'FB-50', 'FB-75', 'FB-100',
        'WN_V1', 'WN_V2', 'WN_V3', 'WN_V4',
        'FB_V1', 'FB_V2', 'FB_V3', 'FB_V4',
        'NL_V1', 'NL_V2', 'NL_V3', 'NL_V4'
    ],
    'Relations': [
        146, 150, 138, 99,
        67, 102, 77, 103,
        233, 228, 213, 202,
        9, 10, 11, 9,
        180, 200, 215, 219,
        14, 88, 142, 76
    ],
    'INGRAM': [
        1610, 1748, 1626, 892,
        598, 1130, 732, 1052,
        7172, 6294, 5042, 4058,
        15, 18, 22, 15,
        1622, 2692, 3398, 4624,
        30, 1574, 1942, 1296
    ],
    'ULTRA': [
        2300, 2526, 2336, 1159,
        947, 2164, 1253, 1695,
        10479, 9300, 7375, 5728,
        14, 17, 20, 14,
        2416, 4050, 5015, 7036,
        25, 2065, 2558, 1657
    ],
    'GraphOracle': [
        797, 861, 787, 416,
        256, 508, 313, 460,
        3501, 3135, 2524, 2017,
        13, 15, 17, 13,
        712, 1237, 1640, 2231,
        21, 842, 1017, 744
    ]
}

df = pd.DataFrame(data)

plt.figure(figsize=(14, 9), dpi=300)
ax = plt.gca()

# Enhanced colors palette
colors = {
    'INGRAM': '#36a2eb',
    'ULTRA':  '#4bc0c0',
    'GraphOracle': '#ff6384'
}

# Different line styles for each method
# GraphOracle is solid, others are different styles
line_styles = {
    'INGRAM': '--',      # dashed
    'ULTRA': ':',        # dotted
    'GraphOracle': '-'   # solid
}

# Different markers for scatter points
markers = {
    'INGRAM': 'o',
    'ULTRA': 's',      # square
    'GraphOracle': '^'  # triangle
}

# Grid & spines styling
ax.grid(color='#E0E0E0', linestyle='-', linewidth=1, alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#CCCCCC')
ax.spines['left'].set_color('#CCCCCC')

# Fit & plot for each method
methods = ['INGRAM', 'ULTRA', 'GraphOracle']
for method in methods:
    x_data = df['Relations']
    y_data = df[method]
    
    # Plot the scattered points
    plt.scatter(x_data, y_data, marker=markers[method], s=100, 
                color=colors[method], alpha=0.7, edgecolor='white', 
                linewidth=1, zorder=11)
    
    # Fit the curve
    X = np.column_stack((x_data**2, x_data))
    a, b = np.linalg.lstsq(X, y_data, rcond=None)[0]
    x_line = np.linspace(0, max(df['Relations']), 500)
    y_line = a * x_line**2 + b * x_line
    
    # Plot the fitted curve with different line styles
    plt.plot(x_line, y_line, line_styles[method], linewidth=4, solid_capstyle='round',
             label=method, color=colors[method], zorder=10)
    
    # subtle shadow
    plt.plot(x_line, y_line, line_styles[method], linewidth=5, alpha=0.1,
             color=colors[method], zorder=9)

# Labels & ticks
plt.xlabel('Number of Relations', fontsize=30, labelpad=15,
           fontweight='bold', color='#505050')
plt.ylabel('Number of Edges',      fontsize=30, labelpad=15,
           fontweight='bold', color='#505050')
plt.xticks(fontsize=30, color='#505050')
plt.yticks(fontsize=30, color='#505050')

# Vertical markers
for val in [50, 100, 150, 200]:
    plt.axvline(x=val, color='#CCCCCC', linestyle='--',
                alpha=0.5, zorder=0)

# Reorder legend: GraphOracle, ULTRA, INGRAM
handles, labels = ax.get_legend_handles_labels()
order = [2, 1, 0]
legend = ax.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    fontsize=22, frameon=True, loc='upper left',
    facecolor='white', edgecolor='#CCCCCC',
    framealpha=0.9, title='Methods', title_fontsize=20
)
plt.setp(legend.get_title(), fontweight='bold')

plt.tight_layout()
plt.savefig('relation_edge_curve.pdf', dpi=300, bbox_inches='tight')