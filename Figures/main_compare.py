import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# Define the data from the tables in the image
datasets = [

        # Benchmark datasets
    {"name": "WN18RR", "graphoracle": 0.675, "sota": 0.567, "sota_model": "one-shot-subgraph"},
    {"name": "FB15k237", "graphoracle": 0.471, "sota": 0.411, "sota_model": "A*Net"},
    {"name": "NELL-995", "graphoracle": 0.621, "sota": 0.543, "sota_model": "RED-GNN"},
    {"name": "YAGO3-10", "graphoracle": 0.683, "sota": 0.606, "sota_model": "one-shot-subgraph"},

    # NELL datasets
    {"name": "NELL-100", "graphoracle": 0.686, "sota": 0.458, "sota_model": "ULTRA"},
    {"name": "NELL-75", "graphoracle": 0.612, "sota": 0.374, "sota_model": "ULTRA"},
    {"name": "NELL-50", "graphoracle": 0.589, "sota": 0.225, "sota_model": "NBFNet"},
    {"name": "NELL-25", "graphoracle": 0.579, "sota": 0.283, "sota_model": "NBFNet"},

    # WK datasets
    {"name": "WK-100", "graphoracle": 0.369, "sota": 0.168, "sota_model": "ULTRA"},
    {"name": "WK-75", "graphoracle": 0.469, "sota": 0.380, "sota_model": "ULTRA"},
    {"name": "WK-50", "graphoracle": 0.189, "sota": 0.140, "sota_model": "ULTRA"},
    {"name": "WK-25", "graphoracle": 0.582, "sota": 0.321, "sota_model": "ULTRA"},

    # FB datasets
    {"name": "FB-100", "graphoracle": 0.525, "sota": 0.444, "sota_model": "ULTRA"},
    {"name": "FB-75", "graphoracle": 0.538, "sota": 0.400, "sota_model": "ULTRA"},
    {"name": "FB-50", "graphoracle": 0.585, "sota": 0.334, "sota_model": "ULTRA"},
    {"name": "FB-25", "graphoracle": 0.562, "sota": 0.383, "sota_model": "ULTRA"},


    # WN datasets
    {"name": "WN_V1", "graphoracle": 0.807, "sota": 0.733, "sota_model": "one-shot-subgraph"},
    {"name": "WN_V2", "graphoracle": 0.793, "sota": 0.715, "sota_model": "A*Net"},
    {"name": "WN_V3", "graphoracle": 0.569, "sota": 0.474, "sota_model": "RED-GNN"},
    {"name": "WN_V4", "graphoracle": 0.762, "sota": 0.662, "sota_model": "one-shot-subgraph"},

    # FB V datasets
    {"name": "FB_V1", "graphoracle": 0.619, "sota": 0.509, "sota_model": "one-shot-subgraph"},
    {"name": "FB_V2", "graphoracle": 0.631, "sota": 0.524, "sota_model": "A*Net"},
    {"name": "FB_V3", "graphoracle": 0.694, "sota": 0.504, "sota_model": "RED-GNN"},
    {"name": "FB_V4", "graphoracle": 0.658, "sota": 0.496, "sota_model": "one-shot-subgraph"},

    # NL datasets
    {"name": "NL_V1", "graphoracle": 0.794, "sota": 0.757, "sota_model": "one-shot-subgraph"},
    {"name": "NL_V2", "graphoracle": 0.684, "sota": 0.575, "sota_model": "A*Net"},
    {"name": "NL_V3", "graphoracle": 0.659, "sota": 0.563, "sota_model": "RED-GNN"},
    {"name": "NL_V4", "graphoracle": 0.569, "sota": 0.469, "sota_model": "one-shot-subgraph"},

        # NL datasets (Note: these are distinct from NL_V series and represent Protein/Drug/Disease relations)
    {"name": "Protein→BP", "graphoracle": 0.498, "sota": 0.334, "sota_model": "one-shot-subgraph"},
    {"name": "Protein→MF", "graphoracle": 0.499, "sota": 0.402, "sota_model": "A*Net"},
    {"name": "Protein→CC", "graphoracle": 0.475, "sota": 0.385, "sota_model": "RED-GNN"},
    {"name": "Drug→Disease", "graphoracle": 0.268, "sota": 0.202, "sota_model": "one-shot-subgraph"}, # First instance
    {"name": "Protein→Drug", "graphoracle": 0.232, "sota": 0.187, "sota_model": "A*Net"},
    {"name": "Disease→Protein", "graphoracle": 0.299, "sota": 0.239, "sota_model": "RED-GNN"},
    {"name": "Drug→Disease", "graphoracle": 0.192, "sota": 0.167, "sota_model": "one-shot-subgraph"}, # Second instance


    {"name": "Amazon-book", "graphoracle": 0.3142, "sota": 0.2237, "sota_model": "KUCNet"},
    {"name": "GeoKG", "graphoracle": 0.606, "sota": 0.493, "sota_model": "Adaprop"},
]

# Calculate averages
graphoracle_values_orig = [d["graphoracle"] for d in datasets]
sota_values_orig = [d["sota"] for d in datasets]
avg_graphoracle = sum(graphoracle_values_orig) / len(graphoracle_values_orig)
avg_sota = sum(sota_values_orig) / len(sota_values_orig)

# Add Avg. to datasets
datasets.append({"name": "Average", "graphoracle": avg_graphoracle, "sota": avg_sota, "sota_model": "N/A"})

# Create an enhanced version with more polished design
with PdfPages('main_compare.pdf') as pdf:
    # Set up the figure with elegant aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Palatino', 'Times New Roman', 'Georgia'],
        'font.size': 26,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 1.6,
    })

    # Extract dataset names for x-axis labels
    labels = [d["name"] for d in datasets]
    graphoracle_values = [d["graphoracle"] for d in datasets]
    sota_values = [d["sota"] for d in datasets]

    # Number of datasets
    n = len(labels)

    # Create the figure and axis
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    # Add extra space at the top for dataset titles
    plt.subplots_adjust(top=0.85)

    # Define the width of the bars and the positions
    bar_width = 0.3
    group_positions = np.arange(n)
    graphoracle_positions = group_positions - bar_width/2
    sota_positions = group_positions + bar_width/2

    # Create colors that match the highlighted GraphOracle in the tables
    graphoracle_color = '#FF9E9E'
    sota_color = '#8DD2C5'

    dataset_groups = [
        {"name": "NELL", "start": 0, "end": 4},
        {"name": "WK", "start": 4, "end": 8},
        {"name": "FB", "start": 8, "end": 12},
        {"name": "Benchmarks", "start": 12, "end": 16}
    ]

    all_values = graphoracle_values + sota_values
    y_min = max(0, min(all_values) * 0.9 if all_values else 0)
    y_max = max(all_values) * 1.1 if all_values else 1
    ax.set_ylim(y_min, y_max)

    ax.yaxis.grid(True, linestyle='--', alpha=0.15, color='#9EAEFF', zorder=0)
    ax.set_facecolor('white')

    ax.set_ylabel('MRR', fontsize=32, fontweight='medium',
                  color='#484D6D', fontstyle='italic', labelpad=15)
    ax.set_xticks(group_positions)

    label_colors = {
        "group1": "#6A7FDB",
        "group2": "#FF9666",
        "group3": "#52B788",
        "group4": "#E07A5F",
        "group5": "#B185DB",
        "group6": "#3D83C6",
        "default": "#000000"
    }

    def get_dataset_group(dataset_name):
        if dataset_name == "Average":
            return "default"  # This will use the default black color
        elif (dataset_name.startswith("NELL-") and dataset_name != "NELL-995" or
            dataset_name.startswith("WK-") or
            dataset_name.startswith("FB-") and dataset_name != "FB15k237"):
            return "group1"
        elif dataset_name in ["WN18RR", "FB15k237", "NELL-995", "YAGO3-10"]:
            return "group2"
        elif (dataset_name.startswith("WN_V") or
              dataset_name.startswith("FB_V") or
              dataset_name.startswith("NL_V")):
            return "group3"
        elif (dataset_name.startswith("Protein→") or
              dataset_name.startswith("Drug→") or
              dataset_name.startswith("Disease→")):
            return "group4"
        elif dataset_name == "GeoKG":
            return "group5"
        elif dataset_name == "Amazon-book":
            return "group6"
        else:
            print(f"Warning: Dataset '{dataset_name}' did not fall into a specific color group.")
            return "default"

    # Set x-axis tick labels (objects are created here)
    ax.set_xticklabels(labels, fontsize=20, rotation=90, ha='center', va='top')

    # Apply custom colors to each x-axis tick label
    # This loop should run *before* any general tick_params that might override colors,
    # OR those general tick_params should be modified not to affect x-axis label colors.
    # The solution below modifies tick_params.
    for i, tick_label_obj in enumerate(ax.get_xticklabels()):
        if i < len(labels): # Safety check
            dataset_name = labels[i]
            group = get_dataset_group(dataset_name)
            color = label_colors.get(group, label_colors["default"])
            # print(f"Applying color to {dataset_name}: Group {group}, Color {color}") # Uncomment for debugging
            tick_label_obj.set_color(color)

    graphoracle_bars = ax.bar(graphoracle_positions, graphoracle_values, bar_width * 0.92,
                              color=graphoracle_color, label='GraphOracle',
                              edgecolor='#e67a7f', linewidth=0.8, alpha=0.9,
                              zorder=10)
    sota_bars = ax.bar(sota_positions, sota_values, bar_width * 0.92,
                       color=sota_color, label='Supervised SOTA',
                       edgecolor='#299d8f', linewidth=0.8, alpha=0.9,
                       zorder=10)

    for i, bar in enumerate(graphoracle_bars):
        x = bar.get_x()
        width = bar.get_width()
        height = bar.get_height()
        if height > 0:
             ax.add_patch(plt.Rectangle((x, 0), width*0.5, height,
                                     color='white', alpha=0.15, zorder=15))

    for i, bar in enumerate(sota_bars):
        x = bar.get_x()
        width = bar.get_width()
        height = bar.get_height()
        if height > 0:
            ax.add_patch(plt.Rectangle((x, 0), width*0.5, height,
                                     color='white', alpha=0.15, zorder=15))

    for i, group in enumerate(dataset_groups[:-1]):
        separator_x = group["end"] - 0.5
        ax.axvline(x=separator_x, color='#C9D5FF', linestyle='-',
                    linewidth=1.0, alpha=0.6, zorder=2)

    title_y = 1.02
    title_positions = []
    for group in dataset_groups:
        start_idx = group["start"]
        end_idx = group["end"]
        if start_idx < len(group_positions) and end_idx <= len(group_positions) and start_idx < end_idx:
            title_positions.append((group_positions[start_idx] + group_positions[end_idx-1]) / 2)

    legend_elements = [
        Line2D([0], [0], color='white', marker='s', markerfacecolor=graphoracle_color,
               markersize=18, label='GraphOracle', markeredgecolor=graphoracle_color, markeredgewidth=0.8),
        Line2D([0], [0], color='white', marker='s', markerfacecolor=sota_color,
               markersize=18, label='Supervised SOTA', markeredgecolor=sota_color, markeredgewidth=0.8)
    ]

    legend = ax.legend(handles=legend_elements,
                       loc='upper left',
                       frameon=True,
                       framealpha=0.95,
                       fontsize=23,
                       ncol=2,
                       title_fontsize=28,
                       edgecolor='#CCCCEE')

    # MODIFIED SECTION FOR TICK PARAMS:
    # Apply general tick parameters (like size, pad) to both axes first.
    ax.tick_params(axis='both', which='major', labelsize=24, length=6, width=1.0, pad=8)
    # Specifically set the color for Y-axis tick labels.
    # The x-axis tick labels have already been custom-colored by the loop above.
    ax.tick_params(axis='y', colors='#484D6D')
    # The following line is not needed if labelsize is set in 'axis=both' and colors are handled.
    # ax.tick_params(axis='y', which='major', labelsize=24) # This was in original, potentially redundant for labelsize.

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.0)
        ax.spines[spine].set_color('#AABCDE')

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2f}'))

    fig.patch.set_linewidth(1.0)
    fig.patch.set_edgecolor('#DDDDFF')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight', dpi=300)
    plt.close()

print("Final PDF file generated: 'main_compare.pdf'")