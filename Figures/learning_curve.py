import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.ticker import FixedLocator

# ====== Global Serif Font Configuration ======
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Times New Roman', 'Georgia'],
    'font.size': 26,
})
# ===============================================

# Nell-100 data
nell_train_scratch = [0, 0.1237, 0.1765, 0.2539, 0.3247, 0.3834, 0.5876, 0.5903]
nell_fine_tune     = [0.447, 0.4841, 0.5207, 0.6863, 0.6862, 0.6854]
nell_supervised    = [0.471, 0.461, 0.4631, 0.4514, 0.4579, 0.4428]

# WK-100 data
wk_train_scratch   = [0, 0.0989, 0.1536, 0.1814, 0.2103, 0.2537, 0.3322, 0.3327]
wk_fine_tune       = [0.213, 0.283, 0.3683, 0.3546, 0.3675, 0.3789]
wk_supervised      = [0.164, 0.1668, 0.1638, 0.1573, 0.1694, 0.1678]

# FB-100 data
fb_train_scratch   = [0, 0.1375, 0.2048, 0.2474, 0.2857, 0.3645, 0.4437, 0.4496]
fb_fine_tune       = [0.427, 0.4722, 0.5174, 0.5283, 0.5153, 0.5249]
fb_supervised      = [0.449, 0.4329, 0.4154, 0.3937, 0.3785, 0.3812]

# Refreshed color palette
train_scratch_color = '#4da6ff'   # brighter sky‑blue
fine_tune_color     = '#ff6666'   # coral pink
supervised_color    = '#66cc88'   # mint green

def create_beautiful_plot(dataset_name, train_scratch, fine_tune, supervised):
    # High‑resolution settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f9f9f9')

    # Create continuous x-axis with a smooth break
    # We'll compress the middle section (between 5 and 15)
    all_epochs = [0, 1, 2, 3, 4, 5, 15, 25]
    
    # Create a transformed x-axis that visually compresses the gap
    transformed_x = [0, 1, 2, 3, 4, 5, 6, 7]  # Equal spacing for display
    
    # Line/marker styles
    line_styles = ['-', '--', '-.']
    markers = ['o', 'D', '^']
    markersize = 14
    colors = [train_scratch_color, fine_tune_color, supervised_color]
    
    # Plot train-from-scratch - all epochs
    line1, = ax.plot(transformed_x, train_scratch, linestyle=line_styles[0], color=colors[0], 
                     linewidth=4.5, label='Train-from-scratch')
    line1.set_path_effects([pe.withStroke(linewidth=7, foreground='white')])
    ax.plot(transformed_x, train_scratch, ls='none', marker=markers[0], markersize=markersize,
            markerfacecolor='white', markeredgecolor=colors[0], markeredgewidth=2.5)
    #ax.fill_between(transformed_x, 0, train_scratch, color=colors[0], alpha=0.10)
    
    # Plot fine-tune - only up to epoch 5
    fine_tune_x = transformed_x[:6]
    line2, = ax.plot(fine_tune_x, fine_tune, linestyle=line_styles[1], color=colors[1], 
                     linewidth=4.5, label='Fine-tune')
    line2.set_path_effects([pe.withStroke(linewidth=7, foreground='white')])
    ax.plot(fine_tune_x, fine_tune, ls='none', marker=markers[1], markersize=markersize,
            markerfacecolor='white', markeredgecolor=colors[1], markeredgewidth=2.5)
    #ax.fill_between(fine_tune_x, 0, fine_tune, color=colors[1], alpha=0.10)
    
    # Plot supervised - only up to epoch 5
    supervised_x = transformed_x[:6]
    line3, = ax.plot(supervised_x, supervised, linestyle=line_styles[2], color=colors[2], 
                     linewidth=4.5, label='ULTRA-finetune')
    line3.set_path_effects([pe.withStroke(linewidth=7, foreground='white')])
    ax.plot(supervised_x, supervised, ls='none', marker=markers[2], markersize=markersize,
            markerfacecolor='white', markeredgecolor=colors[2], markeredgewidth=2.5)
    #ax.fill_between(supervised_x, 0, supervised, color=colors[2], alpha=0.10)
    
    # Add a subtle break indicator
    break_pos = 5.5  # Between transformed positions 5 and 6
    break_width = 0.2
    
    # Subtle shaded rectangle to indicate the break
    ax.axvspan(break_pos - break_width/2, break_pos + break_width/2, 
               color='#f0f0f0', alpha=0.7, zorder=0)
    
    # Custom x-tick labels that match the original epochs
    ax.set_xticks(transformed_x)
    ax.set_xticklabels([str(e) for e in all_epochs])
    
    # Grid and styling
    ax.grid(True, linestyle='--', alpha=0.3, color='#e0e0e0')
    ax.tick_params(axis='both', which='major', labelsize=24)
    
    # Labels
    ax.set_xlabel('Epoch', fontsize=28, fontstyle='italic', fontweight='medium', labelpad=15)
    ax.set_ylabel('MRR', fontsize=28, fontstyle='italic', fontweight='medium', labelpad=15)
    
    # Y-axis limits
    max_value = max(max(train_scratch), max(fine_tune), max(supervised))
    ax.set_ylim(0, max_value * 1.1)
    
    # Legend
    legend = ax.legend(loc='lower right', fontsize=20, frameon=True,
                      framealpha=0.95, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#dddddd')
    
    # Border styling
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#dddddd')
        spine.set_linewidth(1)
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {dataset_name}_performance.pdf with serif fonts!")

# Generate the three plots
create_beautiful_plot('Nell-100', nell_train_scratch, nell_fine_tune, nell_supervised)
create_beautiful_plot('WK-100', wk_train_scratch, wk_fine_tune, wk_supervised)
create_beautiful_plot('FB-100', fb_train_scratch, fb_fine_tune, fb_supervised)