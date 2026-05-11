import numpy as np
import matplotlib.pyplot as plt

# Use seaborn whitegrid style and serif fonts
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Times New Roman', 'Georgia'],
})

# ==== Data preparation ====
data = {
    'Nell-100':   {'INGRAM': [0.378, 0.275, 0.563],
                   'ULTRA':  [0.548, 0.490, 0.720],
                   'GraphOracle': [0.686, 0.589, 0.898]},
    'WK-100':     {'INGRAM': [0.159, 0.092, 0.289],
                   'ULTRA':  [0.200, 0.105, 0.466],
                   'GraphOracle': [0.369, 0.176, 0.693]},
    'FB-100':     {'INGRAM': [0.298, 0.183, 0.457],
                   'ULTRA':  [0.465, 0.297, 0.679],
                   'GraphOracle': [0.525, 0.361, 0.799]},
    'YAGO3-10':   {'INGRAM': [0.378, 0.412, 0.525],
                   'ULTRA':  [0.587, 0.583, 0.725],
                   'GraphOracle': [0.683, 0.644, 0.798]},
    'GeoKG':      {'INGRAM': [0.385, 0.371, 0.543],
                   'ULTRA':  [0.545, 0.491, 0.645],
                   'GraphOracle': [0.606, 0.517, 0.756]},
}

datasets = list(data.keys())
metrics = ['MRR', 'H@1', 'H@10']

# Flatten and compute deltas
ingram_vals = []
ultra_deltas = []
oracle_deltas = []
for ds in datasets:
    ing = data[ds]['INGRAM']
    ult = data[ds]['ULTRA']
    ora = data[ds]['GraphOracle']
    for i in range(len(metrics)):
        ingram_vals.append(ing[i])
        ultra_deltas.append(ult[i] - ing[i])
        oracle_deltas.append(ora[i] - ult[i])

# X positions
x = np.arange(len(datasets) * len(metrics))
width = 0.8

# Pink palette: light, medium, deep
colors = {
    'INGRAM':      '#71b8ed',  
    'ULTRA':       '#b8aeea', 
    'GraphOracle': '#f2a8da'   
}

# ==== Plotting ====
fig, ax = plt.subplots(figsize=(30, 15))
ax.bar(x, ingram_vals, width, label='INGRAM',      color=colors['INGRAM'])
ax.bar(x, ultra_deltas, width, bottom=ingram_vals,
       label='ULTRA', color=colors['ULTRA'])
bottom_for_oracle = np.array(ingram_vals) + np.array(ultra_deltas)
ax.bar(x, oracle_deltas, width, bottom=bottom_for_oracle,
       label='GraphOracle', color=colors['GraphOracle'])

# ==== Formatting ====
xticks = [f'{ds}\n{m}' for ds in datasets for m in metrics]
ax.set_xticks(x)
ax.set_xticklabels(xticks, rotation=20, ha='right')

base = plt.rcParams['font.size']
ax.set_xlabel('Dataset and Metric', fontsize=base * 2)
ax.set_ylabel('Score',           fontsize=base * 2)
ax.tick_params(axis='x', labelsize=base * 2)
ax.tick_params(axis='y', labelsize=base * 2)

ax.legend(fontsize=base * 2.5, loc='upper left')
plt.tight_layout()
plt.savefig('different_relation_graph.pdf', dpi=300, bbox_inches='tight')
