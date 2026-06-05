# -*- coding: utf-8 -*-
"""
plot_fig3c_pulse_soc_rmse.py

Reproduce Figure 3c SOC RMSE plot (full reference style) exactly like i10 original.

Default input:
    results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv
Default output:
    results/figures/main/fig3c/pulse_soc_rmse_COMBO_REF.png
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Config
# ------------------------
file_path = r'results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv'
save_dir = r'results/figures/main/fig3c'
save_path = os.path.join(save_dir, 'pulse_soc_rmse_COMBO_REF.png')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ------------------------
# Read CSV & compute RMSE
# ------------------------
df = pd.read_csv(file_path)

def calculate_rmse(group):
    return np.sqrt(np.mean((group['soc_true'] - group['soc_pred'])**2))

pulse_rmse = df.groupby('pulse_ms').apply(calculate_rmse).reset_index()
pulse_rmse.columns = ['Pulse Duration (ms)', 'RMSE']
pulse_rmse = pulse_rmse.sort_values('Pulse Duration (ms)')

x_labels = pulse_rmse['Pulse Duration (ms)'].astype(str)
y_values = pulse_rmse['RMSE'].values
x_pos = np.arange(len(x_labels))

# ------------------------
# Plot config
# ------------------------
plt.rcParams['font.family'] = 'Arial'
bar_color = '#BDC3C7'
line_color = '#2C3E50'

# ------------------------
# Plot
# ------------------------
fig, ax = plt.subplots(figsize=(6,5))
ax.bar(x_pos, y_values, width=0.5, color=bar_color, edgecolor='none', alpha=0.6, zorder=2)
ax.plot(x_pos, y_values, color=line_color, linewidth=2.5, marker='o', markersize=8,
        markerfacecolor=line_color, markeredgecolor='white', markeredgewidth=1.5, zorder=3)

ax.set_ylim(6,9.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)
ax.set_ylabel('RMSE (SOC %)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_xlabel('Pulse Duration (ms)', fontsize=12, fontweight='bold', labelpad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)

# Value annotation
y_offset = 0.1
for i, val in enumerate(y_values):
    ax.text(i, val + y_offset, f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color=line_color)

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close(fig)

print(f"✅ Figure 3c SOC RMSE saved: {save_path}")