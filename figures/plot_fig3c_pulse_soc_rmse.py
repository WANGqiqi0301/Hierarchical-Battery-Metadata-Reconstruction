# -*- coding: utf-8 -*-
"""
plot_fig3c_pulse_soc_rmse_with_pure.py

Reproduce Figure 3c SOC RMSE plot.

This version saves:
1. A complete reference-style PNG
2. A pure PNG with no axes / no labels / no text, using the slim 2.38:0.32 aspect ratio

Default input:
    results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv
Default outputs:
    results/figures/main/fig3c/pulse_soc_rmse_COMBO_REF.png
    results/figures/main/fig3c/pulse_soc_rmse_COMBO_REF_pure.png
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
pure_save_path = os.path.join(save_dir, 'pulse_soc_rmse_COMBO_REF_pure.png')

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
# Complete reference plot
# ------------------------
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(
    x_pos, y_values,
    width=0.5,
    color=bar_color,
    edgecolor='none',
    alpha=0.6,
    zorder=2
)
ax.plot(
    x_pos, y_values,
    color=line_color,
    linewidth=2.5,
    marker='o',
    markersize=8,
    markerfacecolor=line_color,
    markeredgecolor='white',
    markeredgewidth=1.5,
    zorder=3
)

ax.set_ylim(6, 9.5)
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
    ax.text(
        i, val + y_offset, f'{val:.3f}',
        ha='center', va='bottom',
        fontsize=9, fontweight='bold', color=line_color
    )

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close(fig)


# ------------------------
# Pure plot: match the reference slim aspect ratio 2.38:0.32
# 890 px width corresponds to about 120 px height
# ------------------------
pure_width_px = 89
pure_height_px = 12
pure_dpi = 600
pure_figsize =(1.55, 0.32)

fig, ax = plt.subplots(figsize=pure_figsize, dpi=pure_dpi)

ax.bar(
    x_pos, y_values,
    width=0.8,
    color=bar_color,
    edgecolor='none',
    alpha=0.9,
    zorder=1
)
ax.plot(
    x_pos, y_values,
    color=line_color,
    linewidth=1.0,
    linestyle='-',
    marker='o',
    markersize=4,
    markerfacecolor=line_color,
    markeredgecolor='white',
    markeredgewidth=0.7,
    zorder=2
)

ax.set_ylim(6, 9.5)
ax.set_xlim(-0.5, len(x_pos) - 0.5)
ax.set_axis_off()

fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

plt.savefig(
    pure_save_path,
    dpi=pure_dpi,
    transparent=True,
    bbox_inches='tight',
    pad_inches=0
)
plt.close(fig)

print(f"✅ Figure 3c SOC RMSE saved: {save_path}")
print(f"✅ Figure 3c SOC RMSE pure saved: {pure_save_path}")
print(f"✅ Pure size target: {pure_width_px}x{pure_height_px} px")
