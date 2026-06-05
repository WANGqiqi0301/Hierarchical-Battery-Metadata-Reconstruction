# -*- coding: utf-8 -*-
"""
plot_fig4c_soc_error_distribution.py

Figure 4c:
SOC error distribution boxplot (median + percentile whiskers).

This script produces only the full academic version (no clean version)
and saves the figure in the standardized location for new code.
"""

from __future__ import annotations
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Style configuration
# ==========================================
sns.set_theme(style="white")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['lines.linewidth'] = 1.5

# ==========================================
# 2. Colors (low-saturation, muted)
# ==========================================
BOX_COLORS = ['#6B8EAC', '#BC8585']  # E0 / E1

# ==========================================
# 3. Core drawing function
# ==========================================
def draw_boxplot(ax, data_dict: dict):
    labels = list(data_dict.keys())
    line_color = '#2D2D2D'  # dark gray lines

    for i, label in enumerate(labels):
        d = data_dict[label]

        # Box (25th-75th percentile)
        ax.add_patch(
            plt.Rectangle(
                (i - 0.28, d['p25']),
                0.56,
                d['p75'] - d['p25'],
                facecolor=BOX_COLORS[i],
                edgecolor=line_color,
                alpha=0.9,
                lw=1.5
            )
        )

        # Median line
        ax.plot([i - 0.28, i + 0.28], [d['median'], d['median']], color=line_color, lw=2.2)

        # Whiskers
        ax.plot([i, i], [d['p05'], d['p25']], color=line_color, lw=1.5)
        ax.plot([i, i], [d['p75'], d['p95']], color=line_color, lw=1.5)
        ax.plot([i - 0.12, i + 0.12], [d['p05'], d['p05']], color=line_color, lw=1.5)
        ax.plot([i - 0.12, i + 0.12], [d['p95'], d['p95']], color=line_color, lw=1.5)

    ax.set_xticks(range(len(labels)))
    ax.set_ylim(0, 48)

# ==========================================
# 4. Save function
# ==========================================
def save_soc_error_boxplot(data_dict: dict):
    SAVE_DIR = os.path.join("results", "figures", "main", "fig4c")
    os.makedirs(SAVE_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(2.2, 2.5))
    draw_boxplot(ax, data_dict)
    ax.set_xticklabels(['Ideal', 'Realistic'], fontweight='bold', fontsize=8)
    ax.set_ylabel('MAPE (%)', fontweight='bold', fontsize=8)
    sns.despine(ax=ax, trim=True, offset=4)
    plt.tight_layout()

    file_path = os.path.join(SAVE_DIR, 'fig4c_soc_error_distribution.png')
    fig.savefig(file_path, dpi=600)
    plt.close(fig)

    print(f"[OK] Saved: {file_path}")

# ==========================================
# 5. Example usage
# ==========================================
if __name__ == "__main__":
    data_example = {
        'E0': {'median': 4.49, 'p25': 1.86, 'p75': 11.22, 'p05': 0.31, 'p95': 31.94, 'n': 1265},
        'E1': {'median': 4.80, 'p25': 1.99, 'p75': 12.01, 'p05': 0.43, 'p95': 35.25, 'n': 1265}
    }

    save_soc_error_boxplot(data_example)