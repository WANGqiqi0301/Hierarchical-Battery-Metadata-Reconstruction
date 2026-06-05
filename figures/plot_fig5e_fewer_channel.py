# -*- coding: utf-8 -*-
"""
plot_fig5e_fewer_channel.py

Figure 5e:
Effect of input-channel composition on model performance.

This script uses manually entered summary results and does not depend on
legacy i10 result folders.

Output:
    results/figures/main/fig5e/fig5e_fewer_channel.png
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================
SAVE_DIR = "results/figures/main/fig5e"
SAVE_NAME = "fig5e_fewer_channel.png"

os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# Data
# =============================================================================
CASE_ORDER = [
    "ch12",
    "ch1_only",
    "ch13",
    "full",
]

CASE_LABELS = {
    "ch1_only": "Ch1\nRaw",
    "ch12": "Ch1+Ch2\nRaw+ΔU",
    "ch13": "Ch1+Ch3\nRaw+OCV",
    "full": "Full\nRaw+ΔU+OCV",
}

MANUAL_RESULTS = {
    "ch1_only": {
        "classification_accuracy_pct": 86.95,
        "soc_median_ape_pct": 11.16,
        "soh_median_ape_pct": 5.26,
    },
    "ch12": {
        "classification_accuracy_pct": 86.95,
        "soc_median_ape_pct": 13.70,
        "soh_median_ape_pct": 5.24,
    },
    "ch13": {
        "classification_accuracy_pct": 88.98,
        "soc_median_ape_pct": 12.86,
        "soh_median_ape_pct": 5.35,
    },
    "full": {
        "classification_accuracy_pct": 92.30,
        "soc_median_ape_pct": 4.80,
        "soh_median_ape_pct": 2.43,
    },
}


# =============================================================================
# Build dataframe
# =============================================================================
rows = []

for case in CASE_ORDER:
    if case not in MANUAL_RESULTS:
        raise RuntimeError(f"Missing manual result for case: {case}")

    item = MANUAL_RESULTS[case]

    rows.append(
        {
            "case": case,
            "case_label": CASE_LABELS.get(case, case),
            "classification_accuracy_pct": float(
                item["classification_accuracy_pct"]
            ),
            "soc_median_ape_pct": float(
                item["soc_median_ape_pct"]
            ),
            "soh_median_ape_pct": float(
                item["soh_median_ape_pct"]
            ),
        }
    )

df = pd.DataFrame(rows)


# =============================================================================
# Plot configuration
# =============================================================================
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    }
)

spacing = 0.55
x = np.arange(len(df)) * spacing
bar_width = 0.18

soc_color = "#4B5B71"
soh_color = "#A3BACB"
acc_color = "#F08C52"


# =============================================================================
# Plot
# =============================================================================
fig, ax1 = plt.subplots(figsize=(14 * 9.11 / 14.5, 5), dpi=600)

ax1.bar(
    x - bar_width / 2,
    df["soc_median_ape_pct"],
    width=bar_width,
    color=soc_color,
    label="SOC error",
)

ax1.bar(
    x + bar_width / 2,
    df["soh_median_ape_pct"],
    width=bar_width,
    color=soh_color,
    label="SOH error",
)

ax2 = ax1.twinx()

ax2.plot(
    x,
    df["classification_accuracy_pct"],
    color=acc_color,
    marker="o",
    linewidth=5,
    markersize=15,
    label="Accuracy",
    zorder=10,
)

for xi, yi in zip(x, df["classification_accuracy_pct"]):
    ax2.text(
        xi,
        yi + 0.4,
        f"{yi:.1f}",
        ha="center",
        fontsize=8,
        color=acc_color,
    )

ax1.set_xticks(x)
ax1.set_xticklabels(df["case_label"], fontsize=9)

ax1.set_ylabel("Estimation error (%)", fontsize=10)
ax2.set_ylabel("Accuracy (%)", fontsize=10)

ax1.set_ylim(
    0,
    max(
        df["soc_median_ape_pct"].max(),
        df["soh_median_ape_pct"].max(),
    )
    * 1.25,
)

ax2.set_ylim(
    df["classification_accuracy_pct"].min() - 5,
    min(100, df["classification_accuracy_pct"].max() + 3),
)

ax1.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.25,
)

for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)
    ax2.spines[spine].set_visible(False)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),
    ncol=3,
    frameon=False,
    fontsize=9,
)

ax1.set_title(
    "Effect of channel composition on model performance",
    fontsize=11,
    pad=22,
)

plt.tight_layout()

save_path = os.path.join(SAVE_DIR, SAVE_NAME)
fig.savefig(save_path, dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"[OK] Saved: {save_path}")