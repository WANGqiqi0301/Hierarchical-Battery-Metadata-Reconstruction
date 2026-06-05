# -*- coding: utf-8 -*-
"""
plot_fig5f_hierarchy_order_bubble.py

Figure 5f:
Hierarchy-order trade-off bubble plot.

Reads hierarchy-order ablation CSV output from:
    ablation/hierarchy_order_ablation.py

Only supports new code output format:
    order, cls_acc, soc_medape_raw, soh_medape_raw

Output:
    results/figures/main/fig5f/fig5f_hierarchy_order_bubble.png
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Paths
# =============================================================================
INPUT_CSV = "results/ablation/hierarchy_order_ablation/hierarchy_order_ablation_summary.csv"
SAVE_DIR = "results/figures/main/fig5f"
SAVE_NAME = "fig5f_hierarchy_order_bubble.png"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================================================================
# Labels
# =============================================================================
ORDER_LIST = [
    "PARALLEL",
    "SOH_M_SOC",
    "SOC_M_SOH",
    "M_SOH_SOC",
    "M_SOC_SOH",
]

ORDER_LABELS = {
    "PARALLEL": "Parallel",
    "SOH_M_SOC": "SOH→M→SOC",
    "SOC_M_SOH": "SOC→M→SOH",
    "M_SOH_SOC": "M→SOH→SOC",
    "M_SOC_SOH": "M→SOC→SOH",
}

# =============================================================================
# Load and prepare dataframe
# =============================================================================
df = pd.read_csv(INPUT_CSV)

required_cols = ["order", "cls_acc", "soc_medape_raw", "soh_medape_raw"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns in CSV: {missing}")

df = df[df["order"].isin(ORDER_LIST)].copy()
df["order_label"] = df["order"].map(ORDER_LABELS)
df["classification_accuracy_pct"] = df["cls_acc"] * 100.0
df["soc_median_ape_pct"] = df["soc_medape_raw"]
df["soh_median_ape_pct"] = df["soh_medape_raw"]

df["order"] = pd.Categorical(df["order"], categories=ORDER_LIST, ordered=True)
df = df.sort_values("order").reset_index(drop=True)

# =============================================================================
# Plot
# =============================================================================
soc = df["soc_median_ape_pct"].to_numpy(dtype=float)
soh = df["soh_median_ape_pct"].to_numpy(dtype=float)
acc = df["classification_accuracy_pct"].to_numpy(dtype=float)

# Bubble size proportional to classification accuracy
sizes = 520 + (acc - acc.min()) / (acc.max() - acc.min() + 1e-8) * 900

colors = {
    "PARALLEL": "#9FA1A4",
    "SOH_M_SOC": "#7B94A8",
    "SOC_M_SOH": "#7FA49A",
    "M_SOH_SOC": "#A48FB3",
    "M_SOC_SOH": "#5E7F9A",
}

fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
ax.scatter(
    soc,
    soh,
    s=sizes,
    c=[colors[str(order)] for order in df["order"]],
    linewidth=0,
    alpha=0.95,
    zorder=3,
)

ax.set_xlim(soc.min() * 0.94, soc.max() * 1.03)
ax.set_ylim(soh.min() * 0.91, soh.max() * 1.07)

# Annotate order labels
for _, row in df.iterrows():
    dx, dy = 0.03, 0.025
    if row["order"] == "M_SOC_SOH":
        dy = -0.05
    ax.text(
        row["soc_median_ape_pct"] + dx,
        row["soh_median_ape_pct"] + dy,
        row["order_label"],
        fontsize=8,
        color="0.25",
    )

# Legend for bubble sizes
legend_vals = np.linspace(acc.min(), acc.max(), 3)
legend_sizes = 520 + (legend_vals - acc.min()) / (acc.max() - acc.min() + 1e-8) * 900
handles = [ax.scatter([], [], s=size, color="#9FA1A4") for size in legend_sizes]
ax.legend(
    handles,
    [f"{v:.1f}%" for v in legend_vals],
    title="Material acc.",
    loc="lower right",
    frameon=False,
    fontsize=7,
    title_fontsize=8,
)

ax.set_xlabel("SOC MedAPE (%)", fontsize=10)
ax.set_ylabel("SOH MedAPE (%)", fontsize=10)
ax.set_title("Hierarchy-order trade-off", fontsize=11, pad=8)
ax.grid(linestyle="--", linewidth=0.4, alpha=0.22)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, SAVE_NAME), dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"[OK] Saved: {os.path.join(SAVE_DIR, SAVE_NAME)}")