# -*- coding: utf-8 -*-
"""
plot_fig4g_drop_robustness.py

Figure 4g:
Robustness to missing features.

Input:
    results/measurement_sensitivity/input_quality/drop_sensitivity_aggregated.csv

Output:
    results/figures/main/fig4g/fig4g_drop_robustness.png
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Paths
# =============================================================================
INPUT_CSV = "results/measurement_sensitivity/input_quality/drop_sensitivity_aggregated.csv"
SAVE_DIR = "results/figures/main/fig4g"
SAVE_NAME = "fig4g_drop_robustness.png"

os.makedirs(SAVE_DIR, exist_ok=True)

# =============================================================================
# Load data
# =============================================================================
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"Input CSV not found: {INPUT_CSV}\n"
        "Please run measurement_sensitivity/input_quality_sensitivity.py first."
    )

df = pd.read_csv(INPUT_CSV)

required_cols = [
    "drop_count",
    "cls_acc_mean",
    "cls_acc_std",
    "soc_medape_raw_mean",
    "soc_medape_raw_std",
    "soh_medape_raw_mean",
    "soh_medape_raw_std",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns in input CSV: {missing}")

df = df.sort_values("drop_count").reset_index(drop=True)

# =============================================================================
# Plot
# =============================================================================
fig, ax1 = plt.subplots(figsize=(5.36, 4.2), dpi=600)

x = df["drop_count"].to_numpy(dtype=float)
acc_mean = df["cls_acc_mean"].to_numpy(dtype=float)
acc_std = df["cls_acc_std"].to_numpy(dtype=float)
soc_mean = df["soc_medape_raw_mean"].to_numpy(dtype=float)
soc_std = df["soc_medape_raw_std"].to_numpy(dtype=float)
soh_mean = df["soh_medape_raw_mean"].to_numpy(dtype=float)
soh_std = df["soh_medape_raw_std"].to_numpy(dtype=float)

# -------------------------
# Accuracy bar with error bar
# -------------------------
bar_width = 0.8 if np.max(x) >= 1 else (x[1] - x[0]) * 0.8

ax1.bar(
    x,
    acc_mean,
    width=bar_width,
    alpha=0.6,
    edgecolor="none",
    linewidth=0,
    yerr=acc_std,
    capsize=3,
    error_kw=dict(lw=1.5, capthick=1.5),
)

ax1.set_ylabel("Classification Accuracy")
ax1.set_ylim(0, 1.0)
ax1.set_xlabel("Number of Missing Features")

# -------------------------
# SOC / SOH lines with shaded band
# -------------------------
ax2 = ax1.twinx()

ax2.fill_between(x, soc_mean - soc_std, soc_mean + soc_std, alpha=0.20)
ax2.fill_between(x, soh_mean - soh_std, soh_mean + soh_std, alpha=0.20)

ax2.plot(x, soc_mean, marker="o", linewidth=1, markersize=12, label="SOC error")
ax2.plot(x, soh_mean, marker="^", linewidth=1, markersize=12, label="SOH error")

ax2.set_ylabel("Median APE (%)")
ax1.set_title("Robustness to Missing Features")
ax1.grid(alpha=0.3)

# -------------------------
# Legend outside top center
# -------------------------
lines2, labels2 = ax2.get_legend_handles_labels()
fig.legend(
    lines2, labels2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=2,
    fontsize=8,
    frameon=False
)

plt.tight_layout()

# -------------------------
# Save figure
# -------------------------
save_path = os.path.join(SAVE_DIR, SAVE_NAME)
fig.savefig(save_path, dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"[OK] Saved: {save_path}")