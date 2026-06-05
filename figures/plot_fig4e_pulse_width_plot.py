# -*- coding: utf-8 -*-
"""
plot_fig4e_pulse_width_plot.py

Generate Figure 4e:
SOH Median APE vs pulse-width configurations.

Only the standard version with text is saved.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# 1. Paths
# =========================
BASE_DIR = "results/measurement_sensitivity/pulse_width"
SUMMARY_CSV = os.path.join(BASE_DIR, "pulse_width_sensitivity_summary.csv")
SAVE_DIR = "results/figures/main/fig4e"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# 2. Load data
# =========================
if not os.path.exists(SUMMARY_CSV):
    raise FileNotFoundError(f"Summary CSV not found: {SUMMARY_CSV}")

df = pd.read_csv(SUMMARY_CSV)

custom_order = [
    "P1_70",
    "P2_3000",
    "P3_30_50_70_100",
    "P4_300_500_700",
    "P5_1000_3000_5000",
    "P6_30_50_300_500",
    "P7_30_50_3000_5000",
    "P8_300_500_3000_5000",
    "P9_All",
]

df["config"] = pd.Categorical(
    df["config"],
    categories=custom_order,
    ordered=True,
)
df = df.sort_values("config",ascending=False).reset_index(drop=True)

df["plot_label"] = df["config"].astype(str).str.extract(r"^(P\d+)")


# =========================
# 3. Plot
# =========================
x_data = df["SOH Median APE"].values
y_labels = df["plot_label"].values
n_train = df["num_pulse_widths"].values

cmap = sns.color_palette("rocket", as_cmap=True)
norm = plt.Normalize(vmin=0, vmax=max(n_train))
bubble_colors = [cmap(norm(size)) for size in n_train]

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    sharey=True,
    figsize=(12, 8),
    dpi=150,
    gridspec_kw={"width_ratios": [3, 1]},
)

plt.subplots_adjust(wspace=0.1, right=0.8)
fig.patch.set_alpha(0.0)

size_base = max(n_train) if max(n_train) > 0 else 1
point_sizes = (n_train / size_base) * 1800

for ax in [ax1, ax2]:
    ax.patch.set_alpha(0.0)

    ax.hlines(
        y=y_labels,
        xmin=0,
        xmax=x_data,
        color="#CCCCCC",
        alpha=0.3,
        linewidth=1,
        zorder=1,
    )

    ax.scatter(
        x_data,
        y_labels,
        s=point_sizes,
        c=bubble_colors,
        alpha=0.8,
        edgecolors="#111111",
        linewidth=0.8,
        zorder=3,
        clip_on=False,
    )

    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=11, length=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Broken x-axis ranges
left_mask = x_data < 10
right_mask = x_data > 10

if left_mask.any():
    ax1.set_xlim(
        min(x_data[left_mask]) - 0.8,
        max(x_data[left_mask]) + 0.8,
    )

if right_mask.any():
    ax2.set_xlim(
        min(x_data[right_mask]) - 1.0,
        max(x_data[right_mask]) + 1.0,
    )



# Hide duplicated y-axis labels on the right panel
ax2.tick_params(labelleft=False, left=False)
ax2.spines["left"].set_visible(False)

# Axis labels
ax1.set_ylabel(
    "Pulse-width configuration",
    fontsize=12,
    fontweight="bold",
)

fig.text(
    0.42,
    0.03,
    "SOH Median APE (%)",
    ha="center",
    fontsize=12,
    fontweight="bold",
)

# Broken-axis markers
d = 0.015
kwargs = dict(transform=ax1.transAxes, color="#333333", clip_on=False, lw=1.2)
ax1.plot((1 - d / 3, 1 + d / 3), (-d, +d), **kwargs)
ax1.plot((1 - d / 3, 1 + d / 3), (1 - d, 1 + d), **kwargs)

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (-d, +d), **kwargs)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

# Legend for bubble size
legend_sizes = [1, 3, 4, 10]
legend_handles = []

for s in legend_sizes:
    size = (s / size_base) * 1800
    h = ax2.scatter(
        [],
        [],
        s=size,
        c=[cmap(norm(s))],
        alpha=0.85,
        edgecolors="#111111",
        linewidth=0.8,
    )
    legend_handles.append(h)

ax2.legend(
    legend_handles,
    [str(s) for s in legend_sizes],
    title="Number of pulse widths",
    loc="center left",
    bbox_to_anchor=(1.05, 0.5),
    frameon=False,
    fontsize=10,
    title_fontsize=11,
)

# Title
fig.suptitle(
    "SOH prediction error by pulse-width configuration",
    fontsize=15,
    fontweight="bold",
    x=0.12,
    y=0.96,
    ha="left",
)

# Value labels
bbox_props = dict(
    boxstyle="round,pad=0.2",
    facecolor="#FFFFFF90",
    edgecolor="none",
    zorder=4,
)

for i, val in enumerate(x_data):
    target_ax = ax1 if val < 10 else ax2
    target_ax.text(
        val + 0.25,
        i,
        f"{val:.2f}%",
        va="center",
        fontsize=10,
        fontweight="bold" if y_labels[i] == "P9" else "normal",
        bbox=bbox_props,
    )


# =========================
# 4. Save
# =========================
out_file = os.path.join(SAVE_DIR, "fig4e_pulse_width_performance.png")
fig.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"[OK] Figure 4e saved at: {out_file}")