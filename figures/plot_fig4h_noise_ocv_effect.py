# -*- coding: utf-8 -*-
"""
plot_fig4h_noise_robustness.py

Figure 4h:
Robustness to Gaussian noise applied to all voltage-response features.

Noise definition:
    Gaussian noise is added to standardized U1-U41 features.
    A noise level alpha corresponds to a noise standard deviation equal to
    alpha times the training-set standard deviation of each voltage feature.

Input:
    results/measurement_sensitivity/input_quality/noise_sensitivity_summary.csv

Output:
    results/figures/main/fig4h/fig4h_noise_robustness.png
    results/figures/main/fig4h/fig4h_noise_robustness_pure.png
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# Paths
# =============================================================================
INPUT_CSV = (
    "results/measurement_sensitivity/input_quality/"
    "noise_sensitivity_summary.csv"
)

SAVE_DIR = "results/figures/main/fig4h"
SAVE_NAME = "fig4h_noise_robustness.png"
PURE_SAVE_NAME = "fig4h_noise_robustness_pure.png"

os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# Load data
# =============================================================================
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"Input CSV not found: {INPUT_CSV}\n"
        "Please generate noise_sensitivity_summary.csv first."
    )

df = pd.read_csv(INPUT_CSV)

required_base_cols = [
    "type",
    "level",
]

missing_base_cols = [
    col for col in required_base_cols
    if col not in df.columns
]

if missing_base_cols:
    raise RuntimeError(
        f"Missing required columns in input CSV: {missing_base_cols}"
    )


# =============================================================================
# Resolve unified metric columns
# =============================================================================
def find_first_existing_column(dataframe: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in dataframe.columns:
            return col

    raise RuntimeError(
        "None of the required metric columns were found. "
        f"Tried: {candidates}\n"
        f"Available columns: {list(dataframe.columns)}"
    )


material_acc_col = find_first_existing_column(
    df,
    [
        "test_material_acc",
        "material_acc",
        "material_acc_mean",
    ],
)

soc_medae_col = find_first_existing_column(
    df,
    [
        "test_soc_medae_raw",
        "soc_medae_raw",
        "soc_medae_raw_mean",
    ],
)

soh_medae_col = find_first_existing_column(
    df,
    [
        "test_soh_medae_raw",
        "soh_medae_raw",
        "soh_medae_raw_mean",
    ],
)

print(f"[INFO] Material accuracy column: {material_acc_col}")
print(f"[INFO] SOC MedAE column: {soc_medae_col}")
print(f"[INFO] SOH MedAE column: {soh_medae_col}")


# =============================================================================
# Select full-input noise setting
# =============================================================================
df = df[df["type"].astype(str) == "noise"].copy()

full_mode_names = {
    "full_u1_u41_post_zscore",
    "full_u1_u41",
    "all_perturbed",
}

df_full = pd.DataFrame()

if "noise_mode" in df.columns:
    df_full = df[
        df["noise_mode"].astype(str).isin(full_mode_names)
    ].copy()

# Fall back to experiment column if needed
if df_full.empty and "experiment" in df.columns:
    df_full = df[
        df["experiment"].astype(str) == "noise_full"
    ].copy()

if df_full.empty:
    available_modes = (
        sorted(
            df["noise_mode"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if "noise_mode" in df.columns
        else []
    )

    available_experiments = (
        sorted(
            df["experiment"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if "experiment" in df.columns
        else []
    )

    raise RuntimeError(
        "No full-input noise rows were found.\n"
        f"Available noise_mode values: {available_modes}\n"
        f"Available experiment values: {available_experiments}"
    )


# =============================================================================
# Normalize metric names
# =============================================================================
df_full["material_acc"] = pd.to_numeric(
    df_full[material_acc_col],
    errors="coerce",
)

df_full["soc_medae_raw"] = pd.to_numeric(
    df_full[soc_medae_col],
    errors="coerce",
)

df_full["soh_medae_raw"] = pd.to_numeric(
    df_full[soh_medae_col],
    errors="coerce",
)


# =============================================================================
# Aggregate duplicate rows at the same noise level
# =============================================================================
df_full = (
    df_full.groupby("level", as_index=False)
    .agg(
        material_acc=("material_acc", "mean"),
        soc_medae_raw=("soc_medae_raw", "mean"),
        soh_medae_raw=("soh_medae_raw", "mean"),
    )
    .sort_values("level")
    .reset_index(drop=True)
)


# =============================================================================
# Prepare plotting values
# =============================================================================
# Convert alpha to percentage of training-set feature standard deviation:
#   0.001 -> 0.1%
#   0.010 -> 1%
#   0.050 -> 5%
x_percent = (
    df_full["level"]
    .to_numpy(dtype=float)
    * 100.0
)

material_acc_percent = (
    df_full["material_acc"]
    .to_numpy(dtype=float)
    * 100.0
)

soc_medae = (
    df_full["soc_medae_raw"]
    .to_numpy(dtype=float)
)

soh_medae = (
    df_full["soh_medae_raw"]
    .to_numpy(dtype=float)
)


# =============================================================================
# Plot settings
# =============================================================================
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
    }
)

fig, ax1 = plt.subplots(
    figsize=(4.35 / 2.2, 3.40 / 2.1),
    dpi=600,
)

ax2 = ax1.twinx()


# =============================================================================
# Colors
# =============================================================================
color_acc = "#4C78A8"
color_soc = "#E45756"
color_soh = "#59A14F"

background_color = "#FAFAFA"
grid_color = "#D9D9D9"
vertical_grid_color = "#E6E6E6"
spine_color = "#666666"


# =============================================================================
# Axis background
# =============================================================================
ax1.set_facecolor(background_color)
ax2.patch.set_alpha(0.0)


# =============================================================================
# Left axis: material accuracy
# =============================================================================
ax1.fill_between(
    x_percent,
    material_acc_percent,
    0,
    color=color_acc,
    alpha=0.10,
    linewidth=0,
    zorder=1,
)

line_acc = ax1.plot(
    x_percent,
    material_acc_percent,
    color=color_acc,
    marker="o",
    markersize=6.8,
    markerfacecolor=color_acc,
    markeredgecolor="white",
    markeredgewidth=1.0,
    linewidth=2.5,
    solid_capstyle="round",
    solid_joinstyle="round",
    label="Material accuracy",
    zorder=6,
)

ax1.set_xlabel(
    "Noise level (% of feature SD)"
)

ax1.set_ylabel(
    "Material accuracy (%)"
)

ax1.set_ylim(0, 100)

ax1.set_xticks(x_percent)

ax1.set_xticklabels(
    [
        f"{value:g}"
        for value in x_percent
    ]
)

ax1.tick_params(
    axis="both",
    direction="out",
    length=3,
    width=0.8,
    colors="black",
)


# =============================================================================
# Right axis: SOC and SOH MedAE
# =============================================================================
right_upper = max(
    float(soc_medae.max()),
    float(soh_medae.max()),
) * 1.15

# MedAE is usually much smaller than MedAPE, so do not force a 10% axis.
right_upper = max(
    1.0,
    right_upper,
)

ax2.fill_between(
    x_percent,
    soc_medae,
    0,
    color=color_soc,
    alpha=0.085,
    linewidth=0,
    zorder=2,
)

ax2.fill_between(
    x_percent,
    soh_medae,
    0,
    color=color_soh,
    alpha=0.085,
    linewidth=0,
    zorder=3,
)

line_soc = ax2.plot(
    x_percent,
    soc_medae,
    color=color_soc,
    marker="s",
    markersize=6.5,
    markerfacecolor=color_soc,
    markeredgecolor="white",
    markeredgewidth=1.0,
    linewidth=2.5,
    solid_capstyle="round",
    solid_joinstyle="round",
    label="SOC MedAE",
    zorder=7,
)

line_soh = ax2.plot(
    x_percent,
    soh_medae,
    color=color_soh,
    marker="^",
    markersize=7.0,
    markerfacecolor=color_soh,
    markeredgecolor="white",
    markeredgewidth=1.0,
    linewidth=2.5,
    solid_capstyle="round",
    solid_joinstyle="round",
    label="SOH MedAE",
    zorder=8,
)

ax2.set_ylabel(
    "MedAE (%)"
)

ax2.set_ylim(
    0,
    right_upper,
)

ax2.tick_params(
    axis="y",
    direction="out",
    length=3,
    width=0.8,
    colors="black",
)


# =============================================================================
# Grid and reference lines
# =============================================================================
ax1.grid(
    axis="y",
    linestyle="-",
    linewidth=0.55,
    color=grid_color,
    alpha=0.65,
    zorder=0,
)

for x_value in x_percent:
    ax1.axvline(
        x=x_value,
        color=vertical_grid_color,
        linewidth=0.45,
        alpha=0.55,
        zorder=0,
    )


# =============================================================================
# Endpoint annotations
# =============================================================================
ax1.annotate(
    f"{material_acc_percent[-1]:.1f}%",
    xy=(
        x_percent[-1],
        material_acc_percent[-1],
    ),
    xytext=(-6, -11),
    textcoords="offset points",
    ha="right",
    va="top",
    fontsize=8,
    color=color_acc,
)

ax2.annotate(
    f"{soc_medae[-1]:.2f}%",
    xy=(
        x_percent[-1],
        soc_medae[-1],
    ),
    xytext=(-6, 7),
    textcoords="offset points",
    ha="right",
    va="bottom",
    fontsize=8,
    color=color_soc,
)

ax2.annotate(
    f"{soh_medae[-1]:.2f}%",
    xy=(
        x_percent[-1],
        soh_medae[-1],
    ),
    xytext=(-6, 7),
    textcoords="offset points",
    ha="right",
    va="bottom",
    fontsize=8,
    color=color_soh,
)


# =============================================================================
# Appearance
# =============================================================================
ax1.set_title(
    "Robustness to Input Noise",
    pad=28,
)

ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)

ax1.spines["bottom"].set_color(spine_color)
ax1.spines["left"].set_color(spine_color)
ax2.spines["right"].set_color(spine_color)

ax1.spines["bottom"].set_linewidth(0.8)
ax1.spines["left"].set_linewidth(0.8)
ax2.spines["right"].set_linewidth(0.8)


# =============================================================================
# Combined legend
# =============================================================================
lines = (
    line_acc
    + line_soc
    + line_soh
)

labels = [
    line.get_label()
    for line in lines
]

ax1.legend(
    lines,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.16),
    ncol=3,
    frameon=False,
    handlelength=2.1,
    handletextpad=0.5,
    columnspacing=1.15,
    borderaxespad=0.0,
)


# =============================================================================
# Save full version
# =============================================================================
fig.tight_layout(pad=0.8)

save_path = os.path.join(
    SAVE_DIR,
    SAVE_NAME,
)

fig.savefig(
    save_path,
    dpi=600,
    bbox_inches="tight",
)

print(f"[OK] Saved: {save_path}")


# =============================================================================
# Save pure version
# =============================================================================
# Remove title and axis labels
ax1.set_title("")
ax1.set_xlabel("")
ax1.set_ylabel("")
ax2.set_ylabel("")

# Remove legend
legend = ax1.get_legend()
if legend is not None:
    legend.remove()

# Remove endpoint annotations
for text in list(ax1.texts):
    text.remove()

for text in list(ax2.texts):
    text.remove()

# Remove grid lines
ax1.grid(False)

# Remove vertical reference lines added using axvline
for line in list(ax1.lines):
    if line not in line_acc:
        line.remove()

# Remove tick labels and tick marks
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_yticks([])

ax1.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False,
)

ax2.tick_params(
    axis="both",
    which="both",
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelbottom=False,
    labelright=False,
)

# Remove all spines
for spine in ax1.spines.values():
    spine.set_visible(False)

for spine in ax2.spines.values():
    spine.set_visible(False)

# Tight margins
fig.subplots_adjust(
    left=0.01,
    right=0.99,
    bottom=0.01,
    top=0.99,
)

pure_save_path = os.path.join(
    SAVE_DIR,
    PURE_SAVE_NAME,
)

fig.savefig(
    pure_save_path,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.02,
    transparent=True,
)

plt.close(fig)

print(f"[OK] Saved pure version: {pure_save_path}")
print(
    "[INFO] Plotted full-input noise setting "
    "for standardized U1-U41 features."
)
print(
    "[INFO] Noise levels shown as percentages of "
    "the training-set feature standard deviation."
)
print(
    "[INFO] Metrics: material accuracy, SOC MedAE, SOH MedAE."
)
