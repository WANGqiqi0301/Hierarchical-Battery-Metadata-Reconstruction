# -*- coding: utf-8 -*-
"""
plot_fig4g_drop_robustness.py

Figure 4g:
Robustness to missing features.

Input:
    results/measurement_sensitivity/input_quality/drop_sensitivity_aggregated.csv

Output:
    results/figures/main/fig4g/fig4g_drop_robustness.png
    results/figures/main/fig4g/fig4g_drop_robustness_pure.png

Primary metrics:
    Material accuracy
    SOC MedAE
    SOH MedAE
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

INPUT_CSV = (
    "results/measurement_sensitivity/input_quality/"
    "drop_sensitivity_aggregated.csv"
)

SAVE_DIR = "results/figures/main/fig4g"

SAVE_NAME = "fig4g_drop_robustness.png"
PURE_SAVE_NAME = "fig4g_drop_robustness_pure.png"

os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# Plot settings
# =============================================================================

FIG_SIZE = (5.36, 4.6)
DPI = 600

BAR_ALPHA = 0.60
BAND_ALPHA = 0.20

LINE_WIDTH = 1.0
MARKER_SIZE = 12

ERROR_LINE_WIDTH = 1.5
ERROR_CAP_THICKNESS = 1.5
ERROR_CAP_SIZE = 3


# =============================================================================
# Helpers
# =============================================================================

def _find_column(df: pd.DataFrame, candidates: list[str], metric_name: str) -> str:
    """Return the first existing column from candidates."""
    for col in candidates:
        if col in df.columns:
            return col

    raise RuntimeError(
        f"Missing required column for {metric_name}. "
        f"Expected one of: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


# =============================================================================
# Load data
# =============================================================================

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"Input CSV not found: {INPUT_CSV}\n"
        "Please run "
        "measurement_sensitivity/input_quality_sensitivity.py first."
    )


df = pd.read_csv(INPUT_CSV)

if "drop_count" not in df.columns:
    raise RuntimeError(
        "Missing required column in input CSV: drop_count"
    )

# Prefer the new unified metric field names, while keeping limited
# compatibility with possible test_* aggregated naming.
material_acc_mean_col = _find_column(
    df,
    [
        "material_acc_mean",
        "test_material_acc_mean",
    ],
    "material accuracy mean",
)

material_acc_std_col = _find_column(
    df,
    [
        "material_acc_std",
        "test_material_acc_std",
    ],
    "material accuracy std",
)

soc_medae_mean_col = _find_column(
    df,
    [
        "soc_medae_raw_mean",
        "test_soc_medae_raw_mean",
    ],
    "SOC MedAE mean",
)

soc_medae_std_col = _find_column(
    df,
    [
        "soc_medae_raw_std",
        "test_soc_medae_raw_std",
    ],
    "SOC MedAE std",
)

soh_medae_mean_col = _find_column(
    df,
    [
        "soh_medae_raw_mean",
        "test_soh_medae_raw_mean",
    ],
    "SOH MedAE mean",
)

soh_medae_std_col = _find_column(
    df,
    [
        "soh_medae_raw_std",
        "test_soh_medae_raw_std",
    ],
    "SOH MedAE std",
)


df = df.sort_values("drop_count").reset_index(drop=True)


# =============================================================================
# Extract arrays
# =============================================================================

x = df["drop_count"].to_numpy(dtype=float)

material_acc_mean = df[material_acc_mean_col].to_numpy(dtype=float)
material_acc_std = df[material_acc_std_col].to_numpy(dtype=float)

soc_medae_mean = df[soc_medae_mean_col].to_numpy(dtype=float)
soc_medae_std = df[soc_medae_std_col].to_numpy(dtype=float)

soh_medae_mean = df[soh_medae_mean_col].to_numpy(dtype=float)
soh_medae_std = df[soh_medae_std_col].to_numpy(dtype=float)


# =============================================================================
# Determine bar width
# =============================================================================

if len(x) <= 1:
    bar_width = 0.8
elif np.max(x) >= 1:
    bar_width = 0.8
else:
    bar_width = float(np.min(np.diff(np.sort(x)))) * 0.8


# =============================================================================
# Common plotting function
# =============================================================================

def draw_figure(pure: bool = False):
    """
    Draw the missing-feature robustness figure.

    Parameters
    ----------
    pure : bool
        False:
            Draw the complete publication version.

        True:
            Remove title, axis labels, tick labels, legend, and grid,
            while retaining the bars, error bars, curves, and uncertainty
            bands.
    """

    fig, ax1 = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    # -------------------------------------------------------------------------
    # Material accuracy bars
    # -------------------------------------------------------------------------
    ax1.bar(
        x,
        material_acc_mean,
        width=bar_width,
        alpha=BAR_ALPHA,
        edgecolor="none",
        linewidth=0,
        yerr=material_acc_std,
        capsize=ERROR_CAP_SIZE,
        error_kw={
            "lw": ERROR_LINE_WIDTH,
            "capthick": ERROR_CAP_THICKNESS,
        },
    )

    ax1.set_ylim(0, 1.0)

    # Give the first and last bars enough horizontal space.
    if len(x) > 0:
        ax1.set_xlim(
            np.min(x) - bar_width * 0.75,
            np.max(x) + bar_width * 0.75,
        )

    # -------------------------------------------------------------------------
    # SOC and SOH MedAE curves
    # -------------------------------------------------------------------------
    ax2 = ax1.twinx()

    soc_lower = np.maximum(
        soc_medae_mean - soc_medae_std,
        0.0,
    )
    soc_upper = soc_medae_mean + soc_medae_std

    soh_lower = np.maximum(
        soh_medae_mean - soh_medae_std,
        0.0,
    )
    soh_upper = soh_medae_mean + soh_medae_std

    ax2.fill_between(
        x,
        soc_lower,
        soc_upper,
        alpha=BAND_ALPHA,
        linewidth=0,
    )

    ax2.fill_between(
        x,
        soh_lower,
        soh_upper,
        alpha=BAND_ALPHA,
        linewidth=0,
    )

    ax2.plot(
        x,
        soc_medae_mean,
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label="SOC MedAE",
    )

    ax2.plot(
        x,
        soh_medae_mean,
        marker="^",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        label="SOH MedAE",
    )

    # -------------------------------------------------------------------------
    # Complete version
    # -------------------------------------------------------------------------
    if not pure:
        ax1.set_xlabel("Number of Missing Features")
        ax1.set_ylabel("Material Accuracy")
        ax2.set_ylabel("MedAE (%)")

        ax1.set_title("Robustness to Missing Features")
        ax1.grid(alpha=0.3)

        lines2, labels2 = ax2.get_legend_handles_labels()

        fig.legend(
            lines2,
            labels2,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=2,
            fontsize=8,
            frameon=False,
        )

    # -------------------------------------------------------------------------
    # Pure version
    # -------------------------------------------------------------------------
    else:
        ax1.set_xlabel("")
        ax1.set_ylabel("")
        ax2.set_ylabel("")
        ax1.set_title("")

        ax1.grid(False)
        ax2.grid(False)

        # Remove all tick marks and tick labels.
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

        # Remove axis spines.
        for spine in ax1.spines.values():
            spine.set_visible(False)

        for spine in ax2.spines.values():
            spine.set_visible(False)

    plt.tight_layout()

    return fig


# =============================================================================
# Save standard figure
# =============================================================================

fig_standard = draw_figure(pure=False)

save_path = os.path.join(SAVE_DIR, SAVE_NAME)

fig_standard.savefig(
    save_path,
    dpi=DPI,
    bbox_inches="tight",
)

plt.close(fig_standard)

print(f"[OK] Saved standard figure: {save_path}")


# =============================================================================
# Save pure figure
# =============================================================================

fig_pure = draw_figure(pure=True)

pure_save_path = os.path.join(SAVE_DIR, PURE_SAVE_NAME)

fig_pure.savefig(
    pure_save_path,
    dpi=DPI,
    bbox_inches="tight",
    pad_inches=0.02,
    transparent=True,
)

plt.close(fig_pure)

print(f"[OK] Saved pure figure: {pure_save_path}")
