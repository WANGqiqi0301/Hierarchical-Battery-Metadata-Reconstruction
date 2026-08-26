# -*- coding: utf-8 -*-
"""
plot_fig5g_material_conditioning.py

Figure 5g:
Soft vs hard material-conditioning ablation using SOC/SOH MedAE.

Read from:
    results/ablation/material_conditioning_ablation/
    material_conditioning_ablation_summary.csv

Output:
    results/figures/main/fig5g/fig5g_material_conditioning.png
    results/figures/main/fig5g/fig5g_material_conditioning_pure.png
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

HARD_CSV = (
    "results/ablation/material_conditioning_ablation/"
    "material_hard/metrics_stable_eval/"
    "final_metrics_nmc500.csv"
)

SOFT_CSV = (
    "results/proposed_framework/further_analysis/"
    "tables/proposed_method_summary.csv"
)

SAVE_DIR = "results/figures/main/fig5g"

FULL_SAVE_NAME = "fig5g_material_conditioning.png"
PURE_SAVE_NAME = "fig5g_material_conditioning_pure.png"

os.makedirs(
    SAVE_DIR,
    exist_ok=True,
)


# =============================================================================
# Load soft result from proposed further-analysis TEST
# =============================================================================

if not os.path.exists(SOFT_CSV):
    raise FileNotFoundError(
        f"Soft result CSV not found: {SOFT_CSV}"
    )

soft_df = pd.read_csv(SOFT_CSV)

if "split" not in soft_df.columns:
    raise RuntimeError(
        f"'split' column missing from soft CSV: {SOFT_CSV}"
    )

soft_test_df = soft_df.loc[
    soft_df["split"]
    .astype(str)
    .str.strip()
    .str.lower()
    == "test"
]

if soft_test_df.empty:
    raise RuntimeError(
        "No split='test' row found in proposed summary."
    )

soft_row = soft_test_df.iloc[0]

required_soft_cols = [
    "soc_medae",
    "soh_medae",
]

missing_soft = [
    col
    for col in required_soft_cols
    if col not in soft_df.columns
]

if missing_soft:
    raise RuntimeError(
        f"Missing required columns in soft CSV: {missing_soft}\n"
        f"Available columns: {list(soft_df.columns)}"
    )


# =============================================================================
# Load latest hard n_mc=500 result
# =============================================================================

if not os.path.exists(HARD_CSV):
    raise FileNotFoundError(
        f"Hard n_mc=500 result CSV not found: {HARD_CSV}"
    )

hard_df = pd.read_csv(HARD_CSV)

if hard_df.empty:
    raise RuntimeError(
        f"Hard n_mc=500 result CSV is empty: {HARD_CSV}"
    )

required_hard_cols = [
    "test_soc_medae_raw",
    "test_soh_medae_raw",
]

missing = [
    col
    for col in required_hard_cols
    if col not in hard_df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns in hard CSV: {missing}\n"
        f"Available columns: {list(hard_df.columns)}"
    )

hard_row = hard_df.iloc[0]


# =============================================================================
# Prepare data
# =============================================================================

df = pd.DataFrame(
    [
        {
            "condition": "Soft",
            "soc_medae_raw": float(
                soft_row["soc_medae"]
            ),
            "soh_medae_raw": float(
                soft_row["soh_medae"]
            ),
        },
        {
            "condition": "Hard",
            "soc_medae_raw": float(
                hard_row["test_soc_medae_raw"]
            ),
            "soh_medae_raw": float(
                hard_row["test_soh_medae_raw"]
            ),
        },
    ]
)

CONDITION_ORDER = [
    "Soft",
    "Hard",
]

df["condition"] = pd.Categorical(
    df["condition"],
    categories=CONDITION_ORDER,
    ordered=True,
)

df = (
    df
    .sort_values("condition")
    .reset_index(drop=True)
)

plot_df = pd.DataFrame(
    {
        "Method": df["condition"].astype(str),
        "SOC error": df["soc_medae_raw"].astype(float),
        "SOH error": df["soh_medae_raw"].astype(float),
    }
)
# =============================================================================
# Style
# =============================================================================

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "Soft": "#B08A78",
    "Hard": "#5F7F95",
}

MARKERS = {
    "Soft": "s",
    "Hard": "o",
}


# =============================================================================
# Plot
# =============================================================================

def plot_figure(
    plot_df: pd.DataFrame,
    pure: bool = False,
) -> None:

    metrics = [
        "SOC error",
        "SOH error",
    ]

    y_pos = (
        np.arange(
            len(metrics)
        )[::-1]
    )

    fig_w_cm = 5
    fig_h_cm = 4.4

    fig, ax = plt.subplots(
        figsize=(
            fig_w_cm / 2.54,
            fig_h_cm / 2.54,
        ),
        constrained_layout=False,
    )

    # =========================================================================
    # Values
    # =========================================================================

    soft_vals = (
        plot_df.loc[
            plot_df["Method"] == "Soft",
            metrics,
        ]
        .iloc[0]
        .to_numpy(dtype=float)
    )

    hard_vals = (
        plot_df.loc[
            plot_df["Method"] == "Hard",
            metrics,
        ]
        .iloc[0]
        .to_numpy(dtype=float)
    )

    # =========================================================================
    # Connecting lines
    # =========================================================================

    for i, y in enumerate(y_pos):
        ax.plot(
            [
                soft_vals[i],
                hard_vals[i],
            ],
            [
                y,
                y,
            ],
            color="#B8B8B8",
            lw=1.4,
            zorder=1,
            solid_capstyle="round",
        )

    # =========================================================================
    # Scatter
    # =========================================================================

    for method in [
        "Soft",
        "Hard",
    ]:
        vals = (
            plot_df.loc[
                plot_df["Method"] == method,
                metrics,
            ]
            .iloc[0]
            .to_numpy(dtype=float)
        )

        ax.scatter(
            vals,
            y_pos,
            s=36,
            color=COLORS[method],
            marker=MARKERS[method],
            edgecolors="none",
            label=method,
            zorder=3,
        )

        if not pure:
            for x, y in zip(
                vals,
                y_pos,
            ):
                ax.text(
                    x,
                    y + 0.14,
                    f"{x:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color=COLORS[method],
                )

    # =========================================================================
    # Axis limits
    # =========================================================================

    all_vals = np.concatenate(
        [
            soft_vals,
            hard_vals,
        ]
    )

    xmin = np.nanmin(all_vals)
    xmax = np.nanmax(all_vals)

    pad = max(
        (xmax - xmin) * 0.35,
        0.4,
    )

    ax.set_xlim(
        max(
            0,
            xmin - pad,
        ),
        xmax + pad,
    )

    ax.set_ylim(
        -0.55,
        len(metrics) - 0.45,
    )

    # =========================================================================
    # Full version
    # =========================================================================

    if not pure:
        ax.set_yticks(
            y_pos
        )

        ax.set_yticklabels(
            [
                "SOC",
                "SOH",
            ]
        )

        ax.set_xlabel(
            "MedAE (%)"
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(
            axis="x",
            color="#E6E6E6",
            lw=0.5,
            zorder=0,
        )

        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                1.25,
            ),
            ncol=2,
            handletextpad=0.4,
            columnspacing=1.0,
            fontsize=6.5,
        )

    # =========================================================================
    # Pure version
    # =========================================================================

    else:
        ax.set_yticks(
            y_pos
        )

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.tick_params(
            axis="both",
            length=0,
        )

        ax.set_xlabel("")
        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_visible(False)

    # =========================================================================
    # Layout
    # =========================================================================

    plt.subplots_adjust(
        left=0.24,
        right=0.98,
        bottom=0.25,
        top=0.80,
    )

    # =========================================================================
    # Save
    # =========================================================================

    if pure:
        save_name = PURE_SAVE_NAME
    else:
        save_name = FULL_SAVE_NAME

    save_path = os.path.join(
        SAVE_DIR,
        save_name,
    )

    fig.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
        transparent=pure,
    )

    plt.close(fig)

    print(
        f"[OK] Saved: {save_path}"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("[METRIC] Soft SOC/SOH <- soc_medae / soh_medae")
    print("[METRIC] Hard SOC/SOH <- test_soc_medae_raw / test_soh_medae_raw")

    print(
        "\n========== Figure 5g data =========="
    )

    print(
        plot_df.to_string(
            index=False
        )
    )

    plot_figure(
        plot_df=plot_df,
        pure=False,
    )

    plot_figure(
        plot_df=plot_df,
        pure=True,
    )


if __name__ == "__main__":
    main()