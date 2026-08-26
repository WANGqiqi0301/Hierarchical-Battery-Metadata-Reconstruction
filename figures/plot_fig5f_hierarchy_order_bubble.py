
# -*- coding: utf-8 -*-
"""
plot_fig5f_hierarchy_order_bubble.py

Figure 5f:
Hierarchy-order trade-off bubble plot.

Reads hierarchy-order ablation CSV output from:
    ablation/hierarchy_order_ablation.py

Only supports unified output format:
    order, test_material_acc, test_soc_medae_raw, test_soh_medae_raw

Output:
    results/figures/main/fig5f/fig5f_hierarchy_order_bubble.png
    results/figures/main/fig5f/fig5f_hierarchy_order_bubble_pure.png
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
    "results/ablation/hierarchy_order_ablation/"
    "hierarchy_order_ablation_summary.csv"
)

SAVE_DIR = "results/figures/main/fig5f"

FULL_SAVE_NAME = "fig5f_hierarchy_order_bubble.png"
PURE_SAVE_NAME = "fig5f_hierarchy_order_bubble_pure.png"

os.makedirs(
    SAVE_DIR,
    exist_ok=True,
)


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

def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        INPUT_CSV
    )

    required_cols = [
        "order",
        "test_material_acc",
        "test_soc_medae_raw",
        "test_soh_medae_raw",
    ]

    missing = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns in CSV: {missing}"
        )

    df = df[
        df["order"].isin(
            ORDER_LIST
        )
    ].copy()

    df["order_label"] = (
        df["order"]
        .map(ORDER_LABELS)
    )

    df["material_accuracy_pct"] = (
        df["test_material_acc"]
        .astype(float)
        * 100.0
    )

    df["soc_medae_pct"] = (
        df["test_soc_medae_raw"]
        .astype(float)
    )

    df["soh_medae_pct"] = (
        df["test_soh_medae_raw"]
        .astype(float)
    )

    df["order"] = pd.Categorical(
        df["order"],
        categories=ORDER_LIST,
        ordered=True,
    )

    df = (
        df
        .sort_values("order")
        .reset_index(drop=True)
    )

    print("[METRIC] Bubble size <- test_material_acc")
    print("[METRIC] X-axis      <- test_soc_medae_raw")
    print("[METRIC] Y-axis      <- test_soh_medae_raw")
    print("\n[FINAL DATA SENT TO MATPLOTLIB]")
    print(
        df[[
            "order",
            "test_material_acc",
            "test_soc_medae_raw",
            "test_soh_medae_raw",
        ]].to_string(index=False)
    )

    return df


# =============================================================================
# Plot
# =============================================================================

def plot_figure(
    df: pd.DataFrame,
    pure: bool = False,
) -> None:

    # =========================================================================
    # Data
    # =========================================================================

    soc = (
        df["soc_medae_pct"]
        .to_numpy(dtype=float)
    )

    soh = (
        df["soh_medae_pct"]
        .to_numpy(dtype=float)
    )

    acc = (
        df["material_accuracy_pct"]
        .to_numpy(dtype=float)
    )

    # =========================================================================
    # Bubble size
    # =========================================================================

    sizes = (
        520
        + (
            acc - acc.min()
        )
        / (
            acc.max()
            - acc.min()
            + 1e-8
        )
        * 900
    )

    # =========================================================================
    # Colors
    # =========================================================================

    colors = {
        "PARALLEL": "#9FA1A4",
        "SOH_M_SOC": "#7B94A8",
        "SOC_M_SOH": "#7FA49A",
        "M_SOH_SOC": "#A48FB3",
        "M_SOC_SOH": "#5E7F9A",
    }

    # =========================================================================
    # Figure
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(6/1.5, 3.5/1.5),
        dpi=600,
    )

    # =========================================================================
    # Bubble plot
    # =========================================================================

    ax.scatter(
        soc,
        soh,
        s=sizes,
        c=[
            colors[str(order)]
            for order in df["order"]
        ],
        linewidth=0,
        alpha=0.95,
        zorder=3,
        clip_on=False,
    )

    # =========================================================================
    # Limits
    # =========================================================================

    # =========================================================================
    # Limits
    # =========================================================================

    soc_range = soc.max() - soc.min()
    soh_range = soh.max() - soh.min()

    x_pad = max(
        soc_range * 0.20,
        0.35,
    )

    y_pad = max(
        soh_range * 0.25,
        0.08,
    )

    ax.set_xlim(
        soc.min() - x_pad,
        soc.max() + x_pad,
    )

    ax.set_ylim(
        soh.min() - y_pad,
        soh.max() + y_pad,
    )

    # =========================================================================
    # Full version
    # =========================================================================

    if not pure:

        # ---------------------------------------------------------------------
        # Order labels
        # ---------------------------------------------------------------------

        for _, row in df.iterrows():
            dx = 0.03
            dy = 0.025

            if row["order"] == "M_SOC_SOH":
                dy = -0.05

            ax.text(
                row["soc_medae_pct"] + dx,
                row["soh_medae_pct"] + dy,
                row["order_label"],
                fontsize=8,
                color="0.25",
            )

        # ---------------------------------------------------------------------
        # Bubble-size legend
        # ---------------------------------------------------------------------

        legend_vals = np.linspace(
            acc.min(),
            acc.max(),
            3,
        )

        legend_sizes = (
            520
            + (
                legend_vals - acc.min()
            )
            / (
                acc.max()
                - acc.min()
                + 1e-8
            )
            * 900
        )

        handles = [
            ax.scatter(
                [],
                [],
                s=size,
                color="#9FA1A4",
            )
            for size in legend_sizes
        ]

        ax.legend(
            handles,
            [
                f"{value:.1f}%"
                for value in legend_vals
            ],
            title="Material acc.",
            loc="lower right",
            frameon=False,
            fontsize=7,
            title_fontsize=8,
        )

        # ---------------------------------------------------------------------
        # Axis labels and title
        # ---------------------------------------------------------------------

        ax.set_xlabel(
            "SOC MedAE (%)",
            fontsize=10,
        )

        ax.set_ylabel(
            "SOH MedAE (%)",
            fontsize=10,
        )

        ax.set_title(
            "Hierarchy-order trade-off",
            fontsize=11,
            pad=8,
        )

        # ---------------------------------------------------------------------
        # Grid
        # ---------------------------------------------------------------------

        ax.grid(
            linestyle="--",
            linewidth=0.4,
            alpha=0.22,
        )

        # ---------------------------------------------------------------------
        # Spines
        # ---------------------------------------------------------------------

        for spine in [
            "top",
            "right",
        ]:
            ax.spines[
                spine
            ].set_visible(False)

    # =========================================================================
    # Pure version
    # =========================================================================

    else:

        # Remove tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Remove tick marks
        ax.tick_params(
            axis="both",
            length=0,
        )

        # Remove axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Remove grid
        ax.grid(False)

        # Remove all spines
        for spine in ax.spines.values():
            spine.set_visible(False)

    # =========================================================================
    # Layout
    # =========================================================================

    plt.tight_layout()

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

    plt.close(
        fig
    )

    print(
        f"[OK] Saved: {save_path}"
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("[VERSION] FIG5F_LATEST_SUMMARY_TEST_FIELDS")
    df = load_data()

    plot_figure(
        df=df,
        pure=False,
    )

    plot_figure(
        df=df,
        pure=True,
    )


if __name__ == "__main__":
    main()

