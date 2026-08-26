
# -*- coding: utf-8 -*-
"""
plot_fig5b_hierarchy_ablation.py

Figure 5b:
Hierarchy ablation comparison.

Bar:
    MedAE (%)

Dot:
    MAE (%)

Mapping:
    hierarchical -> Full Hierarchical
    soc_to_soh   -> Hierarchical (no material)
    independent  -> Direct

Input:
    results/ablation/hierarchy_ablation/
        hierarchy_ablation_summary.csv

Output:
    results/figures/main/fig5b/
        fig5b_hierarchy_ablation.png
        fig5b_hierarchy_ablation_pure.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =============================================================================
# 1. Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "ablation"
    / "hierarchy_ablation"
    / "hierarchy_ablation_summary.csv"
)

SAVE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "main"
    / "fig5b"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. Plot configuration
# =============================================================================

MODE_ORDER = [
    "hierarchical",
    "soc_to_soh",
    "independent",
]

MODE_LABELS = {
    "hierarchical": "Full Hierarchical",
    "soc_to_soh": "Hierarchical\n(no material)",
    "independent": "Direct",
}

MODE_DISPLAY = {
    "hierarchical": "Full Hierarchical",
    "soc_to_soh": "Hierarchical (no material)",
    "independent": "Direct",
}


# =============================================================================
# 3. Load data
# =============================================================================

def load_plot_data():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Hierarchy ablation summary not found:\n"
            f"{INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    mode_col = None

    for candidate in [
        "struct_mode",
        "config",
    ]:
        if candidate in df.columns:
            mode_col = candidate
            break

    if mode_col is None:
        raise KeyError(
            "Could not find 'struct_mode' or 'config' "
            "in hierarchy ablation summary.\n"
            f"Available columns:\n"
            f"{df.columns.tolist()}"
        )

    df = df.copy()

    df["_mode"] = (
        df[mode_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    rows = {}

    for mode in MODE_ORDER:
        matched = df.loc[
            df["_mode"] == mode
        ]

        if matched.empty:
            raise RuntimeError(
                f"Could not find mode='{mode}' in:\n"
                f"{INPUT_CSV}\n\n"
                f"Available modes:\n"
                f"{df['_mode'].tolist()}"
            )

        rows[mode] = matched.iloc[0]

    models = [
        MODE_LABELS[mode]
        for mode in MODE_ORDER
    ]

    # -------------------------------------------------------------------------
    # SOC
    # -------------------------------------------------------------------------

    soc_data = {
        "medae": np.array(
            [
                float(
                    rows[mode][
                        "test_soc_medae_raw"
                    ]
                )
                for mode in MODE_ORDER
            ],
            dtype=float,
        ),

        "mae": np.array(
            [
                float(
                    rows[mode][
                        "test_soc_mae_raw"
                    ]
                )
                for mode in MODE_ORDER
            ],
            dtype=float,
        ),
    }

    # -------------------------------------------------------------------------
    # SOH
    # -------------------------------------------------------------------------

    soh_data = {
        "medae": np.array(
            [
                float(
                    rows[mode][
                        "test_soh_medae_raw"
                    ]
                )
                for mode in MODE_ORDER
            ],
            dtype=float,
        ),

        "mae": np.array(
            [
                float(
                    rows[mode][
                        "test_soh_mae_raw"
                    ]
                )
                for mode in MODE_ORDER
            ],
            dtype=float,
        ),
    }

    # -------------------------------------------------------------------------
    # Result check
    # -------------------------------------------------------------------------

    print("\n[FIGURE 5b DATA]")

    for mode in MODE_ORDER:
        row = rows[mode]

        print(
            f"{MODE_DISPLAY[mode]:30s} | "
            f"SOC MedAE="
            f"{float(row['test_soc_medae_raw']):.3f}% | "
            f"SOC MAE="
            f"{float(row['test_soc_mae_raw']):.3f}% | "
            f"SOH MedAE="
            f"{float(row['test_soh_medae_raw']):.3f}% | "
            f"SOH MAE="
            f"{float(row['test_soh_mae_raw']):.3f}%"
        )

    return (
        models,
        soc_data,
        soh_data,
    )


# =============================================================================
# 4. Plot
# =============================================================================

def run_plot(
    models,
    soc_data,
    soh_data,
    mode: str = "full",
):
    plt.rcParams["font.family"] = "Arial"

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(3.5, 1.8),
        sharey=True,
        dpi=600,
    )

    if mode == "full":
        plt.subplots_adjust(
            wspace=0.10,
            bottom=0.27,
        )
    else:
        plt.subplots_adjust(
            wspace=0.10,
        )

    # =========================================================================
    # Panel drawing
    # =========================================================================

    def draw_panel(
        ax,
        data,
        title,
        x_ticks,
        is_left=True,
    ):
        y_pos = np.arange(
            len(models)
        )

        # ---------------------------------------------------------------------
        # Bars: MedAE
        # ---------------------------------------------------------------------

        bar_colors = [
            "#D3D3D3",
            "#D3D3D3",
            "#404040",
        ]

        ax.barh(
            y_pos,
            data["medae"],
            height=0.60,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.8,
            zorder=2,
        )

        # ---------------------------------------------------------------------
        # Dashed connection:
        # MedAE -> MAE
        # ---------------------------------------------------------------------

        for i in range(
            len(models)
        ):
            ax.plot(
                [
                    data["medae"][i],
                    data["mae"][i],
                ],
                [
                    y_pos[i],
                    y_pos[i],
                ],
                color="#888888",
                linestyle="--",
                linewidth=1.0,
                zorder=1,
            )

        # ---------------------------------------------------------------------
        # Dots: MAE
        # ---------------------------------------------------------------------

        dot_colors = [
            "white",
            "white",
            "black",
        ]

        for i in range(
            len(models)
        ):
            ax.scatter(
                data["mae"][i],
                y_pos[i],
                s=55,
                color=dot_colors[i],
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )

        # ---------------------------------------------------------------------
        # Axis range
        # ---------------------------------------------------------------------

        ax.set_xticks(
            x_ticks
        )

        max_val = max(
            float(
                np.max(
                    data["medae"]
                )
            ),
            float(
                np.max(
                    data["mae"]
                )
            ),
        )

        ax.set_xlim(
            0,
            max_val * 1.15,
        )

        # ---------------------------------------------------------------------
        # Full version
        # ---------------------------------------------------------------------

        if mode == "full":
            ax.set_title(
                title,
                fontsize=9,
                fontweight="bold",
                pad=8,
            )

            ax.spines[
                "top"
            ].set_visible(False)

            ax.spines[
                "right"
            ].set_visible(False)

            ax.tick_params(
                axis="x",
                labelsize=9,
            )

            if is_left:
                ax.set_yticks(
                    y_pos
                )

                ax.set_yticklabels(
                    models,
                    fontsize=9,
                )

            else:
                ax.spines[
                    "left"
                ].set_visible(False)

                ax.yaxis.set_ticks_position(
                    "none"
                )

        # ---------------------------------------------------------------------
        # Pure version
        # ---------------------------------------------------------------------

        else:
            ax.axis(
                "off"
            )

    # =========================================================================
    # Draw two panels
    # =========================================================================

    draw_panel(
        ax=ax1,
        data=soc_data,
        title="SOC Error (%)",
        x_ticks=[
            0,
            5,
            10,
            15,
        ],
        is_left=True,
    )

    draw_panel(
        ax=ax2,
        data=soh_data,
        title="SOH Error (%)",
        x_ticks=[
            0,
            2,
            4,
            6,
        ],
        is_left=False,
    )

    # =========================================================================
    # Legend
    # =========================================================================

    if mode == "full":
        legend_handles = [
            Patch(
                facecolor="#D3D3D3",
                edgecolor="black",
                linewidth=0.8,
                label="MedAE",
            ),

            Line2D(
                [0],
                [0],
                marker="o",
                color="#888888",
                linestyle="--",
                linewidth=1.0,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=7,
                label="MAE",
            ),
        ]

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(
                0.57,
                0.01,
            ),
            ncol=2,
            frameon=False,
            fontsize=8,
            handlelength=1.8,
            columnspacing=1.3,
            handletextpad=0.5,
        )

    # =========================================================================
    # Save
    # =========================================================================

    if mode == "full":
        filename = (
            "fig5b_hierarchy_ablation.png"
        )
    else:
        filename = (
            "fig5b_hierarchy_ablation_pure.png"
        )

    output_path = (
        SAVE_DIR
        / filename
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
        transparent=(
            mode == "clean"
        ),
        dpi=600,
    )

    plt.close(
        fig
    )

    print(
        f"[SAVED] {output_path}"
    )


# =============================================================================
# 5. Main
# =============================================================================

def main():
    (
        models,
        soc_data,
        soh_data,
    ) = load_plot_data()

    run_plot(
        models=models,
        soc_data=soc_data,
        soh_data=soh_data,
        mode="full",
    )

    run_plot(
        models=models,
        soc_data=soc_data,
        soh_data=soh_data,
        mode="clean",
    )


if __name__ == "__main__":
    main()

