# -*- coding: utf-8 -*-
"""
plot_fig4c_soc_error_distribution.py

Figure 4c:
SOC absolute error distribution under ideal and realistic
material-information settings.

Input:
    results/analysis/error_propagation/e0_predictions_per_sample.csv
    results/analysis/error_propagation/e1_predictions_per_sample.csv

Output:
    results/figures/main/fig4c/fig4c_soc_error_distribution.png
    results/figures/main/fig4c/fig4c_soc_error_distribution_pure.png
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# =============================================================================
# 1. Paths
# =============================================================================
INPUT_DIR = os.path.join(
    "results",
    "analysis",
    "error_propagation",
)

E0_CSV = os.path.join(
    INPUT_DIR,
    "e0_predictions_per_sample.csv",
)

E1_CSV = os.path.join(
    INPUT_DIR,
    "e1_predictions_per_sample.csv",
)

SAVE_DIR = os.path.join(
    "results",
    "figures",
    "main",
    "fig4c",
)

SAVE_NAME = "fig4c_soc_error_distribution.png"
PURE_SAVE_NAME = "fig4c_soc_error_distribution_pure.png"


# =============================================================================
# 2. Style configuration
# =============================================================================
sns.set_theme(style="white")

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2
plt.rcParams["lines.linewidth"] = 1.5


# =============================================================================
# 3. Colors
# =============================================================================
BOX_COLORS = [
    "#6B8EAC",  # E0 / Ideal
    "#BC8585",  # E1 / Realistic
]

LINE_COLOR = "#2D2D2D"
TEXT_COLOR = "#2D2D2D"


# =============================================================================
# 4. Data loading
# =============================================================================
def load_soc_error_statistics(csv_path: str) -> dict:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = [
        "soc_true",
        "soc_pred",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"Missing required columns in {csv_path}: "
            f"{missing_columns}"
        )

    soc_true = df["soc_true"].to_numpy(
        dtype=np.float64
    )

    soc_pred = df["soc_pred"].to_numpy(
        dtype=np.float64
    )

    # Absolute error on the original SOC scale
    soc_ae = np.abs(soc_pred - soc_true)

    return {
        "median": float(np.median(soc_ae)),
        "p25": float(np.percentile(soc_ae, 25)),
        "p75": float(np.percentile(soc_ae, 75)),
        "p05": float(np.percentile(soc_ae, 5)),
        "p95": float(np.percentile(soc_ae, 95)),
        "mean": float(np.mean(soc_ae)),
        "n": int(len(soc_ae)),
    }


def load_data() -> dict:
    return {
        "E0": load_soc_error_statistics(E0_CSV),
        "E1": load_soc_error_statistics(E1_CSV),
    }


# =============================================================================
# 5. Core drawing function
# =============================================================================
def draw_boxplot(
    ax: plt.Axes,
    data_dict: dict,
) -> None:
    labels = list(data_dict.keys())

    for i, label in enumerate(labels):
        d = data_dict[label]

        # Box: P25-P75
        ax.add_patch(
            plt.Rectangle(
                (
                    i - 0.28,
                    d["p25"],
                ),
                0.56,
                d["p75"] - d["p25"],
                facecolor=BOX_COLORS[i],
                edgecolor=LINE_COLOR,
                alpha=0.9,
                lw=1.5,
            )
        )

        # Median line
        ax.plot(
            [
                i - 0.28,
                i + 0.28,
            ],
            [
                d["median"],
                d["median"],
            ],
            color=LINE_COLOR,
            lw=2.2,
        )

        # Lower whisker
        ax.plot(
            [
                i,
                i,
            ],
            [
                d["p05"],
                d["p25"],
            ],
            color=LINE_COLOR,
            lw=1.5,
        )

        # Upper whisker
        ax.plot(
            [
                i,
                i,
            ],
            [
                d["p75"],
                d["p95"],
            ],
            color=LINE_COLOR,
            lw=1.5,
        )

        # Lower cap
        ax.plot(
            [
                i - 0.12,
                i + 0.12,
            ],
            [
                d["p05"],
                d["p05"],
            ],
            color=LINE_COLOR,
            lw=1.5,
        )

        # Upper cap
        ax.plot(
            [
                i - 0.12,
                i + 0.12,
            ],
            [
                d["p95"],
                d["p95"],
            ],
            color=LINE_COLOR,
            lw=1.5,
        )

    ax.set_xticks(
        range(len(labels))
    )
    ax.set_ylim(bottom=0)


# =============================================================================
# 6. Add numerical annotations
# =============================================================================
def add_value_annotations(
    ax: plt.Axes,
    data_dict: dict,
) -> None:
    labels = list(data_dict.keys())

    for i, label in enumerate(labels):
        d = data_dict[label]

        # Put E0 labels on the left, E1 labels on the right
        if i == 0:
            x_text = i - 0.38
            ha = "right"
        else:
            x_text = i + 0.38
            ha = "left"

        for key in ["p95", "p75", "median", "p25", "p05"]:
            ax.text(
                x_text,
                d[key],
                f'{d[key]:.2f}',
                ha=ha,
                va="center",
                fontsize=7,
                fontweight="bold",
                color=TEXT_COLOR,
            )


# =============================================================================
# 7. Academic version
# =============================================================================
def save_academic_version(
    data_dict: dict,
) -> None:
    fig, ax = plt.subplots(
        figsize=(2.8, 2.7)
    )

    draw_boxplot(
        ax=ax,
        data_dict=data_dict,
    )

    add_value_annotations(
        ax=ax,
        data_dict=data_dict,
    )

    ax.set_xticklabels(
        [
            "Ideal",
            "Realistic",
        ],
        fontweight="bold",
        fontsize=8,
    )

    ax.set_ylabel(
        "SOC absolute error (%)",
        fontweight="bold",
        fontsize=8,
    )

    sns.despine(
        ax=ax,
        trim=True,
        offset=4,
    )

    plt.tight_layout()

    file_path = os.path.join(
        SAVE_DIR,
        SAVE_NAME,
    )

    fig.savefig(
        file_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[OK] Saved: {file_path}")


# =============================================================================
# 8. Pure version
# =============================================================================
def save_pure_version(
    data_dict: dict,
) -> None:
    fig, ax = plt.subplots(
        figsize=(2.2, 2.5)
    )

    draw_boxplot(
        ax=ax,
        data_dict=data_dict,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.tick_params(
        axis="both",
        which="both",
        length=0,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(
        pad=0.1
    )

    file_path = os.path.join(
        SAVE_DIR,
        PURE_SAVE_NAME,
    )

    fig.savefig(
        file_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
    )

    plt.close(fig)

    print(f"[OK] Saved: {file_path}")


# =============================================================================
# 9. Main
# =============================================================================
def main() -> None:
    os.makedirs(
        SAVE_DIR,
        exist_ok=True,
    )

    data_dict = load_data()

    print("\n[SOC AE statistics]")

    for label, stats in data_dict.items():
        print(
            f"{label}: "
            f"mean={stats['mean']:.4f}, "
            f"median={stats['median']:.4f}, "
            f"P25={stats['p25']:.4f}, "
            f"P75={stats['p75']:.4f}, "
            f"P05={stats['p05']:.4f}, "
            f"P95={stats['p95']:.4f}, "
            f"n={stats['n']}"
        )

    save_academic_version(
        data_dict=data_dict,
    )

    save_pure_version(
        data_dict=data_dict,
    )


if __name__ == "__main__":
    main()