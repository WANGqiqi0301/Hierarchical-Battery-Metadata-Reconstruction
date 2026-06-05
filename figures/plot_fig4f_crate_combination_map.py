# -*- coding: utf-8 -*-
"""
plot_fig4f_crate_combination_map.py

Figure 4f:
C-rate combination map.

This script visualizes different C-rate combinations as a binary matrix.
Each row corresponds to one configuration and each column corresponds to
one C-rate value.

Default output:
    results/figures/main/fig4f/fig4f_crate_combination_map.png

Optional clean output:
    results/figures/main/fig4f/fig4f_crate_combination_map_clean.png
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# =============================================================================
# Configuration
# =============================================================================
DEFAULT_OUTPUT_DIR = r"results/figures/main/fig4f"

X_LABELS = [0.5, 1, 1.5, 2, 2.5]

COMBINATIONS = [
    [1],
    [3],
    [5],
    [1, 2],
    [3, 4],
    [4, 5],
    [1, 5],
    [1, 2, 3],
    [3, 4, 5],
    [1, 3, 5],
    [1, 2, 3, 4, 5],
]

COLOR_BG = "#F2F4F7"
COLOR_ACTIVE = "#2E5984"
COLOR_GRID = "#FFFFFF"

DPI = 600


# =============================================================================
# Style
# =============================================================================
def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.linewidth": 0.8,
            "savefig.dpi": DPI,
        }
    )


# =============================================================================
# Arguments
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 4f C-rate combination map."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output figures.",
    )

    parser.add_argument(
        "--save_clean",
        action="store_true",
        help="Also save a clean version without text labels.",
    )

    return parser.parse_args()


# =============================================================================
# Data
# =============================================================================
def build_combination_matrix(
    combinations: list[list[int]],
    n_cols: int,
) -> np.ndarray:
    """
    Build a binary matrix where each row is one combination and each column
    indicates whether the corresponding C-rate is included.
    """
    matrix = np.zeros((len(combinations), n_cols), dtype=int)

    for i, combo in enumerate(combinations):
        for val in combo:
            idx = val - 1
            if 0 <= idx < n_cols:
                matrix[i, idx] = 1

    return matrix


# =============================================================================
# Plotting
# =============================================================================
def plot_crate_combination_map(
    matrix: np.ndarray,
    output_dir: str,
    with_text: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    cmap = ListedColormap([COLOR_BG, COLOR_ACTIVE])

    fig_size = (7, 8) if with_text else (6, 7.5)
    fig, ax = plt.subplots(figsize=fig_size, facecolor="white")

    ax.imshow(
        matrix,
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
        origin="upper",
    )

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(
        which="minor",
        color=COLOR_GRID,
        linestyle="-",
        linewidth=4,
    )

    if with_text:
        ax.set_xticks(np.arange(len(X_LABELS)))
        ax.set_xticklabels(
            X_LABELS,
            fontsize=11,
            fontweight="bold",
            color="#333333",
        )
        ax.xaxis.tick_top()

        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(
            [f"C{i+1}" for i in range(matrix.shape[0])],
            fontsize=11,
            color="#555555",
        )

        ax.tick_params(which="both", length=0)

        save_name = "fig4f_crate_combination_map.png"
    else:
        ax.set_xticks(np.arange(len(X_LABELS)))
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(which="both", length=0)

        save_name = "fig4f_crate_combination_map_clean.png"

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    save_path = os.path.join(output_dir, save_name)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    set_plot_style()

    matrix = build_combination_matrix(
        combinations=COMBINATIONS,
        n_cols=len(X_LABELS),
    )

    plot_crate_combination_map(
        matrix=matrix,
        output_dir=args.output_dir,
        with_text=True,
    )

    if args.save_clean:
        plot_crate_combination_map(
            matrix=matrix,
            output_dir=args.output_dir,
            with_text=False,
        )

    print("[DONE] Figure 4f C-rate combination map generated.")


if __name__ == "__main__":
    main()