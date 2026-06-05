# -*- coding: utf-8 -*-
"""
plot_fig4e_pulse_width_matrix.py

Figure 4e:
Pulse-width configuration matrix.

This script saves one standard version with text labels only.

Output:
    results/figures/main/fig4e/fig4e_pulse_width_matrix.png
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR = Path("results") / "figures" / "main" / "fig4e"
OUTPUT_NAME = "fig4e_pulse_width_matrix.png"
DPI = 600

X_LABELS = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

COMBINATIONS = [
    [70],
    [3000],
    [30, 50, 70, 100],
    [300, 500, 700],
    [1000, 3000, 5000],
    [30, 50, 300, 500],
    [30, 50, 3000, 5000],
    [300, 500, 3000, 5000],
    [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000],
]

COLOR_BG = "#F2F4F7"
COLOR_ACTIVE = "#2E5984"
COLOR_GRID = "#FFFFFF"


# =============================================================================
# Helpers
# =============================================================================
def build_matrix(x_labels: list[int], combinations: list[list[int]]) -> np.ndarray:
    matrix = np.zeros((len(combinations), len(x_labels)), dtype=float)
    for i, combo in enumerate(combinations):
        for val in combo:
            if val in x_labels:
                matrix[i, x_labels.index(val)] = 1.0
    return matrix


# =============================================================================
# Plot
# =============================================================================
def plot_pulse_width_matrix() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix(X_LABELS, COMBINATIONS)
    cmap = ListedColormap([COLOR_BG, COLOR_ACTIVE])

    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor="white")

    ax.imshow(
        matrix,
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
        origin="upper",
    )

    ax.set_xticks(np.arange(-0.5, len(X_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(COMBINATIONS), 1), minor=True)
    ax.grid(which="minor", color=COLOR_GRID, linestyle="-", linewidth=4)

    ax.set_xticks(np.arange(len(X_LABELS)))
    ax.set_xticklabels(
        X_LABELS,
        fontsize=10,
        fontweight="bold",
        color="#333333",
    )
    ax.xaxis.tick_top()

    ax.set_yticks(np.arange(len(COMBINATIONS)))
    ax.set_yticklabels(
        [f"P{i+1}" for i in range(len(COMBINATIONS))],
        fontsize=10,
        color="#555555",
    )

    ax.tick_params(which="both", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    save_path = OUTPUT_DIR / OUTPUT_NAME
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    plot_pulse_width_matrix()
    print("[DONE] Figure 4e pulse-width matrix generated.")


if __name__ == "__main__":
    main()