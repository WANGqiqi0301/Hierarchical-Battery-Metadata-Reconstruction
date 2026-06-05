# -*- coding: utf-8 -*-
"""
plot_fig2c_soh_distribution_by_material.py

Figure 2c:
SOH distribution by material type.

This script reads SOH values from the original Excel files:
    data/<capacity material>/<material_capacity_W_30.xlsx>

For each material type:
    LMO, NMC, LFP

It aggregates all SOH values from the "SOC TEST RANDOM" sheet,
converts SOH to percentage, and plots a histogram with KDE.

Default output:
    results/figures/main/fig2c/LMO_soh_distribution.png
    results/figures/main/fig2c/NMC_soh_distribution.png
    results/figures/main/fig2c/LFP_soh_distribution.png
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Default configuration
# =============================================================================
DEFAULT_DATA_ROOT = "data"
DEFAULT_OUTPUT_DIR = "results/figures/main/fig2c"

FOLDERS = [
    "10Ah LMO",
    "15Ah NMC",
    "21Ah NMC",
    "24Ah LMO",
    "25Ah LMO",
    "26Ah LMO",
    "35Ah LFP",
    "68Ah LFP",
]

MATERIAL_ORDER = ["LMO", "NMC", "LFP"]

BAR_COLOR = "#7FA8C9"
KDE_COLOR = "#666666"
KDE_LINEWIDTH = 6

FIG_WIDTH = 8
FIG_HEIGHT = 6
DPI = 300


# =============================================================================
# Argument parser
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 2c SOH distribution plots by material."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing material-capacity folders.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output figures.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI,
        help="Output figure resolution.",
    )

    return parser.parse_args()


# =============================================================================
# Data loading
# =============================================================================
def build_excel_path(data_root: str, folder: str) -> str:
    """
    Build Excel path from folder name.

    Example:
        folder = "10Ah LMO"
        file  = "LMO_10Ah_W_30.xlsx"
    """
    parts = folder.split(" ")
    capacity = parts[0]
    material = parts[-1]

    file_name = f"{material}_{capacity}_W_30.xlsx"
    return os.path.join(data_root, folder, file_name)


def load_soh_by_material(
    data_root: str,
    folders: List[str],
) -> Tuple[Dict[str, List[float]], List[float]]:
    """
    Load SOH values from all material-capacity Excel files.

    SOH values are converted to percentage:
        SOH_percent = SOH * 100
    """
    material_data: Dict[str, List[float]] = {
        "LMO": [],
        "NMC": [],
        "LFP": [],
    }

    all_values: List[float] = []

    for folder in folders:
        parts = folder.split(" ")
        material_type = parts[-1]
        file_path = build_excel_path(data_root, folder)

        if not os.path.exists(file_path):
            print(f"[WARN] Missing file: {file_path}")
            continue

        try:
            df = pd.read_excel(
                file_path,
                sheet_name="SOC TEST RANDOM",
                usecols=["SOH"],
            )

            soh_percent = (df["SOH"].dropna().astype(float) * 100.0).tolist()

            material_data[material_type].extend(soh_percent)
            all_values.extend(soh_percent)

            print(
                f"[DATA] Loaded {folder}: "
                f"{len(soh_percent)} SOH values"
            )

        except Exception as exc:
            print(f"[WARN] Failed to read {file_path}: {exc}")

    return material_data, all_values


def compute_global_xlim(all_values: List[float]) -> Tuple[float, float]:
    """Compute global x-axis limits with 5% margin."""
    if not all_values:
        return 0.0, 100.0

    x_min = min(all_values)
    x_max = max(all_values)
    margin = (x_max - x_min) * 0.05

    return x_min - margin, x_max + margin


# =============================================================================
# Plotting
# =============================================================================
def plot_material_soh_distribution(
    material: str,
    values: List[float],
    x_limit: Tuple[float, float],
    save_path: str,
    dpi: int = DPI,
) -> None:
    """
    Plot SOH distribution for one material.

    Visual settings follow the original code exactly:
    - seaborn white style
    - figsize=(8, 6)
    - transparent background
    - histplot with kde=True
    - bar color #7FA8C9
    - KDE line color #666666
    - KDE linewidth 6
    """
    if not values:
        print(f"[WARN] No SOH values for {material}. Skipped.")
        return

    sns.set_style("white", {"axes.grid": False})

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor="none")
    ax = fig.add_subplot(111, facecolor="none")

    sns.histplot(
        values,
        kde=True,
        color=BAR_COLOR,
        stat="count",
        alpha=0.5,
        edgecolor=None,
        line_kws={
            "color": KDE_COLOR,
            "lw": KDE_LINEWIDTH,
        },
        ax=ax,
    )

    ax.set_xlim(x_limit)

    ax.set_title(
        f"{material} SOH Distribution",
        fontsize=14,
        color="black",
    )

    ax.set_xlabel("SOH (%)", color="black")
    ax.set_ylabel("Count", color="black")

    sns.despine()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        transparent=True,
    )

    plt.close(fig)

    print(
        f"[OK] Saved {material} SOH distribution: {save_path} "
        f"| xlim={x_limit[0]:.2f}-{x_limit[1]:.2f}"
    )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    material_data, all_values = load_soh_by_material(
        data_root=args.data_root,
        folders=FOLDERS,
    )

    global_xlim = compute_global_xlim(all_values)

    for material in MATERIAL_ORDER:
        save_path = os.path.join(
            args.output_dir,
            f"{material}_soh_distribution.png",
        )

        plot_material_soh_distribution(
            material=material,
            values=material_data.get(material, []),
            x_limit=global_xlim,
            save_path=save_path,
            dpi=args.dpi,
        )

    print("[DONE] Figure 2c SOH distribution plots generated.")


if __name__ == "__main__":
    main()