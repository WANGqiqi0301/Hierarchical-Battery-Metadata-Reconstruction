# -*- coding: utf-8 -*-
"""
plot_fig3d_soc_bin_median_ape.py

Figure 3d (panel 1):
Median SOC APE by SOC bin.

This script reproduces the original full reference-style plot and saves
only one final PNG.

Default input:
    results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv

Default output:
    results/figures/main/fig3d/soc_bin_median_ape_SOLID_REF.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Default paths
# =============================================================================
DEFAULT_INPUT_CSV = (
    r"results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv"
)
DEFAULT_OUTPUT_DIR = r"results/figures/main/fig3d"
DEFAULT_OUTPUT_NAME = "soc_bin_median_ape_SOLID_REF.png"


# =============================================================================
# Argument parser
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 3d median SOC APE by SOC bin plot."
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default=DEFAULT_INPUT_CSV,
        help="Path to test_predictions_per_sample.csv.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the output figure.",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help="Output figure filename.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure resolution.",
    )

    return parser.parse_args()


# =============================================================================
# Utilities
# =============================================================================
def read_csv_robust(csv_path: str) -> pd.DataFrame:
    """Read CSV with encoding fallback."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    encodings = ["utf-8-sig", "utf-8", "cp1252", "gbk"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"Failed to read CSV with supported encodings. Last error: {last_error}",
    )


def validate_input_columns(df: pd.DataFrame) -> None:
    required_cols = {"soc_true", "soc_pred", "soc_bin"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")


def get_bin_start(bin_str) -> float:
    """Extract the left boundary from a bin string for sorting."""
    try:
        return float(str(bin_str).strip("()[]").split(",")[0])
    except Exception:
        return 0.0


def compute_soc_bin_median_ape(df: pd.DataFrame) -> pd.DataFrame:
    """Compute median SOC APE for each SOC bin."""
    validate_input_columns(df)

    work = df.copy()
    work = work.dropna(subset=["soc_true", "soc_pred", "soc_bin"]).copy()

    if work.empty:
        raise ValueError("No valid rows after dropping NaN values.")

    if "soc_ape_pct" not in work.columns:
        work["ape"] = (
            np.abs((work["soc_true"] - work["soc_pred"]) / (work["soc_true"] + 1e-5))
            * 100.0
        )
    else:
        work["ape"] = work["soc_ape_pct"].astype(float)

    bin_stats = work.groupby("soc_bin", observed=True)["ape"].median().reset_index()
    bin_stats["start_val"] = bin_stats["soc_bin"].apply(get_bin_start)
    bin_stats = bin_stats.sort_values("start_val", ascending=True).reset_index(drop=True)

    return bin_stats


# =============================================================================
# Plot
# =============================================================================
def plot_soc_bin_median_ape(
    bin_stats: pd.DataFrame,
    save_path: str,
    dpi: int = 300,
) -> None:
    """
    Plot the full reference-style SOC-bin median APE bar chart.

    Visual settings are intentionally kept the same as the original code.
    """
    plt.rcParams["font.family"] = "Arial"

    x_labels = bin_stats["soc_bin"].astype(str)
    y_values = bin_stats["ape"].values
    x_pos = np.arange(len(x_labels))

    # Keep colors exactly the same as the original code
    peacock_blue = "#5D6D70"
    axis_color = "#2F3E46"

    y_max = float(y_values.max())
    y_top = y_max * 1.2 if y_max > 0 else 1.0

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        x_pos,
        y_values,
        width=0.7,
        color=peacock_blue,
        edgecolor="none",
        alpha=1.0,
        zorder=2,
    )

    ax.set_ylim(0, y_top)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)

    ax.set_ylabel(
        "Median APE (%)",
        fontsize=12,
        fontweight="bold",
        labelpad=10,
    )

    ax.set_xlabel(
        "SOC Range (%)",
        fontsize=12,
        fontweight="bold",
        labelpad=10,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(axis_color)
    ax.spines["bottom"].set_color(axis_color)

    ax.yaxis.grid(
        True,
        linestyle="--",
        alpha=0.2,
        zorder=0,
    )

    # Keep annotation style exactly the same
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (y_max * 0.02),
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=axis_color,
            fontweight="bold",
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, transparent=False)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()

    save_path = os.path.join(args.output_dir, args.output_name)

    df = read_csv_robust(args.input_csv)
    bin_stats = compute_soc_bin_median_ape(df)

    plot_soc_bin_median_ape(
        bin_stats=bin_stats,
        save_path=save_path,
        dpi=args.dpi,
    )

    print(f"[OK] Saved Figure 3d SOC-bin median APE plot: {save_path}")


if __name__ == "__main__":
    main()