# -*- coding: utf-8 -*-
"""
plot_fig3c_pulse_soh_rmse.py

Generate Figure 3c pulse-duration SOH RMSE plot.

This script reads per-sample prediction results and computes SOH RMSE
for each pulse duration. The visual style follows the original full
non-gradient SOH RMSE reference plot.

Default input:
    results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv

Required columns:
    pulse_ms, soh_true, soh_pred

Default output:
    results/figures/main/fig3c/pulse_soh_rmse_COMBO_REF.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Default configuration
# =============================================================================
DEFAULT_INPUT_CSV = (
    r"results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv"
)

DEFAULT_OUTPUT_DIR = r"results/figures/main/fig3c"
DEFAULT_OUTPUT_NAME = "pulse_soh_rmse_COMBO_REF.png"


# =============================================================================
# Argument parser
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 3c pulse-duration SOH RMSE plot."
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

    parser.add_argument(
        "--fig_width",
        type=float,
        default=6.0,
        help="Figure width in inches.",
    )

    parser.add_argument(
        "--fig_height",
        type=float,
        default=5.0,
        help="Figure height in inches.",
    )

    return parser.parse_args()


# =============================================================================
# Data utilities
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
    """Validate required columns."""
    required_cols = {"pulse_ms", "soh_true", "soh_pred"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")


def compute_pulse_soh_rmse(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SOH RMSE for each pulse duration."""
    validate_input_columns(df)

    work = df.copy()
    work = work.dropna(subset=["pulse_ms", "soh_true", "soh_pred"])

    if work.empty:
        raise ValueError("No valid rows after dropping NaN values.")

    work["pulse_ms"] = work["pulse_ms"].astype(float)
    work["soh_true"] = work["soh_true"].astype(float)
    work["soh_pred"] = work["soh_pred"].astype(float)

    def calculate_soh_rmse(group: pd.DataFrame) -> float:
        mse = np.mean((group["soh_true"] - group["soh_pred"]) ** 2)
        return float(np.sqrt(mse))

    pulse_stats = work.groupby("pulse_ms").apply(calculate_soh_rmse).reset_index()
    pulse_stats.columns = ["Pulse Duration (ms)", "SOH_RMSE"]
    pulse_stats = pulse_stats.sort_values("Pulse Duration (ms)").reset_index(drop=True)

    return pulse_stats


# =============================================================================
# Plot
# =============================================================================
def plot_pulse_soh_rmse(
    pulse_stats: pd.DataFrame,
    save_path: str,
    dpi: int = 300,
    fig_width: float = 6.0,
    fig_height: float = 5.0,
) -> None:
    """Plot full non-gradient SOH RMSE bar-line figure."""
    plt.rcParams["font.family"] = "Arial"

    x_labels = pulse_stats["Pulse Duration (ms)"].astype(str)
    y_values = pulse_stats["SOH_RMSE"].values
    x_pos = np.arange(len(x_labels))

    bar_color = "#BDC3C7"
    line_color = "#76448A"

    y_min = float(y_values.min())
    y_max = float(y_values.max())
    y_range = y_max - y_min

    if y_range == 0:
        y_range = max(abs(y_max), 1.0) * 0.1

    y_bottom = y_min - 0.2 * y_range
    y_top = y_max + 0.3 * y_range

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    ax.bar(
        x_pos,
        y_values,
        width=0.5,
        color=bar_color,
        edgecolor="none",
        alpha=0.6,
        zorder=2,
    )

    ax.plot(
        x_pos,
        y_values,
        color=line_color,
        linewidth=2.5,
        marker="o",
        markersize=8,
        markerfacecolor=line_color,
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=3,
    )

    ax.set_ylim(y_bottom, y_top)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)

    ax.set_ylabel(
        "RMSE (SOH)",
        fontsize=12,
        fontweight="bold",
        labelpad=10,
    )

    ax.set_xlabel(
        "Pulse Duration (ms)",
        fontsize=12,
        fontweight="bold",
        labelpad=10,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.yaxis.grid(
        True,
        linestyle="--",
        alpha=0.3,
        zorder=0,
    )

    for i, val in enumerate(y_values):
        ax.text(
            i,
            val + (y_range * 0.05),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=line_color,
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()

    save_path = os.path.join(args.output_dir, args.output_name)

    df = read_csv_robust(args.input_csv)
    pulse_stats = compute_pulse_soh_rmse(df)

    plot_pulse_soh_rmse(
        pulse_stats=pulse_stats,
        save_path=save_path,
        dpi=args.dpi,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
    )

    print(f"[OK] Saved Figure 3c SOH RMSE plot: {save_path}")


if __name__ == "__main__":
    main()