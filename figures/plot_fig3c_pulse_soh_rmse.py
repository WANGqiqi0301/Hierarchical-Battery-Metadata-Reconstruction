# -*- coding: utf-8 -*-
"""
plot_fig3c_pulse_soh_rmse.py

Generate Figure 3c pulse-duration SOH RMSE plot.

This script reads per-sample prediction results and computes SOH RMSE
for each pulse duration. It saves:
1. A complete reference-style figure.
2. A pure slim figure with no axes, labels, text, or grid.

Default input:
    results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv

Required columns:
    pulse_ms, soh_true, soh_pred

Default outputs:
    results/figures/main/fig3c/pulse_soh_rmse_COMBO_REF.png
    results/figures/main/fig3c/pulse_soh_rmse_COMBO_REF_pure.png
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
DEFAULT_PURE_OUTPUT_NAME = "pulse_soh_rmse_COMBO_REF_pure.png"


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
        help="Directory to save the output figures.",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help="Complete output figure filename.",
    )

    parser.add_argument(
        "--pure_output_name",
        type=str,
        default=DEFAULT_PURE_OUTPUT_NAME,
        help="Pure output figure filename.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Complete output figure resolution.",
    )

    parser.add_argument(
        "--pure_dpi",
        type=int,
        default=600,
        help="Pure output figure resolution.",
    )

    parser.add_argument(
        "--fig_width",
        type=float,
        default=6.0,
        help="Complete figure width in inches.",
    )

    parser.add_argument(
        "--fig_height",
        type=float,
        default=5.0,
        help="Complete figure height in inches.",
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


def compute_y_limits(y_values: np.ndarray) -> tuple[float, float, float]:
    """Compute shared y-axis limits for complete and pure plots."""
    y_min = float(y_values.min())
    y_max = float(y_values.max())
    y_range = y_max - y_min

    if y_range == 0:
        y_range = max(abs(y_max), 1.0) * 0.1

    y_bottom = y_min - 0.2 * y_range
    y_top = y_max + 0.3 * y_range

    return y_bottom, y_top, y_range


# =============================================================================
# Plot: complete
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

    y_bottom, y_top, y_range = compute_y_limits(y_values)

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
# Plot: pure
# =============================================================================
def plot_pulse_soh_rmse_pure(
    pulse_stats: pd.DataFrame,
    save_path: str,
    dpi: int = 600,
) -> None:
    """
    Plot pure slim SOH RMSE figure.

    The pure canvas and save settings are intentionally matched to
    the SOC pure figure:
    - figsize=(1.55, 0.32)
    - dpi=600 by default
    - axes fill the entire figure
    - no axes / ticks / labels / text / grid
    - transparent background
    - pad_inches=0
    """
    x_labels = pulse_stats["Pulse Duration (ms)"].astype(str)
    y_values = pulse_stats["SOH_RMSE"].values
    x_pos = np.arange(len(x_labels))

    bar_color = "#BDC3C7"
    line_color = "#76448A"

    y_bottom, y_top, _ = compute_y_limits(y_values)

    # Match SOC pure figure exactly.
    pure_figsize = (1.55, 0.32)

    fig_clean, ax_clean = plt.subplots(
        figsize=pure_figsize,
        dpi=dpi,
    )

    ax_clean.bar(
        x_pos,
        y_values,
        width=0.8,
        color=bar_color,
        edgecolor="none",
        alpha=0.9,
        zorder=1,
    )

    ax_clean.plot(
        x_pos,
        y_values,
        color=line_color,
        linewidth=1.0,
        linestyle="-",
        marker="o",
        markersize=4,
        markerfacecolor=line_color,
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=2,
    )

    ax_clean.set_ylim(y_bottom, y_top)
    ax_clean.set_xlim(-0.5, len(x_pos) - 0.5)

    ax_clean.set_axis_off()

    fig_clean.patch.set_alpha(0)
    ax_clean.patch.set_alpha(0)

    # Important: SOC uses the full figure canvas.
    fig_clean.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
    )

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )

    # Important: match SOC save settings exactly.
    plt.savefig(
        save_path,
        dpi=dpi,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close(fig_clean)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()

    save_path = os.path.join(args.output_dir, args.output_name)
    pure_save_path = os.path.join(args.output_dir, args.pure_output_name)

    df = read_csv_robust(args.input_csv)
    pulse_stats = compute_pulse_soh_rmse(df)

    plot_pulse_soh_rmse(
        pulse_stats=pulse_stats,
        save_path=save_path,
        dpi=args.dpi,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
    )

    plot_pulse_soh_rmse_pure(
        pulse_stats=pulse_stats,
        save_path=pure_save_path,
        dpi=args.pure_dpi,
    )

    print(f"[OK] Saved Figure 3c SOH RMSE plot: {save_path}")
    print(f"[OK] Saved Figure 3c SOH RMSE pure plot: {pure_save_path}")


if __name__ == "__main__":
    main()
