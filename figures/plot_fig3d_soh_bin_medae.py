# -*- coding: utf-8 -*-

"""
plot_fig3d_soh_bin_medae.py

Figure 3d:
SOH MedAE by SOH bin.

This script reproduces the original full reference-style plot and saves:

1. The complete reference PNG.
2. A pure slim PNG with only the bars.

Default input:
results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv

Default outputs:
results/figures/main/fig3d/soh_bin_medae_SOLID_REF.png
results/figures/main/fig3d/soh_bin_medae_SOLID_REF_pure.png
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

DEFAULT_OUTPUT_NAME = "soh_bin_medae_SOLID_REF.png"
DEFAULT_PURE_OUTPUT_NAME = "soh_bin_medae_SOLID_REF_pure.png"


# =============================================================================
# Argument parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 3d SOH MedAE by SOH bin plot."
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

    return parser.parse_args()


# =============================================================================
# Utilities
# =============================================================================

def read_csv_robust(csv_path: str) -> pd.DataFrame:
    """Read CSV with encoding fallback."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}"
        )

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "gbk",
    ]

    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(
                csv_path,
                encoding=enc,
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        (
            "Failed to read CSV with supported encodings. "
            f"Last error: {last_error}"
        ),
    )


def validate_input_columns(
    df: pd.DataFrame,
) -> None:
    required_cols = {
        "soh_true",
        "soh_pred",
    }

    missing_cols = (
        required_cols
        -
        set(df.columns)
    )

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {sorted(missing_cols)}"
        )


def normalize_soh_pair_to_percent(
    soh_true,
    soh_pred,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert SOH true/predicted values to percentage-point scale.

    If SOH appears to be stored on a 0-1 scale, both true and predicted
    values are multiplied by 100. Values already on a 0-100 scale are
    left unchanged.
    """
    soh_true = pd.to_numeric(
        pd.Series(soh_true),
        errors="coerce",
    ).to_numpy(dtype=float)

    soh_pred = pd.to_numeric(
        pd.Series(soh_pred),
        errors="coerce",
    ).to_numpy(dtype=float)

    finite_true = soh_true[
        np.isfinite(soh_true)
    ]

    if (
        len(finite_true) > 0
        and
        np.nanmax(np.abs(finite_true)) <= 1.5
    ):
        soh_true = soh_true * 100.0
        soh_pred = soh_pred * 100.0

    return soh_true, soh_pred


def get_bin_start(bin_str) -> float:
    """
    Extract the left boundary from a bin string for sorting.
    """
    try:
        return float(
            str(bin_str)
            .strip("()[]")
            .split(",")[0]
        )
    except Exception:
        return 0.0


def compute_soh_bin_medae(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute SOH MedAE for each SOH bin.
    """
    validate_input_columns(
        df
    )

    work = df.copy()

    # ---------------------------------------------------------
    # Convert SOH to reporting scale first
    # ---------------------------------------------------------
    soh_true_pct, soh_pred_pct = (
        normalize_soh_pair_to_percent(
            work["soh_true"],
            work["soh_pred"],
        )
    )

    work["soh_true"] = soh_true_pct
    work["soh_pred"] = soh_pred_pct

    # ---------------------------------------------------------
    # Match original further_analysis:
    # N_BINS = 10, BINNING = "uniform" -> pd.cut(x, bins=10)
    # ---------------------------------------------------------
    work["soh_bin"] = pd.cut(
        work["soh_true"],
        bins=10,
    )

    work = work.dropna(
        subset=[
            "soh_true",
            "soh_pred",
            "soh_bin",
        ]
    ).copy()

    if work.empty:
        raise ValueError(
            "No valid rows after dropping NaN values."
        )

    # ---------------------------------------------------------
    # Absolute error for MedAE
    # ---------------------------------------------------------
    recomputed_ae = np.abs(
        work["soh_pred"]
        -
        work["soh_true"]
    )

    if "soh_ae" in work.columns:
        existing_ae = pd.to_numeric(
            work["soh_ae"],
            errors="coerce",
        )

        finite_existing = existing_ae.dropna()

        if len(finite_existing) == 0:
            work["ae_soh"] = recomputed_ae

        else:
            existing_median = float(
                np.nanmedian(
                    np.abs(
                        existing_ae.to_numpy(dtype=float)
                    )
                )
            )

            recomputed_median = float(
                np.nanmedian(
                    recomputed_ae.to_numpy(dtype=float)
                )
            )

            if (
                np.isfinite(existing_median)
                and
                np.isfinite(recomputed_median)
                and
                recomputed_median > 0
                and
                (
                    existing_median / recomputed_median > 20
                    or
                    recomputed_median / max(existing_median, 1e-12) > 20
                )
            ):
                work["ae_soh"] = recomputed_ae
            else:
                work["ae_soh"] = existing_ae

    else:
        work["ae_soh"] = recomputed_ae

    work = work.dropna(
        subset=["ae_soh"]
    ).copy()

    if work.empty:
        raise ValueError(
            "No valid SOH absolute-error values were found."
        )

    # ---------------------------------------------------------
    # Bin-wise MedAE
    # ---------------------------------------------------------
    bin_stats = (
        work.groupby(
            "soh_bin",
            observed=True,
        )["ae_soh"]
        .median()
        .reset_index()
    )

    bin_stats = bin_stats.rename(
        columns={
            "ae_soh": "medae",
        }
    )

    bin_stats["start_val"] = (
        bin_stats["soh_bin"]
        .apply(get_bin_start)
    )

    bin_stats = (
        bin_stats
        .sort_values(
            "start_val",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    return bin_stats


# =============================================================================
# Plot: complete
# =============================================================================

def plot_soh_bin_medae(
    bin_stats: pd.DataFrame,
    save_path: str,
    dpi: int = 300,
) -> None:
    """
    Plot the full reference-style SOH-bin MedAE bar chart.

    Visual settings are intentionally kept the same as the original code.
    """
    plt.rcParams[
        "font.family"
    ] = "Arial"

    x_labels = (
        bin_stats["soh_bin"]
        .astype(str)
    )

    y_values = (
        bin_stats["medae"]
        .to_numpy(dtype=float)
    )

    x_pos = np.arange(
        len(x_labels)
    )

    soh_solid_pink = "#675265"
    axis_color = "#2F3E46"

    y_max = float(
        np.nanmax(y_values)
    )

    y_top = (
        y_max * 1.25
        if y_max > 0
        else 1.0
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        x_pos,
        y_values,
        width=0.7,
        color=soh_solid_pink,
        edgecolor="none",
        alpha=1.0,
        zorder=2,
    )

    ax.set_ylim(
        0,
        y_top,
    )

    ax.set_xticks(
        x_pos
    )

    ax.set_xticklabels(
        x_labels,
        rotation=45,
        ha="right",
        fontsize=10,
    )

    ax.set_ylabel(
        "SOH MedAE (%)",
        fontsize=12,
        fontweight="bold",
        labelpad=10,
    )

    ax.set_xlabel(
        "SOH Range (%)",
        fontsize=12,
        fontweight="bold",
        labelpad=10,
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.spines[
        "left"
    ].set_color(
        axis_color
    )

    ax.spines[
        "bottom"
    ].set_color(
        axis_color
    )

    ax.yaxis.grid(
        True,
        linestyle="--",
        alpha=0.2,
        zorder=0,
    )

    for bar in bars:
        height = (
            bar.get_height()
        )

        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2.0,
            height
            +
            (y_max * 0.02),
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=axis_color,
            fontweight="bold",
        )

    os.makedirs(
        os.path.dirname(
            save_path
        ),
        exist_ok=True,
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=dpi,
        transparent=False,
    )

    plt.close(
        fig
    )


# =============================================================================
# Plot: pure
# =============================================================================

def plot_soh_bin_medae_pure(
    bin_stats: pd.DataFrame,
    save_path: str,
    dpi: int = 600,
) -> None:
    """
    Plot the pure slim SOH-bin MedAE figure.

    Pure style follows the same clean settings as the SOC-bin figure:
    - figsize=(2.38, 0.55)
    - bar width=0.8
    - alpha=0.8
    - no axes
    - no ticks
    - no labels
    - no annotation text
    - no grid
    - transparent background
    - bars only
    """
    y_values = (
        bin_stats["medae"]
        .to_numpy(dtype=float)
    )

    x_pos = np.arange(
        len(y_values)
    )

    soh_solid_pink = "#675265"

    y_max = float(
        np.nanmax(y_values)
    )

    y_top = (
        y_max * 1.25
        if y_max > 0
        else 1.0
    )

    fig_clean, ax_clean = plt.subplots(
        figsize=(2.38, 0.55)
    )

    ax_clean.bar(
        x_pos,
        y_values,
        width=0.8,
        color=soh_solid_pink,
        edgecolor="none",
        alpha=0.8,
        zorder=1,
    )

    ax_clean.set_ylim(
        0,
        y_top,
    )

    ax_clean.set_xlim(
        -0.5,
        len(x_pos) - 0.5,
    )

    ax_clean.set_axis_off()

    fig_clean.patch.set_alpha(
        0
    )

    ax_clean.patch.set_alpha(
        0
    )

    os.makedirs(
        os.path.dirname(
            save_path
        ),
        exist_ok=True,
    )

    plt.savefig(
        save_path,
        dpi=dpi,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close(
        fig_clean
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    save_path = os.path.join(
        args.output_dir,
        args.output_name,
    )

    pure_save_path = os.path.join(
        args.output_dir,
        args.pure_output_name,
    )

    df = read_csv_robust(
        args.input_csv
    )

    bin_stats = compute_soh_bin_medae(
        df
    )

    plot_soh_bin_medae(
        bin_stats=bin_stats,
        save_path=save_path,
        dpi=args.dpi,
    )

    plot_soh_bin_medae_pure(
        bin_stats=bin_stats,
        save_path=pure_save_path,
        dpi=args.pure_dpi,
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "[SOH BIN-WISE MEDAE]"
    )

    print(
        "=" * 80
    )

    print(
        bin_stats[
            [
                "soh_bin",
                "medae",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print(
        f"\n[OK] Saved Figure 3d SOH-bin MedAE plot: "
        f"{save_path}"
    )

    print(
        f"[OK] Saved Figure 3d SOH-bin MedAE pure plot: "
        f"{pure_save_path}"
    )


if __name__ == "__main__":
    main()
