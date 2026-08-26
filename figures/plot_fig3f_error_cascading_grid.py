# -*- coding: utf-8 -*-
"""
plot_fig3f_error_cascading_grid.py

Figure 3f:
Cascading-grid KDE plots of SOC absolute error vs SOH absolute error
for each material-capacity group.

IMPORTANT:
This script generates Figure 3f using the current prediction table:
    zzz6_plot_nc_figures_CASCADING_GRID_V2.py

It keeps the old plotting/data logic exactly:
1) Read old i10-style CSV:
       soc_true, soc_pred, soc_sigma, soh_true, soh_pred, soh_sigma, label
2) If true_label does not exist, use label as true_label
3) Convert true_label to int
4) Apply visualization-only LMO_24Ah and NMC_15Ah filtering
5) Use sorted(df["true_label"].unique()) for material order
6) Use global 1%-99% quantile axis limits
7) Use seaborn whitegrid style, KDE levels=6, linewidth=2.5
8) Save main grid, single panels, clean single panels, and summary CSV

Default input:
    results/proposed_framework/further_analysis/tables/test_predictions_with_uncertainty.csv

Default output:
    draw_figures/figure3_main/cascading_grid_lmo24_filtered
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Default configuration
# =============================================================================
DEFAULT_INPUT_CSV = (
    r"results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv"
)

# Use the old output folder by default to match the old script.
# If you prefer the new organized output folder, change this to:
# r"results/figures/main/fig3f"
DEFAULT_OUTPUT_DIR = (
    r"results/figures/main/fig3f"
)

MATERIAL_NAME_MAP: Dict[int, str] = {
    0: "LFP_35Ah",
    1: "LFP_68Ah",
    2: "LMO_10Ah",
    3: "LMO_24Ah",
    4: "LMO_25Ah",
    5: "LMO_26Ah",
    6: "NMC_15Ah",
    7: "NMC_21Ah",
}

MATERIAL_LABEL_MAP: Dict[str, int] = {v: k for k, v in MATERIAL_NAME_MAP.items()}

MAX_MATERIALS = 8
MIN_SAMPLES_PER_PANEL = 30
KDE_LEVELS = 6
LINE_WIDTH = 2.5
DPI = 600

# The old script comment said top-8, but the actual code used head(9).
# Keep 9 to reproduce the old figure.
LMO24_OUTLIER_REMOVE_N = 9
NMC15_ABNORMAL_ID = "PJP17601529001A010435"


# =============================================================================
# Argument parser
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 3f cascading error-grid KDE plots."
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default=DEFAULT_INPUT_CSV,
        help="Path to prediction CSV.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save Figure 3f outputs.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI,
        help="Output figure resolution.",
    )

    parser.add_argument(
        "--remove_lmo24_outliers",
        action="store_true",
        default=True,
        help="Remove high-SOC-error LMO_24Ah outliers before plotting.",
    )

    parser.add_argument(
        "--keep_lmo24_outliers",
        action="store_true",
        help="Disable LMO_24Ah outlier removal.",
    )

    parser.add_argument(
        "--lmo24_remove_n",
        type=int,
        default=LMO24_OUTLIER_REMOVE_N,
        help="Number of high-SOC-error LMO_24Ah samples to remove.",
    )

    return parser.parse_args()


# =============================================================================
# Global style
# =============================================================================
def set_plot_style() -> None:
    """
    Set global seaborn style exactly as in the old script.
    """
    sns.set(style="whitegrid", font_scale=1.15)


# =============================================================================
# Data loading and preprocessing
# =============================================================================
def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV exactly like the old script.

    The old script used pd.read_csv(path) directly.
    Here we keep that behavior to avoid hidden differences.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    # print(" Loaded:", df.shape)
    # print("Columns:", df.columns.tolist())
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize material labels and recompute absolute errors directly from
    the current prediction columns.
    """
    df = df.copy()

    if "true_label" not in df.columns:
        if "label" in df.columns:
            df["true_label"] = df["label"]
        else:
            raise ValueError("CSV must contain either true_label or label.")

    if pd.api.types.is_numeric_dtype(df["true_label"]):
        df["true_label"] = df["true_label"].astype(int)
    else:
        label_text = df["true_label"].astype(str)
        mapped = label_text.map(MATERIAL_LABEL_MAP)

        if mapped.isna().any():
            unknown = sorted(label_text[mapped.isna()].unique().tolist())
            raise ValueError(f"Unknown material labels: {unknown}")

        df["true_label"] = mapped.astype(int)

    required_cols = {"soc_true", "soc_pred", "soh_true", "soh_pred"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"Missing required prediction columns: {sorted(missing_cols)}"
        )

    df["soc_abs_err"] = np.abs(
        df["soc_pred"].astype(float) - df["soc_true"].astype(float)
    )
    df["soh_abs_err"] = np.abs(
        df["soh_pred"].astype(float) - df["soh_true"].astype(float)
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=["true_label", "soc_abs_err", "soh_abs_err"]
    ).copy()

    return df


# =============================================================================
# Utility functions
# =============================================================================
def label_to_name(label) -> str:
    """
    Convert integer label to material-capacity name.
    This is the old script's label mapping behavior.
    """
    return MATERIAL_NAME_MAP.get(label, f"Material {label}")


def compute_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient.
    """
    if len(x) < 3:
        return np.nan

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def draw_diagonal_line(
    ax,
    xlim,
    ylim,
    color="gray",
    linestyle="--",
    linewidth=1.0,
    alpha=0.8,
) -> None:
    """
    Draw the visual diagonal reference line across the current display box.

    This is not mathematically y = x, because SOC error and SOH error
    have different scales.
    """
    x0, x1 = xlim
    y0, y1 = ylim

    ax.plot(
        [x0, x1],
        [y0, y1],
        linestyle=linestyle,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=1,
    )


# =============================================================================
# Single-panel plotting
# =============================================================================
def save_single_panel(
    sub: pd.DataFrame,
    label,
    xlim,
    ylim,
    out_png,
    clean_png,
    dpi: int = DPI,
) -> None:
    """
    Save standard and clean versions of a single material panel.
    This follows the old script's single-panel logic.
    """
    mat_name = label_to_name(label)
    n = len(sub)
    r = compute_corr(sub["soc_abs_err"].values, sub["soh_abs_err"].values)

    # ---------- Standard panel ----------
    fig, ax = plt.subplots(figsize=(4.2, 4.0))

    sns.kdeplot(
        x=sub["soc_abs_err"],
        y=sub["soh_abs_err"],
        levels=KDE_LEVELS,
        linewidths=LINE_WIDTH,
        ax=ax,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    draw_diagonal_line(ax, xlim, ylim)

    ax.set_title(f"{mat_name} (n={n})")
    ax.set_xlabel("SOC Absolute Error")
    ax.set_ylabel("SOH Absolute Error")

    r_text = "r = NA" if np.isnan(r) else f"r = {r:.2f}"

    ax.text(
        0.97,
        0.95,
        r_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            alpha=0.75,
            edgecolor="none",
        ),
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # ---------- Clean panel ----------
    fig, ax = plt.subplots(figsize=(4.2, 4.0))

    sns.kdeplot(
        x=sub["soc_abs_err"],
        y=sub["soh_abs_err"],
        levels=KDE_LEVELS,
        linewidths=4,
        ax=ax,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0)
    fig.savefig(clean_png, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# =============================================================================
# Main cascading-grid plotting
# =============================================================================
def plot_cascading_grid(
    df: pd.DataFrame,
    save_root: str,
    dpi: int = DPI,
) -> None:
    """
    Plot the Figure 3f cascading grid and save panel versions.

    This function intentionally follows the old script:
        materials = sorted(df["true_label"].unique())[:MAX_MATERIALS]
    """
    os.makedirs(save_root, exist_ok=True)

    single_dir = os.path.join(save_root, "single_panels")
    clean_dir = os.path.join(save_root, "single_panels_clean")

    os.makedirs(single_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    # Old material order logic
    materials = sorted(df["true_label"].unique())[:MAX_MATERIALS]

    # Global 1%-99% quantile limits, exactly as in the old script
    x_min, x_max = df["soc_abs_err"].quantile([0.01, 0.99])
    y_min, y_max = df["soh_abs_err"].quantile([0.01, 0.99])

    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    xlim = (max(0, x_min - x_pad), x_max + x_pad)
    ylim = (max(0, y_min - y_pad), y_max + y_pad)

    # print("[AXIS] xlim =", xlim)
    # print("[AXIS] ylim =", ylim)
    # print("[MATERIALS]", materials)

    fig, axes = plt.subplots(1, 8, figsize=(20, 5), sharex=True, sharey=True)
    axes = axes.flatten()

    summary_rows = []

    for i, label in enumerate(materials):
        ax = axes[i]
        sub = df[df["true_label"] == label].copy()
        n = len(sub)
        mat_name = label_to_name(label)

        if n < MIN_SAMPLES_PER_PANEL:
            ax.set_title(f"{mat_name} (n={n}, skipped)")
            ax.axis("off")

            summary_rows.append({
                "label": label,
                "material_name": mat_name,
                "n": n,
                "pearson_r": np.nan,
                "status": "skipped_too_few_samples",
            })
            continue

        sns.kdeplot(
            x=sub["soc_abs_err"],
            y=sub["soh_abs_err"],
            levels=KDE_LEVELS,
            linewidths=LINE_WIDTH,
            ax=ax,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        draw_diagonal_line(ax, xlim, ylim)

        r = compute_corr(sub["soc_abs_err"].values, sub["soh_abs_err"].values)
        r_text = "r = NA" if np.isnan(r) else f"r = {r:.2f}"

        ax.set_title(f"{mat_name} (n={n})", fontsize=11)

        ax.text(
            0.97,
            0.95,
            r_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                alpha=0.75,
                edgecolor="none",
            ),
        )

        # Old filename logic
        stem = (
            f"{i + 1:02d}_"
            f"{str(label).replace('/', '_')}_"
            f"{mat_name.replace(' ', '_').replace('/', '_')}"
        )

        out_png = os.path.join(single_dir, f"{stem}.png")
        clean_png = os.path.join(clean_dir, f"{stem}_clean.png")

        save_single_panel(
            sub=sub,
            label=label,
            xlim=xlim,
            ylim=ylim,
            out_png=out_png,
            clean_png=clean_png,
            dpi=dpi,
        )

        summary_rows.append({
            "label": label,
            "material_name": mat_name,
            "n": n,
            "pearson_r": r,
            "status": "saved",
        })

    # Remove unused subplots
    for j in range(len(materials), len(axes)):
        fig.delaxes(axes[j])

    fig.supxlabel("SOC Absolute Error", fontsize=14)
    fig.supylabel("SOH Absolute Error", fontsize=14)
    fig.tight_layout()

    main_png = os.path.join(save_root, "cascading_grid_with_labels_r_diagonal.png")
    fig.savefig(main_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(save_root, "cascading_grid_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")



# =============================================================================
# LMO_24Ah outlier filtering
# =============================================================================
def filter_lmo24_top_soc_error_outliers(
    df: pd.DataFrame,
    remove_n: int = LMO24_OUTLIER_REMOVE_N,
) -> pd.DataFrame:
    """
    Remove the LMO_24Ah rows with the largest SOC absolute errors
    for Figure 3f visualization-level correlation analysis only.
    """
    df = df.copy()
    lmo24_label = MATERIAL_LABEL_MAP["LMO_24Ah"]
    sub = df[df["true_label"] == lmo24_label].copy()

    if sub.empty:
        return df

    r_before = compute_corr(
        sub["soc_abs_err"].values,
        sub["soh_abs_err"].values,
    )

    remove_idx = (
        sub.sort_values("soc_abs_err", ascending=False)
        .head(remove_n)
        .index
    )

    df = df.drop(index=remove_idx).copy()

    sub_after = df[df["true_label"] == lmo24_label].copy()
    r_after = compute_corr(
        sub_after["soc_abs_err"].values,
        sub_after["soh_abs_err"].values,
    )


    return df


def filter_nmc15_abnormal_id(
    df: pd.DataFrame,
    abnormal_id: str = NMC15_ABNORMAL_ID,
) -> pd.DataFrame:
    """
    Exclude one identified NMC_15Ah cell-state showing systematic
    pulse-width-dependent SOC underestimation from Figure 3f only.
    """
    df = df.copy()

    if "ID" not in df.columns:
        raise ValueError(
            "CSV must contain ID for NMC_15Ah ID-level filtering."
        )

    nmc15_label = MATERIAL_LABEL_MAP["NMC_15Ah"]
    sub = df[df["true_label"] == nmc15_label].copy()

    if sub.empty:
        return df

    target_mask = (
        (df["true_label"] == nmc15_label)
        & (df["ID"].astype(str) == str(abnormal_id))
    )

    removed_n = int(target_mask.sum())

    if removed_n == 0:
        return df

    r_before = compute_corr(
        sub["soc_abs_err"].values,
        sub["soh_abs_err"].values,
    )

    df = df.loc[~target_mask].copy()

    sub_after = df[df["true_label"] == nmc15_label].copy()
    r_after = compute_corr(
        sub_after["soc_abs_err"].values,
        sub_after["soh_abs_err"].values,
    )


    return df


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    set_plot_style()

    df = load_data(args.input_csv)
    df = preprocess(df)

    if args.keep_lmo24_outliers:
        pass
    elif args.remove_lmo24_outliers:
        df = filter_lmo24_top_soc_error_outliers(
            df,
            remove_n=args.lmo24_remove_n,
        )

    df = filter_nmc15_abnormal_id(df)

    plot_cascading_grid(
        df=df,
        save_root=args.output_dir,
        dpi=args.dpi,
    )



if __name__ == "__main__":
    main()
