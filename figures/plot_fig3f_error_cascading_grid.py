# -*- coding: utf-8 -*-
"""
plot_fig3f_error_cascading_grid.py

Figure 3f:
Cascading-grid KDE plots of SOC absolute error vs SOH absolute error
for each material-capacity group.

This script is the organized version of:
    zzz6_plot_nc_figures_CASCADING_GRID_V2.py

It keeps the original visual settings:
- seaborn whitegrid style
- KDE contour levels = 6
- KDE linewidth = 2.5
- global 1%-99% quantile axis limits
- diagonal reference line
- Pearson r annotation
- material title and sample number
- main 1x8 grid
- single-panel versions
- clean single-panel versions
- summary CSV

Default input:
    results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv

Default output:
    results/figures/main/fig3f
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Tuple

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
DEFAULT_OUTPUT_DIR = r"results/figures/main/fig3f"

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

MATERIAL_ORDER = [
    "LFP_35Ah",
    "LFP_68Ah",
    "LMO_10Ah",
    "LMO_24Ah",
    "LMO_25Ah",
    "LMO_26Ah",
    "NMC_15Ah",
    "NMC_21Ah",
]

MAX_MATERIALS = 8
MIN_SAMPLES_PER_PANEL = 30
KDE_LEVELS = 6
LINE_WIDTH = 2.5
DPI = 600

# Kept consistent with the old script: although the comment said top-8,
# the old code used head(9). Keep 9 for reproducing the old figure.
LMO24_OUTLIER_REMOVE_N = 9


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
    """Set global seaborn style exactly as in the original script."""
    sns.set(style="whitegrid", font_scale=1.15)


# =============================================================================
# Data loading and preprocessing
# =============================================================================
def read_csv_robust(path: str) -> pd.DataFrame:
    """Read CSV with encoding fallback."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")

    encodings = ["utf-8-sig", "utf-8", "cp1252", "gbk"]

    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[DATA] Loaded: {path}")
            print(f"[DATA] Shape: {df.shape}")
            print(f"[DATA] Columns: {df.columns.tolist()}")
            return df
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"Failed to read CSV with supported encodings. Last error: {last_error}",
    )


def normalize_material_label(value: Any) -> str:
    """
    Convert material label to material name.

    Supports both old integer labels:
        0, 1, ..., 7

    and new string labels:
        LFP_35Ah, LMO_24Ah, etc.
    """
    if pd.isna(value):
        return "Unknown"

    # Old integer-like labels.
    try:
        if isinstance(value, str):
            value_strip = value.strip()
            if value_strip.isdigit():
                return MATERIAL_NAME_MAP.get(int(value_strip), f"Material {value_strip}")

        if isinstance(value, (int, np.integer)):
            return MATERIAL_NAME_MAP.get(int(value), f"Material {value}")

        if isinstance(value, float) and float(value).is_integer():
            return MATERIAL_NAME_MAP.get(int(value), f"Material {int(value)}")
    except Exception:
        pass

    # New string labels.
    return str(value)


def material_sort_key(material_name: str) -> int:
    """Sort material names according to the original 8-class order."""
    if material_name in MATERIAL_ORDER:
        return MATERIAL_ORDER.index(material_name)
    return 999


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare required columns.

    Required after preprocessing:
        material_name
        soc_abs_err
        soh_abs_err
    """
    df = df.copy()

    # Label column compatibility.
    if "true_label" not in df.columns:
        if "label" in df.columns:
            df["true_label"] = df["label"]
        else:
            raise ValueError("CSV must contain either 'true_label' or 'label' column.")

    df["material_name"] = df["true_label"].apply(normalize_material_label)

    # SOC absolute error.
    if "soc_abs_err" not in df.columns:
        if "soc_pred" in df.columns and "soc_true" in df.columns:
            df["soc_abs_err"] = np.abs(df["soc_pred"] - df["soc_true"])
        else:
            raise ValueError(
                "CSV does not contain 'soc_abs_err', and it cannot be computed "
                "because 'soc_pred'/'soc_true' are missing."
            )

    # SOH absolute error.
    if "soh_abs_err" not in df.columns:
        if "soh_pred" in df.columns and "soh_true" in df.columns:
            df["soh_abs_err"] = np.abs(df["soh_pred"] - df["soh_true"])
        else:
            raise ValueError(
                "CSV does not contain 'soh_abs_err', and it cannot be computed "
                "because 'soh_pred'/'soh_true' are missing."
            )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["material_name", "soc_abs_err", "soh_abs_err"]).copy()

    return df


# =============================================================================
# Utility functions
# =============================================================================
def compute_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) < 3:
        return np.nan

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def draw_diagonal_line(
    ax,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    color: str = "gray",
    linestyle: str = "--",
    linewidth: float = 1.0,
    alpha: float = 0.8,
) -> None:
    """
    Draw a visual diagonal reference line across the current display box.

    This follows the original code: it is not the mathematical y=x line,
    because SOC error and SOH error have different scales.
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


def safe_filename(text: str) -> str:
    """Make safe filename stem."""
    text = str(text)
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " "]:
        text = text.replace(ch, "_")
    return text


# =============================================================================
# LMO_24Ah outlier filtering
# =============================================================================
def filter_lmo24_high_soc_error_outliers(
    df: pd.DataFrame,
    remove_n: int = LMO24_OUTLIER_REMOVE_N,
) -> pd.DataFrame:
    """
    Remove LMO_24Ah samples with the largest SOC absolute errors.

    Kept consistent with the original script, which used head(9).
    """
    df = df.copy()

    target_material = "LMO_24Ah"
    sub = df[df["material_name"] == target_material].copy()

    if sub.empty:
        print("[FILTER] No LMO_24Ah samples found. Skipping outlier removal.")
        return df

    r_before = compute_corr(
        sub["soc_abs_err"].values,
        sub["soh_abs_err"].values,
    )

    remove_idx = sub.sort_values("soc_abs_err", ascending=False).head(remove_n).index
    removed = df.loc[remove_idx].copy()

    print(f"\n========== Remove LMO_24Ah top-{remove_n} SOC-error outliers ==========")

    columns_to_show = [
        c for c in [
            "soc_true",
            "soc_pred",
            "soc_abs_err",
            "soh_true",
            "soh_pred",
            "soh_abs_err",
        ]
        if c in removed.columns
    ]

    if columns_to_show:
        print("Removed samples:")
        print(removed[columns_to_show].to_string(index=False))
    else:
        print("Removed sample indices:")
        print(remove_idx.tolist())

    df = df.drop(index=remove_idx).copy()

    sub_after = df[df["material_name"] == target_material].copy()
    r_after = compute_corr(
        sub_after["soc_abs_err"].values,
        sub_after["soh_abs_err"].values,
    )

    print("\nLMO_24Ah filtering summary:")
    print(f"n: {len(sub)} -> {len(sub_after)}")
    print(f"r: {r_before:.4f} -> {r_after:.4f}")
    print("=============================================================\n")

    return df


# =============================================================================
# Single-panel plotting
# =============================================================================
def save_single_panel(
    sub: pd.DataFrame,
    material_name: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    out_png: str,
    clean_png: str,
    dpi: int = DPI,
) -> None:
    """Save standard and clean versions of a single material panel."""
    n = len(sub)
    r = compute_corr(
        sub["soc_abs_err"].values,
        sub["soh_abs_err"].values,
    )

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

    ax.set_title(f"{material_name} (n={n})")
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
    """Plot the Figure 3f cascading grid and save panel versions."""
    os.makedirs(save_root, exist_ok=True)

    single_dir = os.path.join(save_root, "single_panels")
    clean_dir = os.path.join(save_root, "single_panels_clean")

    os.makedirs(single_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)

    # Preserve original material order.
    available_materials = sorted(
        df["material_name"].unique().tolist(),
        key=material_sort_key,
    )
    materials = available_materials[:MAX_MATERIALS]

    # Global 1%-99% quantile limits, as in the old script.
    x_min, x_max = df["soc_abs_err"].quantile([0.01, 0.99])
    y_min, y_max = df["soh_abs_err"].quantile([0.01, 0.99])

    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    xlim = (max(0, x_min - x_pad), x_max + x_pad)
    ylim = (max(0, y_min - y_pad), y_max + y_pad)

    fig, axes = plt.subplots(1, 8, figsize=(20, 5), sharex=True, sharey=True)
    axes = axes.flatten()

    summary_rows = []

    for i, material_name in enumerate(materials):
        ax = axes[i]
        sub = df[df["material_name"] == material_name].copy()
        n = len(sub)

        if n < MIN_SAMPLES_PER_PANEL:
            ax.set_title(f"{material_name} (n={n}, skipped)")
            ax.axis("off")

            summary_rows.append({
                "material_name": material_name,
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

        r = compute_corr(
            sub["soc_abs_err"].values,
            sub["soh_abs_err"].values,
        )

        r_text = "r = NA" if np.isnan(r) else f"r = {r:.2f}"

        ax.set_title(f"{material_name} (n={n})", fontsize=11)
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

        stem = f"{i + 1:02d}_{safe_filename(material_name)}"
        out_png = os.path.join(single_dir, f"{stem}.png")
        clean_png = os.path.join(clean_dir, f"{stem}_clean.png")

        save_single_panel(
            sub=sub,
            material_name=material_name,
            xlim=xlim,
            ylim=ylim,
            out_png=out_png,
            clean_png=clean_png,
            dpi=dpi,
        )

        summary_rows.append({
            "material_name": material_name,
            "n": n,
            "pearson_r": r,
            "status": "saved",
        })

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

    print("[OK] Cascading grid main figure saved:")
    print(f"  {main_png}")
    print("[OK] Single panel versions saved:")
    print(f"  {single_dir}")
    print("[OK] Clean single panel versions saved:")
    print(f"  {clean_dir}")
    print("[OK] Summary CSV saved:")
    print(f"  {summary_path}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    set_plot_style()

    df = read_csv_robust(args.input_csv)
    df = preprocess(df)

    if args.keep_lmo24_outliers:
        print("[FILTER] Keeping LMO_24Ah outliers.")
    elif args.remove_lmo24_outliers:
        df = filter_lmo24_high_soc_error_outliers(
            df,
            remove_n=args.lmo24_remove_n,
        )

    plot_cascading_grid(
        df=df,
        save_root=args.output_dir,
        dpi=args.dpi,
    )

    print("[DONE] Figure 3f generation complete.")


if __name__ == "__main__":
    main()