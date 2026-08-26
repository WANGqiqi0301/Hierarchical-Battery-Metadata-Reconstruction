# -*- coding: utf-8 -*-
"""
suppfig6_count_companion_heatmaps.py

Supplementary Figure 6:
Confusion-count and bin-wise MedAE heatmaps.

This script reads the proposed_framework per-sample test prediction file and generates:
1. Material-capacity confusion count heatmap
2. SOC-bin SOC MedAE heatmap
3. SOH-bin SOH MedAE heatmap

For the SOC/SOH MedAE heatmaps, cells with fewer than LOW_COUNT_THRESHOLD
samples are hatched to indicate limited support.

Expected input columns:
    ID, true_label, pred_label,
    soc_true, soc_pred,
    soh_true, soh_pred

Input:
    results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv

Output:
    results/figures/supp/suppfig6

Generated files:
    suppfig6_count_companion_heatmaps.png
    suppfig6_count_companion_heatmaps.pdf
    suppfig6_material_confusion_count_heatmap_pure.png
    suppfig6_material_confusion_count_heatmap_pure.pdf
    suppfig6_soc_bin_medae_heatmap_pure.png
    suppfig6_soc_bin_medae_heatmap_pure.pdf
    suppfig6_soh_bin_medae_heatmap_pure.png
    suppfig6_soh_bin_medae_heatmap_pure.pdf
    suppfig6_material_confusion_count_matrix.csv
    suppfig6_soc_bin_medae_matrix.csv
    suppfig6_soh_bin_medae_matrix.csv
    suppfig6_soc_bin_sample_count_matrix.csv
    suppfig6_soh_bin_sample_count_matrix.csv
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "proposed_framework"
    / "further_analysis"
    / "tables"
    / "test_predictions_per_sample.csv"
)

OUT_DIR = PROJECT_ROOT / "results" / "figures" / "supp" / "suppfig6"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Global style
# ============================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})


# ============================================================
# Configuration
# ============================================================
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

N_BINS = 10

LOW_COUNT_THRESHOLD = 5

VMIN = 0
CMAP_CONF = "Blues"
CMAP_SOC = "PuBu"
CMAP_SOH = "PuBu"

LINE_WIDTH = 2.8
LINE_COLOR = "#66666630"

MISSING_FACE = "#FCFCFC"
MISSING_HATCH = "///"

LOWCOUNT_HATCH = "////"
LOWCOUNT_EDGE = "#6F6F6F55"


# ============================================================
# Data helpers
# ============================================================
def check_required_columns(df: pd.DataFrame):
    required_cols = [
        "ID",
        "true_label",
        "pred_label",
        "soc_true",
        "soc_pred",
        "soh_true",
        "soh_pred",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def add_bins_and_errors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["soc_true", "soc_pred", "soh_true", "soh_pred"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Match the original binning logic:
    # pd.cut(values, bins=10) on the original physical-scale values.
    df["soc_bin"] = pd.cut(df["soc_true"], bins=N_BINS)
    df["soh_bin"] = pd.cut(df["soh_true"], bins=N_BINS)

    # Absolute errors used for MedAE.
    df["soc_ae"] = np.abs(df["soc_pred"] - df["soc_true"])
    df["soh_ae"] = np.abs(df["soh_pred"] - df["soh_true"])

    return df


# ============================================================
# Matrix helpers
# ============================================================
def get_bin_start(bin_value) -> float:
    if pd.isna(bin_value):
        return np.nan

    if hasattr(bin_value, "left"):
        return float(bin_value.left)

    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", str(bin_value))
    if len(nums) >= 1:
        return float(nums[0])

    return np.nan


def sort_bins_desc(bin_list):
    return sorted(bin_list, key=get_bin_start, reverse=True)


def round_up_nice(x: float) -> float:
    if not np.isfinite(x) or x <= 0:
        return 1.0

    if x <= 2:
        return math.ceil(x / 0.25) * 0.25
    if x <= 5:
        return math.ceil(x / 0.5) * 0.5
    if x <= 10:
        return math.ceil(x)
    if x <= 20:
        return math.ceil(x / 2.0) * 2.0
    if x <= 50:
        return math.ceil(x / 5.0) * 5.0
    if x <= 100:
        return math.ceil(x / 10.0) * 10.0

    return math.ceil(x / 20.0) * 20.0


def safe_nanmax(arr) -> float:
    arr = np.asarray(arr, dtype=float)

    if np.all(~np.isfinite(arr)):
        return 0.0

    return float(np.nanmax(arr))


def build_confusion_count_pivot(df: pd.DataFrame) -> pd.DataFrame:
    pivot = pd.crosstab(df["true_label"], df["pred_label"])
    pivot = pivot.reindex(index=MATERIAL_ORDER, columns=MATERIAL_ORDER)
    pivot = pivot.fillna(0)

    return pivot


def build_bin_medae_pivot(
    df: pd.DataFrame,
    bin_col: str,
    error_col: str,
) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=bin_col,
        columns="true_label",
        values=error_col,
        aggfunc="median",
        observed=False,
    )

    valid_cols = [c for c in MATERIAL_ORDER if c in pivot.columns]
    pivot = pivot.reindex(columns=valid_cols)

    sorted_index = sort_bins_desc(pivot.index.tolist())
    pivot = pivot.reindex(sorted_index)

    return pivot


def build_bin_count_pivot(
    df: pd.DataFrame,
    bin_col: str,
) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=bin_col,
        columns="true_label",
        values="ID",
        aggfunc="count",
        observed=False,
    )

    valid_cols = [c for c in MATERIAL_ORDER if c in pivot.columns]
    pivot = pivot.reindex(columns=valid_cols)

    sorted_index = sort_bins_desc(pivot.index.tolist())
    pivot = pivot.reindex(sorted_index)

    return pivot


# ============================================================
# Plot helpers
# ============================================================
def add_missing_background(ax, pivot: pd.DataFrame):
    nrows, ncols = pivot.shape

    for i in range(nrows):
        for j in range(ncols):
            if pd.isna(pivot.iloc[i, j]):
                rect = mpatches.Rectangle(
                    (j, i),
                    1,
                    1,
                    fill=True,
                    facecolor=MISSING_FACE,
                    hatch=MISSING_HATCH,
                    edgecolor=LINE_COLOR,
                    linewidth=0.8,
                    alpha=0.25,
                )
                ax.add_patch(rect)


def add_lowcount_hatch(
    ax,
    count_pivot: pd.DataFrame | None,
    threshold: int = LOW_COUNT_THRESHOLD,
):
    if count_pivot is None:
        return

    nrows, ncols = count_pivot.shape

    for i in range(nrows):
        for j in range(ncols):
            value = count_pivot.iloc[i, j]

            if pd.notna(value) and value > 0 and value < threshold:
                rect = mpatches.Rectangle(
                    (j, i),
                    1,
                    1,
                    fill=False,
                    hatch=LOWCOUNT_HATCH,
                    edgecolor=LOWCOUNT_EDGE,
                    linewidth=0.0,
                )
                ax.add_patch(rect)


def draw_pure_heatmap_on_axis(
    ax,
    pivot: pd.DataFrame,
    cmap_name: str,
    vmin: float,
    vmax: float,
    use_missing_background: bool,
    count_pivot: pd.DataFrame | None = None,
):
    mask = pivot.isnull()

    if use_missing_background:
        add_missing_background(ax, pivot)

    sns.heatmap(
        pivot,
        annot=False,
        cmap=plt.get_cmap(cmap_name),
        mask=mask,
        vmin=vmin,
        vmax=vmax,
        linewidths=LINE_WIDTH,
        linecolor=LINE_COLOR,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )

    add_lowcount_hatch(ax, count_pivot, LOW_COUNT_THRESHOLD)

    ax.set_ylim(len(pivot), 0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.set_axis_off()


def save_single_pure_heatmap(
    pivot: pd.DataFrame,
    cmap_name: str,
    vmin: float,
    vmax: float,
    use_missing_background: bool,
    save_png: Path,
    save_pdf: Path,
    figsize: tuple[float, float],
    count_pivot: pd.DataFrame | None = None,
):
    fig, ax = plt.subplots(figsize=figsize)

    draw_pure_heatmap_on_axis(
        ax=ax,
        pivot=pivot,
        cmap_name=cmap_name,
        vmin=vmin,
        vmax=vmax,
        use_missing_background=use_missing_background,
        count_pivot=count_pivot,
    )

    fig.patch.set_alpha(0.0)

    fig.savefig(
        save_png,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.0,
        transparent=True,
    )

    fig.savefig(
        save_pdf,
        bbox_inches="tight",
        pad_inches=0.0,
        transparent=True,
    )

    plt.close(fig)


def draw_heatmap_on_axis(
    ax,
    pivot: pd.DataFrame,
    cmap_name: str,
    vmin: float,
    vmax: float,
    title: str,
    use_missing_background: bool,
    cbar_ax=None,
    cbar_label: str = "",
    value_fmt: str = ".2f",
    count_pivot: pd.DataFrame | None = None,
):
    mask = pivot.isnull()

    if use_missing_background:
        add_missing_background(ax, pivot)

    sns.heatmap(
        pivot,
        annot=True,
        fmt=value_fmt,
        cmap=plt.get_cmap(cmap_name),
        mask=mask,
        vmin=vmin,
        vmax=vmax,
        linewidths=LINE_WIDTH,
        linecolor=LINE_COLOR,
        cbar=True,
        cbar_ax=cbar_ax,
        cbar_kws={"label": cbar_label, "ticks": [vmin, vmax]},
        xticklabels=True,
        yticklabels=True,
        ax=ax,
    )

    add_lowcount_hatch(ax, count_pivot, LOW_COUNT_THRESHOLD)

    ax.set_ylim(len(pivot), 0)
    ax.set_title(title, pad=10, fontsize=13)

    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    if cbar_ax is not None:
        cbar_ax.tick_params(labelsize=8, length=2)
        cbar_ax.set_ylabel(cbar_label, fontsize=9, rotation=90, labelpad=8)


def draw_combined_heatmaps(
    pivot_conf: pd.DataFrame,
    pivot_soc_medae: pd.DataFrame,
    pivot_soh_medae: pd.DataFrame,
    pivot_soc_count: pd.DataFrame,
    pivot_soh_count: pd.DataFrame,
    save_png: Path,
    save_pdf: Path,
    pure_conf_png: Path,
    pure_conf_pdf: Path,
    pure_soc_png: Path,
    pure_soc_pdf: Path,
    pure_soh_png: Path,
    pure_soh_pdf: Path,
    vmax_conf: float,
    vmax_soc: float,
    vmax_soh: float,
):
    fig = plt.figure(figsize=(7.6, 11.0))

    grid_spec = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1.0, 0.055],
        height_ratios=[1.0, 1.2, 1.2],
        hspace=0.18,
        wspace=0.05,
    )

    ax_conf = fig.add_subplot(grid_spec[0, 0])
    ax_soc = fig.add_subplot(grid_spec[1, 0])
    ax_soh = fig.add_subplot(grid_spec[2, 0])

    cax_conf = fig.add_subplot(grid_spec[0, 1])
    cax_soc = fig.add_subplot(grid_spec[1, 1])
    cax_soh = fig.add_subplot(grid_spec[2, 1])

    draw_heatmap_on_axis(
        ax=ax_conf,
        pivot=pivot_conf,
        cmap_name=CMAP_CONF,
        vmin=VMIN,
        vmax=vmax_conf,
        title="Material-capacity confusion count",
        use_missing_background=False,
        cbar_ax=cax_conf,
        cbar_label="Count",
        value_fmt=".0f",
        count_pivot=None,
    )

    draw_heatmap_on_axis(
        ax=ax_soc,
        pivot=pivot_soc_medae,
        cmap_name=CMAP_SOC,
        vmin=VMIN,
        vmax=vmax_soc,
        title=f"SOC-bin MedAE (hatched if n < {LOW_COUNT_THRESHOLD})",
        use_missing_background=True,
        cbar_ax=cax_soc,
        cbar_label="SOC MedAE (%)",
        value_fmt=".2f",
        count_pivot=pivot_soc_count,
    )

    draw_heatmap_on_axis(
        ax=ax_soh,
        pivot=pivot_soh_medae,
        cmap_name=CMAP_SOH,
        vmin=VMIN,
        vmax=vmax_soh,
        title=f"SOH-bin MedAE (hatched if n < {LOW_COUNT_THRESHOLD})",
        use_missing_background=True,
        cbar_ax=cax_soh,
        cbar_label="SOH MedAE (%)",
        value_fmt=".2f",
        count_pivot=pivot_soh_count,
    )

    fig.savefig(
        save_png,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    fig.savefig(
        save_pdf,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(fig)

    save_single_pure_heatmap(
        pivot=pivot_conf,
        cmap_name=CMAP_CONF,
        vmin=VMIN,
        vmax=vmax_conf,
        use_missing_background=False,
        save_png=pure_conf_png,
        save_pdf=pure_conf_pdf,
        figsize=(6.2, 5.4),
        count_pivot=None,
    )

    save_single_pure_heatmap(
        pivot=pivot_soc_medae,
        cmap_name=CMAP_SOC,
        vmin=VMIN,
        vmax=vmax_soc,
        use_missing_background=True,
        save_png=pure_soc_png,
        save_pdf=pure_soc_pdf,
        figsize=(6.8, 6.0),
        count_pivot=pivot_soc_count,
    )

    save_single_pure_heatmap(
        pivot=pivot_soh_medae,
        cmap_name=CMAP_SOH,
        vmin=VMIN,
        vmax=vmax_soh,
        use_missing_background=True,
        save_png=pure_soh_png,
        save_pdf=pure_soh_pdf,
        figsize=(6.8, 6.0),
        count_pivot=pivot_soh_count,
    )


# ============================================================
# Main
# ============================================================
def main():
    print("[INFO] Generating Supplementary Figure 6...")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)
    check_required_columns(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = add_bins_and_errors(df)

    pivot_conf = build_confusion_count_pivot(df)

    pivot_soc_medae = build_bin_medae_pivot(
        df,
        bin_col="soc_bin",
        error_col="soc_ae",
    )

    pivot_soh_medae = build_bin_medae_pivot(
        df,
        bin_col="soh_bin",
        error_col="soh_ae",
    )

    # Count matrices are retained only for low-sample hatching
    # and for supplementary inspection.
    pivot_soc_count = build_bin_count_pivot(
        df,
        bin_col="soc_bin",
    )

    pivot_soh_count = build_bin_count_pivot(
        df,
        bin_col="soh_bin",
    )

    vmax_conf = max(
        round_up_nice(safe_nanmax(pivot_conf.values)),
        1,
    )

    vmax_soc = max(
        round_up_nice(safe_nanmax(pivot_soc_medae.values)),
        0.5,
    )

    vmax_soh = max(
        round_up_nice(safe_nanmax(pivot_soh_medae.values)),
        0.5,
    )

    # --------------------------------------------------------
    # Save matrices
    # --------------------------------------------------------
    conf_csv = (
        OUT_DIR
        / "suppfig6_material_confusion_count_matrix.csv"
    )

    soc_medae_csv = (
        OUT_DIR
        / "suppfig6_soc_bin_medae_matrix.csv"
    )

    soh_medae_csv = (
        OUT_DIR
        / "suppfig6_soh_bin_medae_matrix.csv"
    )

    soc_count_csv = (
        OUT_DIR
        / "suppfig6_soc_bin_sample_count_matrix.csv"
    )

    soh_count_csv = (
        OUT_DIR
        / "suppfig6_soh_bin_sample_count_matrix.csv"
    )

    pivot_conf.to_csv(
        conf_csv,
        encoding="utf-8-sig",
    )

    pivot_soc_medae.to_csv(
        soc_medae_csv,
        encoding="utf-8-sig",
    )

    pivot_soh_medae.to_csv(
        soh_medae_csv,
        encoding="utf-8-sig",
    )

    pivot_soc_count.to_csv(
        soc_count_csv,
        encoding="utf-8-sig",
    )

    pivot_soh_count.to_csv(
        soh_count_csv,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------
    png_path = (
        OUT_DIR
        / "suppfig6_count_companion_heatmaps.png"
    )

    pdf_path = (
        OUT_DIR
        / "suppfig6_count_companion_heatmaps.pdf"
    )

    pure_conf_png = (
        OUT_DIR
        / "suppfig6_material_confusion_count_heatmap_pure.png"
    )

    pure_conf_pdf = (
        OUT_DIR
        / "suppfig6_material_confusion_count_heatmap_pure.pdf"
    )

    pure_soc_png = (
        OUT_DIR
        / "suppfig6_soc_bin_medae_heatmap_pure.png"
    )

    pure_soc_pdf = (
        OUT_DIR
        / "suppfig6_soc_bin_medae_heatmap_pure.pdf"
    )

    pure_soh_png = (
        OUT_DIR
        / "suppfig6_soh_bin_medae_heatmap_pure.png"
    )

    pure_soh_pdf = (
        OUT_DIR
        / "suppfig6_soh_bin_medae_heatmap_pure.pdf"
    )

    # --------------------------------------------------------
    # Draw figures
    # --------------------------------------------------------
    draw_combined_heatmaps(
        pivot_conf=pivot_conf,
        pivot_soc_medae=pivot_soc_medae,
        pivot_soh_medae=pivot_soh_medae,
        pivot_soc_count=pivot_soc_count,
        pivot_soh_count=pivot_soh_count,
        save_png=png_path,
        save_pdf=pdf_path,
        pure_conf_png=pure_conf_png,
        pure_conf_pdf=pure_conf_pdf,
        pure_soc_png=pure_soc_png,
        pure_soc_pdf=pure_soc_pdf,
        pure_soh_png=pure_soh_png,
        pure_soh_pdf=pure_soh_pdf,
        vmax_conf=vmax_conf,
        vmax_soc=vmax_soc,
        vmax_soh=vmax_soh,
    )

    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------
    print("\n" + "=" * 90)
    print("[MEDAE HEATMAP SUMMARY]")
    print("=" * 90)

    print(
        f"SOC MedAE range: "
        f"{np.nanmin(pivot_soc_medae.values):.3f}% - "
        f"{np.nanmax(pivot_soc_medae.values):.3f}%"
    )

    print(
        f"SOH MedAE range: "
        f"{np.nanmin(pivot_soh_medae.values):.3f}% - "
        f"{np.nanmax(pivot_soh_medae.values):.3f}%"
    )

    print("\n[DONE] Supplementary Figure 6 generated.")
    print(f"[SAVED] Figure PNG: {png_path}")
    print(f"[SAVED] Figure PDF: {pdf_path}")
    print(f"[SAVED] Pure confusion PNG: {pure_conf_png}")
    print(f"[SAVED] Pure confusion PDF: {pure_conf_pdf}")
    print(f"[SAVED] Pure SOC MedAE PNG: {pure_soc_png}")
    print(f"[SAVED] Pure SOC MedAE PDF: {pure_soc_pdf}")
    print(f"[SAVED] Pure SOH MedAE PNG: {pure_soh_png}")
    print(f"[SAVED] Pure SOH MedAE PDF: {pure_soh_pdf}")
    print(f"[SAVED] Confusion matrix: {conf_csv}")
    print(f"[SAVED] SOC MedAE matrix: {soc_medae_csv}")
    print(f"[SAVED] SOH MedAE matrix: {soh_medae_csv}")
    print(f"[SAVED] SOC count matrix: {soc_count_csv}")
    print(f"[SAVED] SOH count matrix: {soh_count_csv}")


if __name__ == "__main__":
    main()
