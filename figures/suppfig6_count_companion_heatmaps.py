# -*- coding: utf-8 -*-
"""
suppfig6_count_companion_heatmaps.py

Supplementary Figure 6:
Count companion heatmaps for group-wise and bin-wise test samples.

This script reads the proposed_framework per-sample test prediction file and generates:
1. Material-capacity confusion count heatmap
2. SOC-bin sample count heatmap
3. SOH-bin sample count heatmap

Expected input columns:
    ID, true_label, pred_label, soc_true, soh_true

Input:
    results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv

Output:
    results/figures/supp/suppfig6

Generated files:
    suppfig6_count_companion_heatmaps.png
    suppfig6_count_companion_heatmaps.pdf
    suppfig6_material_confusion_count_matrix.csv
    suppfig6_soc_bin_material_count_matrix.csv
    suppfig6_soh_bin_material_count_matrix.csv
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib as mpl
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

SOC_BIN_WIDTH = 10.0
SOH_BIN_WIDTH = 5.0
SOC_BIN_RANGE = (0.0, 100.0)
SOH_BIN_RANGE = None

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
        "soh_true",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def normalize_to_percent(values: pd.Series | np.ndarray) -> np.ndarray:
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

    finite_values = values[np.isfinite(values)]

    if len(finite_values) > 0 and np.nanmax(finite_values) <= 1.5:
        values = values * 100.0

    return values


def make_edges_from_range(x_min: float, x_max: float, bin_width: float) -> np.ndarray:
    left = np.floor(x_min / bin_width) * bin_width
    right = np.ceil(x_max / bin_width) * bin_width

    if right <= left:
        right = left + bin_width

    return np.arange(left, right + bin_width * 0.5, bin_width)


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["soc_true_pct"] = normalize_to_percent(df["soc_true"])
    df["soh_true_pct"] = normalize_to_percent(df["soh_true"])

    soc_edges = np.arange(
        SOC_BIN_RANGE[0],
        SOC_BIN_RANGE[1] + SOC_BIN_WIDTH * 0.5,
        SOC_BIN_WIDTH,
    )

    if SOH_BIN_RANGE is None:
        soh_values = df["soh_true_pct"].to_numpy(dtype=float)
        soh_values = soh_values[np.isfinite(soh_values)]

        if len(soh_values) == 0:
            raise RuntimeError("No valid SOH values found for binning.")

        soh_edges = make_edges_from_range(
            x_min=np.nanmin(soh_values),
            x_max=np.nanmax(soh_values),
            bin_width=SOH_BIN_WIDTH,
        )
    else:
        soh_edges = np.arange(
            SOH_BIN_RANGE[0],
            SOH_BIN_RANGE[1] + SOH_BIN_WIDTH * 0.5,
            SOH_BIN_WIDTH,
        )

    df["soc_bin"] = pd.cut(
        df["soc_true_pct"],
        bins=soc_edges,
        include_lowest=True,
        right=True,
    )

    df["soh_bin"] = pd.cut(
        df["soh_true_pct"],
        bins=soh_edges,
        include_lowest=True,
        right=True,
    )

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


def round_up_nice(x: float) -> int:
    if not np.isfinite(x) or x <= 0:
        return 1

    if x <= 10:
        return 10
    if x <= 20:
        return int(math.ceil(x / 2.0) * 2)
    if x <= 50:
        return int(math.ceil(x / 5.0) * 5)
    if x <= 100:
        return int(math.ceil(x / 10.0) * 10)
    if x <= 200:
        return int(math.ceil(x / 20.0) * 20)

    return int(math.ceil(x / 50.0) * 50)


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


def build_bin_count_pivot(df: pd.DataFrame, bin_col: str) -> pd.DataFrame:
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
    pivot: pd.DataFrame,
    threshold: int = LOW_COUNT_THRESHOLD,
):
    nrows, ncols = pivot.shape

    for i in range(nrows):
        for j in range(ncols):
            value = pivot.iloc[i, j]

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


def draw_heatmap_on_axis(
    ax,
    pivot: pd.DataFrame,
    cmap_name: str,
    vmin: float,
    vmax: float,
    title: str,
    use_missing_background: bool,
    cbar_ax=None,
    cbar_label: str = "Count",
):
    mask = pivot.isnull()

    if use_missing_background:
        add_missing_background(ax, pivot)

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
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

    add_lowcount_hatch(ax, pivot, LOW_COUNT_THRESHOLD)

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
    pivot_soc: pd.DataFrame,
    pivot_soh: pd.DataFrame,
    save_png: Path,
    save_pdf: Path,
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
        title="Material confusion count",
        use_missing_background=False,
        cbar_ax=cax_conf,
        cbar_label="Count",
    )

    draw_heatmap_on_axis(
        ax=ax_soc,
        pivot=pivot_soc,
        cmap_name=CMAP_SOC,
        vmin=VMIN,
        vmax=vmax_soc,
        title=f"SOC-bin sample count (hatched if n < {LOW_COUNT_THRESHOLD})",
        use_missing_background=True,
        cbar_ax=cax_soc,
        cbar_label="Count",
    )

    draw_heatmap_on_axis(
        ax=ax_soh,
        pivot=pivot_soh,
        cmap_name=CMAP_SOH,
        vmin=VMIN,
        vmax=vmax_soh,
        title=f"SOH-bin sample count (hatched if n < {LOW_COUNT_THRESHOLD})",
        use_missing_background=True,
        cbar_ax=cax_soh,
        cbar_label="Count",
    )

    fig.savefig(save_png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(save_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    print("[INFO] Generating Supplementary Figure 6...")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    check_required_columns(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = add_bins(df)

    pivot_conf = build_confusion_count_pivot(df)
    pivot_soc = build_bin_count_pivot(df, "soc_bin")
    pivot_soh = build_bin_count_pivot(df, "soh_bin")

    vmax_conf = max(round_up_nice(safe_nanmax(pivot_conf.values)), 1)
    vmax_soc = max(round_up_nice(safe_nanmax(pivot_soc.values)), 1)
    vmax_soh = max(round_up_nice(safe_nanmax(pivot_soh.values)), 1)

    conf_csv = OUT_DIR / "suppfig6_material_confusion_count_matrix.csv"
    soc_csv = OUT_DIR / "suppfig6_soc_bin_material_count_matrix.csv"
    soh_csv = OUT_DIR / "suppfig6_soh_bin_material_count_matrix.csv"

    pivot_conf.to_csv(conf_csv, encoding="utf-8-sig")
    pivot_soc.to_csv(soc_csv, encoding="utf-8-sig")
    pivot_soh.to_csv(soh_csv, encoding="utf-8-sig")

    png_path = OUT_DIR / "suppfig6_count_companion_heatmaps.png"
    pdf_path = OUT_DIR / "suppfig6_count_companion_heatmaps.pdf"

    draw_combined_heatmaps(
        pivot_conf=pivot_conf,
        pivot_soc=pivot_soc,
        pivot_soh=pivot_soh,
        save_png=png_path,
        save_pdf=pdf_path,
        vmax_conf=vmax_conf,
        vmax_soc=vmax_soc,
        vmax_soh=vmax_soh,
    )

    print("[DONE] Supplementary Figure 6 generated.")
    print(f"[SAVED] Figure PNG: {png_path}")
    print(f"[SAVED] Figure PDF: {pdf_path}")
    print(f"[SAVED] Confusion matrix: {conf_csv}")
    print(f"[SAVED] SOC count matrix: {soc_csv}")
    print(f"[SAVED] SOH count matrix: {soh_csv}")


if __name__ == "__main__":
    main()