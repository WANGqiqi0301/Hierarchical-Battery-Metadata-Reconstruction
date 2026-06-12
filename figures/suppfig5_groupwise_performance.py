# -*- coding: utf-8 -*-
"""
suppfig5_groupwise_performance.py

Supplementary Figure 5:
Group-wise test performance by material-capacity group.

This script reads the proposed_framework per-sample test prediction file and computes:
1. Classification accuracy (%)
2. SOC MedAPE (%)
3. SOH MedAPE (%)
4. Sample count

Expected input columns:
    ID, pulse_ms, true_label, pred_label, soc_true, soc_pred, soh_true, soh_pred

Input:
    results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv

Output:
    results/figures/supp/suppfig5

Generated files:
    suppfig5_groupwise_performance.png
    suppfig5_groupwise_metrics.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================================================
# Project paths
# =========================================================
PROJECT_ROOT = Path.cwd()

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "proposed_framework"
    / "further_analysis"
    / "tables"
    / "test_predictions_per_sample.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "figures" / "supp" / "suppfig5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Plot style
# =========================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})

MATERIAL_COLORS = {
    "LFP": "#6E8FA8",
    "LMO": "#B88A64",
    "NMC": "#7E9871",
}

NEUTRAL_LINE = "#D6D6D6"
GRID_COLOR = "#E9E9E9"
TEXT_COLOR = "#222222"

PREFERRED_ORDER = [
    "LFP_35Ah",
    "LFP_68Ah",
    "LMO_10Ah",
    "LMO_24Ah",
    "LMO_25Ah",
    "LMO_26Ah",
    "NMC_15Ah",
    "NMC_21Ah",
]


# =========================================================
# Helper functions
# =========================================================
def cm_to_inch(x: float) -> float:
    return x / 2.54


def get_material(label: str) -> str:
    if not isinstance(label, str):
        return "Unknown"
    return label.split("_")[0]


def get_capacity_number(label: str) -> int:
    if not isinstance(label, str):
        return 999

    match = re.search(r"_(\d+)\s*Ah", label)
    if match:
        return int(match.group(1))

    return 999


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(
        axis="both",
        which="both",
        width=0.8,
        length=3,
        labelsize=8,
        colors=TEXT_COLOR,
    )

    ax.grid(
        axis="x",
        linestyle="-",
        linewidth=0.6,
        color=GRID_COLOR,
        alpha=1.0,
    )

    ax.set_axisbelow(True)


def add_panel_label(ax, label: str):
    ax.text(
        -0.16,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=TEXT_COLOR,
    )


def nice_upper_limit(
    values,
    min_upper=None,
    pad_ratio: float = 0.18,
    round_base: float = 1.0,
) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float(min_upper) if min_upper is not None else 1.0

    vmax = np.nanmax(values)
    upper = vmax * (1.0 + pad_ratio)

    if min_upper is not None:
        upper = max(upper, min_upper)

    upper = np.ceil(upper / round_base) * round_base
    return upper


def annotate_right(
    ax,
    xvals,
    yvals,
    xlim_right: float,
    fmt: str,
    fontsize: float = 7.6,
):
    offset = 0.04 * xlim_right

    for xv, yv in zip(xvals, yvals):
        xt = min(xv + offset, xlim_right * 0.965)

        ax.text(
            xt,
            yv,
            format(xv, fmt),
            ha="left",
            va="center",
            fontsize=fontsize,
            color=TEXT_COLOR,
            clip_on=False,
        )


def check_required_columns(df: pd.DataFrame):
    required_cols = [
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


def safe_ape_pct(y_true, y_pred) -> np.ndarray:
    y_true = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(y_pred, errors="coerce").to_numpy(dtype=float)

    denom = np.abs(y_true)
    ape = np.full_like(y_true, np.nan, dtype=float)

    valid = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & np.isfinite(denom)
        & (denom > 1e-12)
    )

    ape[valid] = np.abs(y_pred[valid] - y_true[valid]) / denom[valid] * 100.0

    return ape


def build_groupwise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    check_required_columns(df)

    df = df.copy()

    df["soc_true"] = pd.to_numeric(df["soc_true"], errors="coerce")
    df["soc_pred"] = pd.to_numeric(df["soc_pred"], errors="coerce")
    df["soh_true"] = pd.to_numeric(df["soh_true"], errors="coerce")
    df["soh_pred"] = pd.to_numeric(df["soh_pred"], errors="coerce")

    df["soc_ape_pct"] = safe_ape_pct(df["soc_true"], df["soc_pred"])
    df["soh_ape_pct"] = safe_ape_pct(df["soh_true"], df["soh_pred"])

    df["is_correct"] = (
        df["true_label"].astype(str) == df["pred_label"].astype(str)
    ).astype(float)

    agg = (
        df.groupby("true_label", dropna=False)
        .agg(
            cls_acc=("is_correct", lambda x: 100.0 * np.mean(x)),
            soc_medape=("soc_ape_pct", lambda x: np.nanmedian(x)),
            soh_medape=("soh_ape_pct", lambda x: np.nanmedian(x)),
            sample_count=("true_label", "size"),
        )
        .reset_index()
        .rename(columns={"true_label": "group"})
    )

    agg["group"] = agg["group"].astype(str)
    agg["material"] = agg["group"].apply(get_material)
    agg["capacity_num"] = agg["group"].apply(get_capacity_number)
    agg["color"] = agg["material"].map(MATERIAL_COLORS).fillna("#888888")

    present = agg["group"].tolist()

    ordered = [g for g in PREFERRED_ORDER if g in present]
    extras = [g for g in present if g not in ordered]
    extras = sorted(extras, key=lambda x: (get_material(x), get_capacity_number(x), x))

    final_order = ordered + extras
    order_map = {g: i for i, g in enumerate(final_order)}

    agg["order"] = agg["group"].map(order_map)
    agg = agg.sort_values("order").reset_index(drop=True)

    return agg


def plot_groupwise_performance(plot_df: pd.DataFrame, save_path: Path):
    groups = plot_df["group"].tolist()
    colors = plot_df["color"].tolist()
    y = np.arange(len(plot_df))

    cls_vals = plot_df["cls_acc"].to_numpy(dtype=float)
    soc_vals = plot_df["soc_medape"].to_numpy(dtype=float)
    soh_vals = plot_df["soh_medape"].to_numpy(dtype=float)
    n_vals = plot_df["sample_count"].to_numpy(dtype=float)

    fig_w = cm_to_inch(19.0)
    fig_h = cm_to_inch(6.8)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(fig_w, fig_h),
        sharey=True,
        gridspec_kw={"width_ratios": [1.20, 1.15, 1.15, 0.95]},
    )

    ax1, ax2, ax3, ax4 = axes

    cls_xlim = nice_upper_limit(cls_vals, min_upper=100, pad_ratio=0.22, round_base=5)
    soc_xlim = nice_upper_limit(soc_vals, min_upper=None, pad_ratio=0.28, round_base=1)
    soh_xlim = nice_upper_limit(soh_vals, min_upper=None, pad_ratio=0.30, round_base=0.5)
    cnt_xlim = nice_upper_limit(n_vals, min_upper=None, pad_ratio=0.28, round_base=10)

    ax1.hlines(y, 0, cls_vals, color=NEUTRAL_LINE, linewidth=1.4, zorder=1)
    ax1.scatter(cls_vals, y, s=38, c=colors, edgecolors="none", zorder=3)
    ax1.set_xlabel("Classification accuracy (%)", fontsize=8.5, color=TEXT_COLOR)
    ax1.set_xlim(0, cls_xlim)
    ax1.set_yticks(y)
    ax1.set_yticklabels(groups, fontsize=8.2, color=TEXT_COLOR)
    ax1.invert_yaxis()
    annotate_right(ax1, cls_vals, y, cls_xlim, ".1f", fontsize=7.6)

    ax2.hlines(y, 0, soc_vals, color=NEUTRAL_LINE, linewidth=1.4, zorder=1)
    ax2.scatter(soc_vals, y, s=38, c=colors, edgecolors="none", zorder=3)
    ax2.set_xlabel("SOC MedAPE (%)", fontsize=8.5, color=TEXT_COLOR)
    ax2.set_xlim(0, soc_xlim)
    annotate_right(ax2, soc_vals, y, soc_xlim, ".2f", fontsize=7.6)

    ax3.hlines(y, 0, soh_vals, color=NEUTRAL_LINE, linewidth=1.4, zorder=1)
    ax3.scatter(soh_vals, y, s=38, c=colors, edgecolors="none", zorder=3)
    ax3.set_xlabel("SOH MedAPE (%)", fontsize=8.5, color=TEXT_COLOR)
    ax3.set_xlim(0, soh_xlim)
    annotate_right(ax3, soh_vals, y, soh_xlim, ".2f", fontsize=7.6)

    ax4.barh(
        y,
        n_vals,
        color=colors,
        alpha=0.92,
        height=0.52,
        edgecolor="none",
        zorder=2,
    )
    ax4.set_xlabel("Sample count", fontsize=8.5, color=TEXT_COLOR)
    ax4.set_xlim(0, cnt_xlim)
    annotate_right(ax4, n_vals, y, cnt_xlim, ".0f", fontsize=7.6)

    for ax in axes:
        style_ax(ax)

    ax2.tick_params(axis="y", left=False, labelleft=False)
    ax3.tick_params(axis="y", left=False, labelleft=False)
    ax4.tick_params(axis="y", left=False, labelleft=False)

    add_panel_label(ax1, "a")
    add_panel_label(ax2, "b")
    add_panel_label(ax3, "c")
    add_panel_label(ax4, "d")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=5.8,
            markerfacecolor=MATERIAL_COLORS["LFP"],
            markeredgecolor="none",
            label="LFP",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=5.8,
            markerfacecolor=MATERIAL_COLORS["LMO"],
            markeredgecolor="none",
            label="LMO",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=5.8,
            markerfacecolor=MATERIAL_COLORS["NMC"],
            markeredgecolor="none",
            label="NMC",
        ),
    ]

    fig.legend(
        handles=handles,
        labels=["LFP", "LMO", "NMC"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.2,
        fontsize=8.3,
    )

    plt.subplots_adjust(
        left=0.20,
        right=0.995,
        bottom=0.19,
        top=0.82,
        wspace=0.24,
    )

    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    print("[INFO] Generating Supplementary Figure 5...")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    plot_df = build_groupwise_metrics(df)

    metrics_path = OUTPUT_DIR / "suppfig5_groupwise_metrics.csv"
    plot_df[
        ["group", "material", "cls_acc", "soc_medape", "soh_medape", "sample_count"]
    ].to_csv(metrics_path, index=False, encoding="utf-8-sig")

    fig_path = OUTPUT_DIR / "suppfig5_groupwise_performance.png"
    plot_groupwise_performance(plot_df, fig_path)

    print("[DONE] Supplementary Figure 5 generated.")
    print(f"[SAVED] Figure: {fig_path}")
    print(f"[SAVED] Metrics: {metrics_path}")


if __name__ == "__main__":
    main()