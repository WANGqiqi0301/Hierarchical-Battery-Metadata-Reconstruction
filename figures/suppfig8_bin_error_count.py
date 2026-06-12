"""
suppfig8_bin_error_count.py

Supplementary Figure 8:
SOC/SOH bin-wise sample size and error uncertainty.

This script reads the proposed_framework per-sample test prediction file and computes:
1. SOC/SOH APE from true and predicted values
2. SOC/SOH bins from true values
3. Bin-wise sample count
4. Bin-wise median APE
5. Bin-wise mean APE
6. Bootstrap 95% CI of median APE

Expected input columns:
    ID, pulse_ms, true_label, pred_label, soc_true, soc_pred, soh_true, soh_pred

Input:
    results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv

Output:
    results/figures/supp/suppfig8

Generated files:
    suppfig8_bin_error_count.png
    suppfig8_bin_error_count.pdf
    suppfig8_soc_bin_summary.csv
    suppfig8_soh_bin_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path.cwd()

CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "proposed_framework"
    / "further_analysis"
    / "tables"
    / "test_predictions_per_sample.csv"
)

OUT_DIR = PROJECT_ROOT / "results" / "figures" / "supp" / "suppfig8"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Global style
# ============================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})


COLOR_SOC = "#5B7894"
COLOR_SOH = "#A97C62"
COLOR_COUNT = "#B8B8B8"
COLOR_MEAN = "#444444"
COLOR_CI = "#AFC1D1"
COLOR_CI_SOH = "#D5B8A6"


# ============================================================
# Bin configuration
# ============================================================
SOC_BIN_WIDTH = 10.0
SOH_BIN_WIDTH = 5.0

SOC_BIN_RANGE = (0.0, 100.0)
SOH_BIN_RANGE = None

BOOTSTRAP_N = 2000
BOOTSTRAP_CI = 95
BOOTSTRAP_SEED = 42


# ============================================================
# Helper functions
# ============================================================
def check_required_columns(df: pd.DataFrame):
    required_cols = [
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


def normalize_to_percent(values: pd.Series | np.ndarray) -> np.ndarray:
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

    finite_values = values[np.isfinite(values)]
    if len(finite_values) > 0 and np.nanmax(finite_values) <= 1.5:
        values = values * 100.0

    return values


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


def bootstrap_median_ci(
    values,
    n_boot: int = BOOTSTRAP_N,
    ci: float = BOOTSTRAP_CI,
    seed: int = BOOTSTRAP_SEED,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return values[0], values[0]

    rng = np.random.default_rng(seed)
    boot_medians = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_medians[i] = np.median(sample)

    alpha = (100.0 - ci) / 2.0
    lo = np.percentile(boot_medians, alpha)
    hi = np.percentile(boot_medians, 100.0 - alpha)

    return lo, hi


def make_edges_from_range(x_min: float, x_max: float, bin_width: float) -> np.ndarray:
    left = np.floor(x_min / bin_width) * bin_width
    right = np.ceil(x_max / bin_width) * bin_width

    if right <= left:
        right = left + bin_width

    return np.arange(left, right + bin_width * 0.5, bin_width)


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["soc_true_pct"] = normalize_to_percent(df["soc_true"])
    df["soc_pred_pct"] = normalize_to_percent(df["soc_pred"])
    df["soh_true_pct"] = normalize_to_percent(df["soh_true"])
    df["soh_pred_pct"] = normalize_to_percent(df["soh_pred"])

    df["soc_ape_pct"] = safe_ape_pct(df["soc_true_pct"], df["soc_pred_pct"])
    df["soh_ape_pct"] = safe_ape_pct(df["soh_true_pct"], df["soh_pred_pct"])

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


def interval_to_label(interval) -> str:
    if pd.isna(interval):
        return ""

    left = float(interval.left)
    right = float(interval.right)

    return f"{left:.1f}-{right:.1f}"


def interval_mid(interval) -> float:
    if pd.isna(interval):
        return np.nan

    return (float(interval.left) + float(interval.right)) / 2.0


def summarize_by_bin(df: pd.DataFrame, bin_col: str, err_col: str) -> pd.DataFrame:
    rows = []

    grouped = df.groupby(bin_col, dropna=True, observed=False)

    for bin_interval, group in grouped:
        values = pd.to_numeric(group[err_col], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        ci_lo, ci_hi = bootstrap_median_ci(values)

        rows.append({
            "bin": str(bin_interval),
            "left": float(bin_interval.left),
            "right": float(bin_interval.right),
            "mid": interval_mid(bin_interval),
            "label": interval_to_label(bin_interval),
            "count": len(values),
            "median": np.median(values),
            "mean": np.mean(values),
            "q25": np.percentile(values, 25),
            "q75": np.percentile(values, 75),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })

    summary = pd.DataFrame(rows)

    if summary.empty:
        raise RuntimeError(f"No valid rows found when summarizing {bin_col}.")

    summary = summary.sort_values(["left", "right"]).reset_index(drop=True)

    return summary


def plot_error_panel(
    ax,
    summary: pd.DataFrame,
    color: str,
    ci_color: str,
    title: str,
    ylabel: str,
    show_legend: bool = True,
):
    x = np.arange(len(summary))
    y_med = summary["median"].to_numpy(dtype=float)
    y_mean = summary["mean"].to_numpy(dtype=float)
    y_lo = summary["ci_lo"].to_numpy(dtype=float)
    y_hi = summary["ci_hi"].to_numpy(dtype=float)

    ax.fill_between(
        x,
        y_lo,
        y_hi,
        color=ci_color,
        alpha=0.45,
        linewidth=0,
        label="95% CI of median",
    )

    ax.plot(
        x,
        y_med,
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        color=color,
        label="Median APE",
    )

    ax.plot(
        x,
        y_mean,
        linestyle="--",
        linewidth=1.3,
        color=COLOR_MEAN,
        alpha=0.85,
        label="Mean APE",
    )

    ax.set_title(title, fontsize=9, pad=5)
    ax.set_ylabel(ylabel)

    ax.set_xticks(x)
    ax.set_xticklabels(summary["label"].tolist(), rotation=45, ha="right")

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        ax.legend(
            frameon=False,
            fontsize=7,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.28),
            ncol=3,
            handlelength=1.8,
            columnspacing=1.1,
        )


def plot_count_panel(
    ax,
    summary: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
):
    x = np.arange(len(summary))
    counts = summary["count"].to_numpy(dtype=float)

    ax.bar(
        x,
        counts,
        width=0.72,
        color=COLOR_COUNT,
        edgecolor="none",
        alpha=0.85,
    )

    ax.set_title(title, fontsize=9, pad=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(x)
    ax.set_xticklabels(summary["label"].tolist(), rotation=45, ha="right")

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.30)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(soc_summary: pd.DataFrame, soh_summary: pd.DataFrame):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 4.6),
        gridspec_kw={
            "height_ratios": [2.1, 1.0],
            "hspace": 0.55,
            "wspace": 0.28,
        },
    )

    ax_soc_err = axes[0, 0]
    ax_soh_err = axes[0, 1]
    ax_soc_cnt = axes[1, 0]
    ax_soh_cnt = axes[1, 1]

    plot_error_panel(
        ax=ax_soc_err,
        summary=soc_summary,
        color=COLOR_SOC,
        ci_color=COLOR_CI,
        title="SOC bins",
        ylabel="SOC APE (%)",
        show_legend=True,
    )

    plot_error_panel(
        ax=ax_soh_err,
        summary=soh_summary,
        color=COLOR_SOH,
        ci_color=COLOR_CI_SOH,
        title="SOH bins",
        ylabel="SOH APE (%)",
        show_legend=True,
    )

    plot_count_panel(
        ax=ax_soc_cnt,
        summary=soc_summary,
        title="SOC-bin sample count",
        xlabel="SOC bin (%)",
        ylabel="Count",
    )

    plot_count_panel(
        ax=ax_soh_cnt,
        summary=soh_summary,
        title="SOH-bin sample count",
        xlabel="SOH bin (%)",
        ylabel="Count",
    )

    return fig


# ============================================================
# Main
# ============================================================
def main():
    print("[INFO] Generating Supplementary Figure 8...")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    check_required_columns(df)

    df = add_bins(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    soc_summary = summarize_by_bin(
        df=df,
        bin_col="soc_bin",
        err_col="soc_ape_pct",
    )

    soh_summary = summarize_by_bin(
        df=df,
        bin_col="soh_bin",
        err_col="soh_ape_pct",
    )

    soc_summary_path = OUT_DIR / "suppfig8_soc_bin_summary.csv"
    soh_summary_path = OUT_DIR / "suppfig8_soh_bin_summary.csv"

    soc_summary.to_csv(soc_summary_path, index=False, encoding="utf-8-sig")
    soh_summary.to_csv(soh_summary_path, index=False, encoding="utf-8-sig")

    fig = make_figure(soc_summary, soh_summary)

    png_path = OUT_DIR / "suppfig8_bin_error_count.png"
    pdf_path = OUT_DIR / "suppfig8_bin_error_count.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print("[DONE] Supplementary Figure 8 generated.")
    print(f"[SAVED] Figure PNG: {png_path}")
    print(f"[SAVED] Figure PDF: {pdf_path}")
    print(f"[SAVED] SOC summary: {soc_summary_path}")
    print(f"[SAVED] SOH summary: {soh_summary_path}")


if __name__ == "__main__":
    main()