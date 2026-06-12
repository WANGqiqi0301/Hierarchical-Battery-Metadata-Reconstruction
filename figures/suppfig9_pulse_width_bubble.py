# -*- coding: utf-8 -*-
"""
suppfig9_pulse_width_bubble.py

Supplementary Figure 9:
Pulse-width bubble heatmaps by material-capacity group.

This script reads the proposed_framework per-sample test prediction file and computes:
1. SOC MedAPE by material-capacity group and pulse width
2. SOH MedAPE by material-capacity group and pulse width
3. Sample count by material-capacity group and pulse width

Expected input columns:
    ID, pulse_ms, true_label, pred_label, soc_true, soc_pred, soh_true, soh_pred

Input:
    results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv

Output:
    results/figures/supp/suppfig9

Generated files:
    suppfig9_pulse_width_bubble_SOC.png
    suppfig9_pulse_width_bubble_SOH.png
    suppfig9_pulse_width_metrics_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path.cwd().resolve()

CSV_PATH = (
    PROJECT_ROOT
    / "results"
    / "proposed_framework"
    / "further_analysis"
    / "tables"
    / "test_predictions_per_sample.csv"
)

SAVE_DIR = PROJECT_ROOT / "results" / "figures" / "supp" / "suppfig9"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Style settings
# ============================================================
mpl.rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "savefig.dpi": 600,
})

CMAP_SOC = "YlGnBu"
CMAP_SOH = "YlOrBr"

GROUP_ORDER = [
    "LFP_35Ah",
    "LFP_68Ah",
    "LMO_10Ah",
    "LMO_24Ah",
    "LMO_25Ah",
    "LMO_26Ah",
    "NMC_15Ah",
    "NMC_21Ah",
]


# ============================================================
# Helper functions
# ============================================================
def check_required_columns(df: pd.DataFrame):
    required = [
        "ID",
        "pulse_ms",
        "true_label",
        "pred_label",
        "soc_true",
        "soc_pred",
        "soh_true",
        "soh_pred",
    ]

    missing = [c for c in required if c not in df.columns]

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


def format_pulse_label(x) -> str:
    if pd.isna(x):
        return ""

    x = float(x)

    if x.is_integer():
        return str(int(x))

    return f"{x:g}"


def size_map(
    n,
    n_min: float,
    n_max: float,
    s_min: float = 35,
    s_max: float = 260,
) -> np.ndarray:
    n = np.asarray(n, dtype=float)

    if n_max == n_min:
        return np.full_like(n, (s_min + s_max) / 2.0)

    return s_min + (n - n_min) / (n_max - n_min) * (s_max - s_min)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["pulse_ms"] = pd.to_numeric(df["pulse_ms"], errors="coerce")
    df["soc_true"] = pd.to_numeric(df["soc_true"], errors="coerce")
    df["soc_pred"] = pd.to_numeric(df["soc_pred"], errors="coerce")
    df["soh_true"] = pd.to_numeric(df["soh_true"], errors="coerce")
    df["soh_pred"] = pd.to_numeric(df["soh_pred"], errors="coerce")

    df["soc_ape_pct"] = safe_ape_pct(df["soc_true"], df["soc_pred"])
    df["soh_ape_pct"] = safe_ape_pct(df["soh_true"], df["soh_pred"])

    df["correct_cls"] = (
        df["true_label"].astype(str) == df["pred_label"].astype(str)
    ).astype(float)

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(
        subset=[
            "ID",
            "pulse_ms",
            "true_label",
            "pred_label",
            "soc_ape_pct",
            "soh_ape_pct",
        ]
    ).copy()

    return df


def build_summary(df: pd.DataFrame):
    present_groups = [
        g for g in GROUP_ORDER
        if g in sorted(df["true_label"].astype(str).unique())
    ]
    extra_groups = [
        g for g in sorted(df["true_label"].astype(str).unique())
        if g not in present_groups
    ]

    group_order = present_groups + extra_groups

    pulse_order = sorted(df["pulse_ms"].dropna().unique())
    pulse_to_x = {p: i for i, p in enumerate(pulse_order)}

    summary = (
        df.groupby(["true_label", "pulse_ms"], as_index=False)
        .agg(
            n=("ID", "count"),
            cls_acc=("correct_cls", "mean"),
            soc_medape=("soc_ape_pct", "median"),
            soh_medape=("soh_ape_pct", "median"),
            soc_meanape=("soc_ape_pct", "mean"),
            soh_meanape=("soh_ape_pct", "mean"),
            soc_q25=("soc_ape_pct", lambda x: np.percentile(x, 25)),
            soc_q75=("soc_ape_pct", lambda x: np.percentile(x, 75)),
            soh_q25=("soh_ape_pct", lambda x: np.percentile(x, 25)),
            soh_q75=("soh_ape_pct", lambda x: np.percentile(x, 75)),
        )
    )

    summary["x"] = summary["pulse_ms"].map(pulse_to_x)
    summary["cls_acc_pct"] = summary["cls_acc"] * 100.0

    return summary, group_order, pulse_order


# ============================================================
# Bubble plot
# ============================================================
def plot_bubble_metric(
    summary: pd.DataFrame,
    group_order: list[str],
    pulse_order: list[float],
    metric_col: str,
    metric_label: str,
    file_tag: str,
    cmap: str,
):
    plot_df = summary.copy()
    plot_df["group_y"] = plot_df["true_label"].map(
        {g: i for i, g in enumerate(group_order)}
    )

    n_min = float(plot_df["n"].min())
    n_max = float(plot_df["n"].max())

    fig_w = 6.2
    fig_h = max(3.0, 0.34 * len(group_order) + 1.35)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmin = 0
    vmax = plot_df[metric_col].max()
    vmax = np.ceil(vmax * 1.05 / 5.0) * 5.0

    if vmax < 10:
        vmax = 10

    sizes = size_map(
        plot_df["n"].values,
        n_min,
        n_max,
        s_min=35,
        s_max=260,
    )

    scatter = ax.scatter(
        plot_df["x"],
        plot_df["group_y"],
        s=sizes,
        c=plot_df[metric_col],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xticks(range(len(pulse_order)))
    ax.set_xticklabels(
        [format_pulse_label(p) for p in pulse_order],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
        fontsize=8,
    )

    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels(group_order, fontsize=8)

    ax.invert_yaxis()

    ax.set_xlabel("Pulse width (ms)", fontsize=9)
    ax.set_ylabel("Material-capacity group", fontsize=9)
    ax.set_title(metric_label, fontsize=10, pad=8)

    ax.grid(
        which="major",
        axis="both",
        color="#DCDCDC",
        linewidth=0.5,
        alpha=0.65,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.025, fraction=0.05)
    cbar.set_label("Median APE (%)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    example_ns = sorted(set([
        int(n_min),
        int(np.median(plot_df["n"])),
        int(n_max),
    ]))

    size_handles = []
    for n in example_ns:
        size_handles.append(
            ax.scatter(
                [],
                [],
                s=size_map([n], n_min, n_max, s_min=35, s_max=260)[0],
                color="#BDBDBD",
                alpha=0.8,
                edgecolor="white",
                linewidth=0.5,
                label=f"n={n}",
            )
        )

    ax.legend(
        handles=size_handles,
        title="Sample size",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=len(size_handles),
        frameon=False,
        fontsize=7.5,
        title_fontsize=8,
        handletextpad=0.6,
        columnspacing=1.2,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    out_png = SAVE_DIR / f"suppfig9_pulse_width_bubble_{file_tag}.png"

    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return out_png


# ============================================================
# Main
# ============================================================
def main():
    print("[INFO] Generating Supplementary Figure 9...")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    check_required_columns(df)

    df = prepare_data(df)

    summary, group_order, pulse_order = build_summary(df)

    summary_path = SAVE_DIR / "suppfig9_pulse_width_metrics_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    soc_png = plot_bubble_metric(
        summary=summary,
        group_order=group_order,
        pulse_order=pulse_order,
        metric_col="soc_medape",
        metric_label="SOC MedAPE",
        file_tag="SOC",
        cmap=CMAP_SOC,
    )

    soh_png = plot_bubble_metric(
        summary=summary,
        group_order=group_order,
        pulse_order=pulse_order,
        metric_col="soh_medape",
        metric_label="SOH MedAPE",
        file_tag="SOH",
        cmap=CMAP_SOH,
    )

    print("[DONE] Supplementary Figure 9 generated.")
    print(f"[SAVED] Summary: {summary_path}")
    print(f"[SAVED] SOC bubble: {soc_png}")
    print(f"[SAVED] SOH bubble: {soh_png}")


if __name__ == "__main__":
    main()