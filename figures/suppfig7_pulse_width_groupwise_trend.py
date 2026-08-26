# -*- coding: utf-8 -*-
"""
suppfig7_pulse_width_groupwise_trend.py

Supplementary Figure 7:
Group-wise pulse-width trend plot.

This script reads the proposed_framework per-sample test prediction file and computes:
1. SOC MedAE by material-capacity group and pulse width
2. SOH MedAE by material-capacity group and pulse width
3. Sample count by material-capacity group and pulse width

Expected input columns:
    ID, pulse_ms, true_label, pred_label, soc_true, soc_pred, soh_true, soh_pred

Input:
    results/proposed_framework/further_analysis/tables/test_predictions_per_sample.csv

Output:
    results/figures/supp/suppfig7

Generated files:
    suppfig7_pulse_width_groupwise_trend.png
    suppfig7_pulse_width_metrics_summary.csv
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

SAVE_DIR = PROJECT_ROOT / "results" / "figures" / "supp" / "suppfig7"
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

COLOR_SOC = "#5B7FA3"
COLOR_SOH = "#C7835A"

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
    s_min: float = 28,
    s_max: float = 180,
) -> np.ndarray:
    n = np.asarray(n, dtype=float)

    if n_max == n_min:
        return np.full_like(n, (s_min + s_max) / 2.0)

    return s_min + (n - n_min) / (n_max - n_min) * (s_max - s_min)


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.5,
        alpha=0.6,
    )


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["pulse_ms"] = pd.to_numeric(
        df["pulse_ms"],
        errors="coerce",
    )

    df["soc_true"] = pd.to_numeric(
        df["soc_true"],
        errors="coerce",
    )

    df["soc_pred"] = pd.to_numeric(
        df["soc_pred"],
        errors="coerce",
    )

    df["soh_true"] = pd.to_numeric(
        df["soh_true"],
        errors="coerce",
    )

    df["soh_pred"] = pd.to_numeric(
        df["soh_pred"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Absolute errors for MedAE
    # --------------------------------------------------------
    df["soc_ae"] = np.abs(
        df["soc_pred"] - df["soc_true"]
    )

    df["soh_ae"] = np.abs(
        df["soh_pred"] - df["soh_true"]
    )

    df["correct_cls"] = (
        df["true_label"].astype(str)
        ==
        df["pred_label"].astype(str)
    ).astype(float)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=[
            "ID",
            "pulse_ms",
            "true_label",
            "pred_label",
            "soc_ae",
            "soh_ae",
        ]
    ).copy()

    return df


def build_summary(df: pd.DataFrame):
    present_groups = [
        g
        for g in GROUP_ORDER
        if g in sorted(
            df["true_label"]
            .astype(str)
            .unique()
        )
    ]

    extra_groups = [
        g
        for g in sorted(
            df["true_label"]
            .astype(str)
            .unique()
        )
        if g not in present_groups
    ]

    group_order = (
        present_groups
        +
        extra_groups
    )

    pulse_order = sorted(
        df["pulse_ms"]
        .dropna()
        .unique()
    )

    pulse_to_x = {
        p: i
        for i, p in enumerate(pulse_order)
    }

    summary = (
        df.groupby(
            ["true_label", "pulse_ms"],
            as_index=False,
        )
        .agg(
            n=("ID", "count"),

            cls_acc=(
                "correct_cls",
                "mean",
            ),

            soc_medae=(
                "soc_ae",
                "median",
            ),

            soh_medae=(
                "soh_ae",
                "median",
            ),

            soc_meanae=(
                "soc_ae",
                "mean",
            ),

            soh_meanae=(
                "soh_ae",
                "mean",
            ),

            soc_q25=(
                "soc_ae",
                lambda x: np.percentile(x, 25),
            ),

            soc_q75=(
                "soc_ae",
                lambda x: np.percentile(x, 75),
            ),

            soh_q25=(
                "soh_ae",
                lambda x: np.percentile(x, 25),
            ),

            soh_q75=(
                "soh_ae",
                lambda x: np.percentile(x, 75),
            ),
        )
    )

    summary["x"] = (
        summary["pulse_ms"]
        .map(pulse_to_x)
    )

    summary["cls_acc_pct"] = (
        summary["cls_acc"]
        * 100.0
    )

    return (
        summary,
        group_order,
        pulse_order,
    )


# ============================================================
# Plot
# ============================================================
def plot_groupwise_trend(
    summary: pd.DataFrame,
    group_order: list[str],
    pulse_order: list[float],
):
    n_groups = len(group_order)
    ncols = 4
    nrows = int(
        np.ceil(
            n_groups / ncols
        )
    )

    fig_w = 7.2
    fig_h = 1.75 * nrows + 0.65

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    n_min = float(
        summary["n"].min()
    )

    n_max = float(
        summary["n"].max()
    )

    y_max = max(
        summary["soc_medae"].max(),
        summary["soh_medae"].max(),
    )

    # Use a tighter scale suitable for MedAE values.
    if y_max <= 2:
        y_max = np.ceil(y_max * 1.20 / 0.25) * 0.25
    elif y_max <= 5:
        y_max = np.ceil(y_max * 1.20 / 0.5) * 0.5
    elif y_max <= 10:
        y_max = np.ceil(y_max * 1.20)
    else:
        y_max = np.ceil(y_max * 1.15 / 2.0) * 2.0

    if y_max <= 0 or not np.isfinite(y_max):
        y_max = 1.0

    for ax_idx, group in enumerate(group_order):
        ax = axes[ax_idx]

        sub = (
            summary[
                summary["true_label"] == group
            ]
            .sort_values("pulse_ms")
        )

        if sub.empty:
            ax.axis("off")
            continue

        sizes = size_map(
            sub["n"].values,
            n_min,
            n_max,
        )

        # ----------------------------------------------------
        # SOC MedAE
        # ----------------------------------------------------
        ax.plot(
            sub["x"],
            sub["soc_medae"],
            color=COLOR_SOC,
            linewidth=1.4,
            alpha=0.95,
            zorder=2,
        )

        ax.scatter(
            sub["x"],
            sub["soc_medae"],
            s=sizes,
            color=COLOR_SOC,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

        # ----------------------------------------------------
        # SOH MedAE
        # ----------------------------------------------------
        ax.plot(
            sub["x"],
            sub["soh_medae"],
            color=COLOR_SOH,
            linewidth=1.4,
            alpha=0.95,
            zorder=2,
        )

        ax.scatter(
            sub["x"],
            sub["soh_medae"],
            s=sizes,
            color=COLOR_SOH,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

        ax.set_title(
            group,
            fontsize=8.5,
            pad=4,
        )

        clean_axis(ax)
        ax.set_ylim(0, y_max)

    for j in range(
        len(group_order),
        len(axes),
    ):
        axes[j].axis("off")

    for ax in axes:
        if ax.has_data():
            ax.set_xticks(
                range(
                    len(pulse_order)
                )
            )

            ax.set_xticklabels(
                [
                    format_pulse_label(p)
                    for p in pulse_order
                ],
                rotation=45,
                ha="right",
                rotation_mode="anchor",
                fontsize=7,
            )

            ax.tick_params(
                axis="y",
                labelsize=7,
            )

    fig.supxlabel(
        "Pulse width (ms)",
        fontsize=9,
        y=0.045,
    )

    fig.supylabel(
        "MedAE (%)",
        fontsize=9,
        x=0.045,
    )

    metric_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color=COLOR_SOC,
            label="SOC MedAE",
            markersize=5,
            linewidth=1.5,
        ),

        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color=COLOR_SOH,
            label="SOH MedAE",
            markersize=5,
            linewidth=1.5,
        ),
    ]

    fig.legend(
        handles=metric_handles,
        loc="upper center",
        bbox_to_anchor=(0.42, 1.01),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=1.8,
        columnspacing=1.6,
    )

    example_ns = sorted(
        set(
            [
                int(n_min),
                int(
                    np.median(
                        summary["n"]
                    )
                ),
                int(n_max),
            ]
        )
    )

    size_handles = []

    for n in example_ns:
        size_handles.append(
            plt.scatter(
                [],
                [],
                s=size_map(
                    [n],
                    n_min,
                    n_max,
                )[0],
                color="#BDBDBD",
                alpha=0.7,
                edgecolor="white",
                linewidth=0.5,
                label=f"n={n}",
            )
        )

    fig.legend(
        handles=size_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 1.012),
        ncol=len(size_handles),
        frameon=False,
        fontsize=7,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    plt.tight_layout(
        rect=[
            0.055,
            0.07,
            0.985,
            0.93,
        ]
    )

    out_png = (
        SAVE_DIR
        / "suppfig7_pulse_width_groupwise_trend.png"
    )

    fig.savefig(
        out_png,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    return out_png


# ============================================================
# Main
# ============================================================
def main():
    print(
        "[INFO] Generating Supplementary Figure 7..."
    )

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {CSV_PATH}"
        )

    df = pd.read_csv(
        CSV_PATH
    )

    check_required_columns(
        df
    )

    df = prepare_data(
        df
    )

    (
        summary,
        group_order,
        pulse_order,
    ) = build_summary(
        df
    )

    summary_path = (
        SAVE_DIR
        / "suppfig7_pulse_width_metrics_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    trend_png = plot_groupwise_trend(
        summary=summary,
        group_order=group_order,
        pulse_order=pulse_order,
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "[PULSE-WIDTH GROUP-WISE MEDAE SUMMARY]"
    )

    print(
        "=" * 100
    )

    print(
        summary[
            [
                "true_label",
                "pulse_ms",
                "n",
                "soc_medae",
                "soh_medae",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print(
        "\n[DONE] Supplementary Figure 7 generated."
    )

    print(
        f"[SAVED] Figure: {trend_png}"
    )

    print(
        f"[SAVED] Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()
