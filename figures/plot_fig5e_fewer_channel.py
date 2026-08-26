# -*- coding: utf-8 -*-
"""
plot_fig5e_fewer_channel.py

Figure 5e:
Effect of input-channel composition on model performance.

Inputs:
    Channel-ablation cases (ch1_only, ch12, ch13):
        results/ablation/channel_ablation/channel_ablation_summary.csv

    Full proposed-framework case:
        results/proposed_framework/further_analysis/tables/proposed_method_summary.csv

Metrics used:
    Channel-ablation cases:
        test_material_acc
        test_soc_medae_raw
        test_soh_medae_raw

    Full proposed-framework test case:
        material_acc
        soc_medae
        soh_medae

Output:
    results/figures/main/fig5e/fig5e_fewer_channel.png
    results/figures/main/fig5e/fig5e_fewer_channel_pure.png
    results/figures/main/fig5e/fig5e_plot_data.csv
"""

from __future__ import annotations

VERSION = "FULL_FROM_FURTHER_ANALYSIS_V3"

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "ablation"
    / "channel_ablation"
    / "channel_ablation_summary.csv"
)

PROPOSED_SUMMARY_CSV = (
    PROJECT_ROOT
    / "results"
    / "proposed_framework"
    / "further_analysis"
    / "tables"
    / "proposed_method_summary.csv"
)

SAVE_DIR = PROJECT_ROOT / "results" / "figures" / "main" / "fig5e"

FULL_SAVE_NAME = "fig5e_fewer_channel.png"
PURE_SAVE_NAME = "fig5e_fewer_channel_pure.png"
PLOT_DATA_NAME = "fig5e_plot_data.csv"

SAVE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Case configuration
# =============================================================================
CASE_ORDER = [
    "ch1_only",
    "ch12",
    "ch13",
    "full",
]

CASE_LABELS = {
    "ch1_only": "Ch1\nRaw",
    "ch12": "Ch1+Ch2\nRaw+ΔU",
    "ch13": "Ch1+Ch3\nRaw+OCV",
    "full": "Full\nRaw+ΔU+OCV",
}


# =============================================================================
# Load data
# =============================================================================
def load_dataframe() -> pd.DataFrame:
    """
    Build Figure 5e plotting data from two sources.

    ch1_only / ch12 / ch13:
        channel_ablation_summary.csv

    full:
        proposed_method_summary.csv

    Channel-ablation metrics are read strictly from:
        test_material_acc
        test_soc_medae_raw
        test_soh_medae_raw

    The full proposed-framework result is read from the row where:
        split == "test"
    using only:
        material_acc
        soc_medae
        soh_medae
    """

    # -------------------------------------------------------------------------
    # 1. Channel-ablation cases: ch1_only / ch12 / ch13
    # -------------------------------------------------------------------------
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Channel-ablation summary not found:\n{INPUT_CSV}"
        )

    raw = pd.read_csv(INPUT_CSV)

    required_cols = [
        "test_material_acc",
        "test_soc_medae_raw",
        "test_soh_medae_raw",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in raw.columns
    ]

    if missing_cols:
        raise RuntimeError(
            "Channel-ablation summary is missing required updated metrics:\n"
            f"{missing_cols}\n"
            f"Available columns: {raw.columns.tolist()}"
        )

    if "config" in raw.columns:
        case_col = "config"
    elif "channel_mode" in raw.columns:
        case_col = "channel_mode"
    else:
        raise RuntimeError(
            "Cannot find channel case column. Expected 'config' or 'channel_mode'.\n"
            f"Available columns: {raw.columns.tolist()}"
        )

    ablation_cases = [
        "ch1_only",
        "ch12",
        "ch13",
    ]

    ablation_df = pd.DataFrame(
        {
            "case": raw[case_col].astype(str).str.strip(),
            "material_accuracy": pd.to_numeric(
                raw["test_material_acc"],
                errors="raise",
            ),
            "soc_medae": pd.to_numeric(
                raw["test_soc_medae_raw"],
                errors="raise",
            ),
            "soh_medae": pd.to_numeric(
                raw["test_soh_medae_raw"],
                errors="raise",
            ),
        }
    )

    ablation_df = ablation_df[
        ablation_df["case"].isin(ablation_cases)
    ].copy()

    missing_cases = [
        case for case in ablation_cases
        if case not in set(ablation_df["case"])
    ]

    if missing_cases:
        raise RuntimeError(
            f"Missing channel-ablation cases: {missing_cases}\n"
            f"Available cases: {sorted(raw[case_col].astype(str).unique().tolist())}"
        )

    duplicate_cases = (
        ablation_df["case"]
        .value_counts()
        .loc[lambda x: x > 1]
    )

    if not duplicate_cases.empty:
        raise RuntimeError(
            "Duplicate rows found for channel-ablation cases:\n"
            f"{duplicate_cases.to_string()}"
        )

    ablation_df = (
        ablation_df
        .set_index("case")
        .loc[ablation_cases]
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # 2. Full case: proposed_method_summary.csv
    # -------------------------------------------------------------------------
    if not PROPOSED_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Proposed-method summary not found:\n{PROPOSED_SUMMARY_CSV}"
        )

    proposed_raw = pd.read_csv(PROPOSED_SUMMARY_CSV)

    if proposed_raw.empty:
        raise RuntimeError(
            f"Proposed-method summary is empty: {PROPOSED_SUMMARY_CSV}"
        )

    proposed_required_cols = [
        "split",
        "material_acc",
        "soc_medae",
        "soh_medae",
    ]

    missing_proposed_cols = [
        col for col in proposed_required_cols
        if col not in proposed_raw.columns
    ]

    if missing_proposed_cols:
        raise RuntimeError(
            "Proposed-method summary is missing required columns:\n"
            f"{missing_proposed_cols}\n"
            f"Available columns: {proposed_raw.columns.tolist()}"
        )

    test_rows = proposed_raw[
        proposed_raw["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "test"
    ].copy()

    if test_rows.empty:
        raise RuntimeError(
            "Could not find split='test' in proposed_method_summary.csv.\n"
            f"Available split values: {proposed_raw['split'].astype(str).tolist()}"
        )

    if len(test_rows) > 1:
        raise RuntimeError(
            "More than one split='test' row found in proposed_method_summary.csv.\n"
            "Expected exactly one test row."
        )

    proposed_row = test_rows.iloc[0]

    full_df = pd.DataFrame(
        [
            {
                "case": "full",
                "material_accuracy": float(
                    proposed_row["material_acc"]
                ),
                "soc_medae": float(
                    proposed_row["soc_medae"]
                ),
                "soh_medae": float(
                    proposed_row["soh_medae"]
                ),
            }
        ]
    )

    # -------------------------------------------------------------------------
    # 3. Combine and enforce Figure 5e order
    # -------------------------------------------------------------------------
    df = pd.concat(
        [ablation_df, full_df],
        ignore_index=True,
    )

    df = (
        df
        .set_index("case")
        .loc[CASE_ORDER]
        .reset_index()
    )

    # test_material_acc is stored as a fraction.
    df["material_accuracy_pct"] = (
        df["material_accuracy"] * 100.0
    )

    # MedAE values are already in percentage-point units.
    df["soc_medae_pct"] = df["soc_medae"]
    df["soh_medae_pct"] = df["soh_medae"]

    df["case_label"] = df["case"].map(CASE_LABELS)

    print("\n[FIGURE 5e SOURCES]")
    print(f"[READ] Ablation cases <- {INPUT_CSV}")
    print(f"[READ] Full case     <- {PROPOSED_SUMMARY_CSV}")
    print("[METRIC] Material accuracy <- test_material_acc")
    print("[METRIC] SOC MedAE         <- test_soc_medae_raw")
    print("[METRIC] SOH MedAE         <- test_soh_medae_raw")

    print("\n[FULL CASE FROM PROPOSED METHOD SUMMARY]")
    print(
        f"Material accuracy = {full_df.iloc[0]['material_accuracy'] * 100.0:.3f}% | "
        f"SOC MedAE = {full_df.iloc[0]['soc_medae']:.4f}% | "
        f"SOH MedAE = {full_df.iloc[0]['soh_medae']:.4f}%"
    )

    print("\n[FINAL DATA SENT TO MATPLOTLIB]")
    print(
        df[
            [
                "case",
                "material_accuracy_pct",
                "soc_medae_pct",
                "soh_medae_pct",
            ]
        ].to_string(index=False)
    )

    plot_data_path = SAVE_DIR / PLOT_DATA_NAME
    df.to_csv(
        plot_data_path,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"[SAVE] Plot data: {plot_data_path}")

    return df


# =============================================================================
# Plot configuration
# =============================================================================
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    }
)


# =============================================================================
# Plot
# =============================================================================
def plot_figure(
    df: pd.DataFrame,
    pure: bool = False,
) -> None:

    # =========================================================================
    # Layout
    # =========================================================================
    spacing = 0.55
    x = np.arange(len(df)) * spacing

    bar_width = 0.18

    # =========================================================================
    # Colors
    # =========================================================================
    soc_color = "#4B5B71"
    soh_color = "#A3BACB"
    acc_color = "#F08C52"

    # =========================================================================
    # Figure
    # =========================================================================
    fig, ax1 = plt.subplots(
        figsize=(
            14 * 9.11 / 14.5 / 2,
            6 / 2.1,
        ),
        dpi=600,
    )

    # =========================================================================
    # Bars: SOC / SOH MedAE
    # =========================================================================
    ax1.bar(
        x - bar_width / 2,
        df["soc_medae_pct"],
        width=bar_width,
        color=soc_color,
        label="SOC MedAE",
    )

    ax1.bar(
        x + bar_width / 2,
        df["soh_medae_pct"],
        width=bar_width,
        color=soh_color,
        label="SOH MedAE",
    )

    # =========================================================================
    # Material accuracy line
    # =========================================================================
    ax2 = ax1.twinx()

    ax2.plot(
        x,
        df["material_accuracy_pct"],
        color=acc_color,
        marker="o",
        linewidth=5,
        markersize=15,
        label="Material accuracy",
        zorder=10,
    )

    # =========================================================================
    # Material-accuracy annotations
    # =========================================================================
    if not pure:
        for xi, yi in zip(
            x,
            df["material_accuracy_pct"],
        ):
            ax2.text(
                xi,
                yi + 0.4,
                f"{yi:.1f}",
                ha="center",
                fontsize=8,
                color=acc_color,
            )

    # =========================================================================
    # Axis
    # =========================================================================
    ax1.set_xticks(x)

    ax1.set_xticklabels(
        df["case_label"],
        fontsize=9,
    )

    ax1.set_ylabel(
        "MedAE (%)",
        fontsize=10,
    )

    ax2.set_ylabel(
        "Material accuracy (%)",
        fontsize=10,
    )

    error_max = max(
        float(df["soc_medae_pct"].max()),
        float(df["soh_medae_pct"].max()),
    )

    ax1.set_ylim(
        0,
        error_max * 1.25 if error_max > 0 else 1.0,
    )

    acc_min = float(df["material_accuracy_pct"].min())
    acc_max = float(df["material_accuracy_pct"].max())

    ax2.set_ylim(
        max(0.0, acc_min - 5.0),
        min(100.0, acc_max + 3.0),
    )

    # =========================================================================
    # Grid
    # =========================================================================
    ax1.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.25,
    )

    # =========================================================================
    # Default spine settings
    # =========================================================================
    for spine in [
        "top",
        "right",
    ]:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    # =========================================================================
    # Full version
    # =========================================================================
    if not pure:
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=3,
            frameon=False,
            fontsize=9,
        )

        ax1.set_title(
            "Effect of channel composition on model performance",
            fontsize=11,
            pad=22,
        )

    # =========================================================================
    # Pure version
    # =========================================================================
    if pure:
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])

        ax1.tick_params(length=0)
        ax2.tick_params(length=0)

        ax1.grid(False)

        for spine in ax1.spines.values():
            spine.set_visible(False)

        for spine in ax2.spines.values():
            spine.set_visible(False)

    # =========================================================================
    # Layout
    # =========================================================================
    plt.tight_layout()

    # =========================================================================
    # Save
    # =========================================================================
    if pure:
        save_name = PURE_SAVE_NAME
    else:
        save_name = FULL_SAVE_NAME

    save_path = SAVE_DIR / save_name

    fig.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
        transparent=pure,
    )

    plt.close(fig)

    print(f"[OK] Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    print(f"[VERSION] {VERSION}")
    df = load_dataframe()

    plot_figure(
        df=df,
        pure=False,
    )

    plot_figure(
        df=df,
        pure=True,
    )


if __name__ == "__main__":
    main()
