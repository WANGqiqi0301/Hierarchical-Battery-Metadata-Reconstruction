# -*- coding: utf-8 -*-
"""
fig5a_input_representation.py

Figure 5a:
Comparison between raw 1D and structured 3-channel input representations.

Input:
    results/ablation/input_representation/
    input_representation_ablation_summary.csv

Output:
    results/figures/main/fig5a/fig5a_input_representation.png
    results/figures/main/fig5a/fig5a_input_representation_pure.png
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# =============================================================================
# 0. Paths
# =============================================================================

INPUT_CSV = os.path.join(
    "results",
    "ablation",
    "input_representation",
    "input_representation_ablation_summary.csv",
)

SAVE_DIR = os.path.join(
    "results",
    "figures",
    "main",
    "fig5a",
)

SAVE_NAME = "fig5a_input_representation.png"
PURE_SAVE_NAME = "fig5a_input_representation_pure.png"

os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# 1. Plot settings
# =============================================================================

COLORS = [
    "#5B7B8E",   # Raw 1D
    "#A2B59F",   # Structured 3-channel
]

REPRESENTATION_ORDER = [
    "raw_1d",
    "structured_3ch",
]

DISPLAY_NAMES = {
    "raw_1d": "Raw 1D",
    "structured_3ch": "Structured 3-channel",
}

Y_LABELS = [
    "Material Acc (%)",
    "SOC MedAE (%)",
    "SOH MedAE (%)",
]


# =============================================================================
# 2. Helpers
# =============================================================================

def find_first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    metric_name: str,
) -> str:
    """
    Return the first available column from candidates.
    """

    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"Cannot find column for '{metric_name}'.\n"
        f"Tried: {candidates}\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


def load_input_representation_results(
    csv_path: str,
) -> pd.DataFrame:
    """
    Load and standardize input-representation ablation results.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Cannot find input-representation ablation summary:\n"
            f"  {csv_path}\n\n"
            "Please run:\n"
            "  python ablation/input_representation_ablation.py --config both"
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise RuntimeError(
            f"Input CSV is empty: {csv_path}"
        )

    # -------------------------------------------------------------------------
    # Representation column
    # -------------------------------------------------------------------------

    representation_col = find_first_existing_column(
        df,
        candidates=[
            "input_representation",
            "config",
        ],
        metric_name="input representation",
    )

    # -------------------------------------------------------------------------
    # Metric columns
    # -------------------------------------------------------------------------

    cls_col = find_first_existing_column(
        df,
        candidates=[
            "material_acc_pct",
            "test_material_acc",
            "material_acc",
            "mat_acc_pct",
        ],
        metric_name="material accuracy",
    )

    soc_col = find_first_existing_column(
        df,
        candidates=[
            "soc_medae_pct",
            "test_soc_medae_raw",
            "test_soc_medae",
            "soc_medae_raw",
            "soc_medae",
        ],
        metric_name="SOC MedAE",
    )

    soh_col = find_first_existing_column(
        df,
        candidates=[
            "soh_medae_pct",
            "test_soh_medae_raw",
            "test_soh_medae",
            "soh_medae_raw",
            "soh_medae",
        ],
        metric_name="SOH MedAE",
    )

    out = pd.DataFrame(
        {
            "input_representation": (
                df[representation_col]
                .astype(str)
                .str.strip()
                .str.lower()
            ),
            "material_acc": pd.to_numeric(
                df[cls_col],
                errors="coerce",
            ),
            "soc_medae": pd.to_numeric(
                df[soc_col],
                errors="coerce",
            ),
            "soh_medae": pd.to_numeric(
                df[soh_col],
                errors="coerce",
            ),
        }
    )

    # -------------------------------------------------------------------------
    # Convert accuracy from 0-1 to percentage if necessary
    # -------------------------------------------------------------------------

    if cls_col in {"test_material_acc", "material_acc"}:
        valid_acc = out["material_acc"].dropna()

        if (
            not valid_acc.empty
            and valid_acc.max() <= 1.0 + 1e-8
        ):
            out["material_acc"] = (
                out["material_acc"] * 100.0
            )

    # -------------------------------------------------------------------------
    # Remove invalid rows
    # -------------------------------------------------------------------------

    out = out.dropna(
        subset=[
            "material_acc",
            "soc_medae",
            "soh_medae",
        ]
    ).copy()

    if out.empty:
        raise RuntimeError(
            "No valid metric rows remain after numeric conversion."
        )

    # -------------------------------------------------------------------------
    # Keep requested representations only
    # -------------------------------------------------------------------------

    out = out[
        out["input_representation"].isin(
            REPRESENTATION_ORDER
        )
    ].copy()

    missing_cases = [
        case
        for case in REPRESENTATION_ORDER
        if case not in set(out["input_representation"])
    ]

    if missing_cases:
        raise RuntimeError(
            "Missing required input-representation cases:\n"
            f"  {missing_cases}\n\n"
            "Available cases:\n"
            f"  {sorted(df[representation_col].astype(str).unique().tolist())}"
        )

    # -------------------------------------------------------------------------
    # Check duplicate rows
    # -------------------------------------------------------------------------

    duplicate_cases = (
        out["input_representation"]
        .value_counts()
        .loc[lambda x: x > 1]
    )

    if not duplicate_cases.empty:
        raise RuntimeError(
            "Duplicate rows found for input representations:\n"
            f"{duplicate_cases.to_string()}\n\n"
            "Please keep one final result per representation."
        )

    # -------------------------------------------------------------------------
    # Enforce plotting order
    # -------------------------------------------------------------------------

    out = (
        out
        .set_index("input_representation")
        .loc[REPRESENTATION_ORDER]
        .reset_index()
    )

    print(f"[READ] {csv_path}")
    print(out.to_string(index=False))

    return out


# =============================================================================
# 3. Main labeled figure
# =============================================================================

def generate_labeled_version(
    df: pd.DataFrame,
) -> None:
    """
    Generate labeled Fig. 5a.
    """

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.linewidth"] = 0.8

    data_list = [
        df["material_acc"].to_numpy(dtype=float),
        df["soc_medae"].to_numpy(dtype=float),
        df["soh_medae"].to_numpy(dtype=float),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(4, 1.9),
        dpi=300,
    )

    bar_width = 0.25

    positions = [
        -0.2,
        0.2,
    ]

    for i, ax in enumerate(axes):
        current_data = data_list[i]

        # ---------------------------------------------------------------------
        # Bars
        # ---------------------------------------------------------------------

        for j, value in enumerate(current_data):
            ax.bar(
                positions[j],
                value,
                width=bar_width,
                color=COLORS[j],
                zorder=3,
            )

        max_val = float(np.max(current_data))

        ax.set_ylim(
            0,
            max_val * 1.2,
        )

        # ---------------------------------------------------------------------
        # Value labels
        # ---------------------------------------------------------------------

        for j, value in enumerate(current_data):

            if i == 0:
                label_text = f"{value:.1f}"
            else:
                label_text = f"{value:.2f}"

            ax.text(
                positions[j],
                value + max_val * 0.02,
                label_text,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
                zorder=4,
            )

        # ---------------------------------------------------------------------
        # Axis formatting
        # ---------------------------------------------------------------------

        ax.set_xlim(
            -0.6,
            0.6,
        )

        ax.set_xticks([])

        ax.tick_params(
            axis="y",
            direction="in",
            length=2,
            labelsize=9,
        )

        ax.set_ylabel(
            Y_LABELS[i],
            fontsize=9,
            labelpad=2,
        )

        # ---------------------------------------------------------------------
        # Divider
        # ---------------------------------------------------------------------

        ax.axvline(
            x=0.0,
            color="gray",
            linestyle="--",
            linewidth=0.5,
            alpha=0.4,
            zorder=1,
        )

    # =========================================================================
    # Legend
    # =========================================================================

    patches = [
        mpatches.Patch(
            color=COLORS[0],
            label=DISPLAY_NAMES["raw_1d"],
        ),
        mpatches.Patch(
            color=COLORS[1],
            label=DISPLAY_NAMES["structured_3ch"],
        ),
    ]

    fig.legend(
        handles=patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        fontsize=9,
        frameon=False,
        columnspacing=1.5,
    )

    plt.tight_layout(
        w_pad=0.8,
        rect=[
            0,
            0.08,
            1,
            1,
        ],
    )

    # =========================================================================
    # Save
    # =========================================================================

    save_path = os.path.join(
        SAVE_DIR,
        SAVE_NAME,
    )

    plt.savefig(
        save_path,
        format="png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close()

    print(f"[SAVE] {save_path}")


# =============================================================================
# 4. Pure figure
# =============================================================================

def generate_pure_version(
    df: pd.DataFrame,
) -> None:
    """
    Generate pure Fig. 5a without text, ticks, labels, legend, or numbers.
    """

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.linewidth"] = 0.8

    data_list = [
        df["material_acc"].to_numpy(dtype=float),
        df["soc_medae"].to_numpy(dtype=float),
        df["soh_medae"].to_numpy(dtype=float),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(4, 1.9),
        dpi=300,
    )

    bar_width = 0.25

    positions = [
        -0.2,
        0.2,
    ]

    for i, ax in enumerate(axes):
        current_data = data_list[i]

        # ---------------------------------------------------------------------
        # Bars
        # ---------------------------------------------------------------------

        for j, value in enumerate(current_data):
            ax.bar(
                positions[j],
                value,
                width=bar_width,
                color=COLORS[j],
                zorder=3,
            )

        max_val = float(np.max(current_data))

        ax.set_ylim(
            0,
            max_val * 1.2,
        )

        ax.set_xlim(
            -0.6,
            0.6,
        )

        # ---------------------------------------------------------------------
        # Remove labels and numbers
        # ---------------------------------------------------------------------

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlabel("")
        ax.set_ylabel("")

        # ---------------------------------------------------------------------
        # Divider
        # ---------------------------------------------------------------------

        ax.axvline(
            x=0.0,
            color="gray",
            linestyle="--",
            linewidth=0.5,
            alpha=0.4,
            zorder=1,
        )

    plt.tight_layout(
        w_pad=0.8,
        rect=[
            0,
            0,
            1,
            1,
        ],
    )

    # =========================================================================
    # Save
    # =========================================================================

    save_path = os.path.join(
        SAVE_DIR,
        PURE_SAVE_NAME,
    )

    plt.savefig(
        save_path,
        format="png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close()

    print(f"[SAVE] {save_path}")


# =============================================================================
# 5. Main
# =============================================================================

def main() -> None:

    df = load_input_representation_results(
        INPUT_CSV,
    )

    generate_labeled_version(
        df,
    )

    generate_pure_version(
        df,
    )


if __name__ == "__main__":
    main()