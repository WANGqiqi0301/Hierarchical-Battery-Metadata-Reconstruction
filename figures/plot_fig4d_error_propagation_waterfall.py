# -*- coding: utf-8 -*-

"""
plot_fig4d_error_propagation_waterfall.py

Figure 4d:
Counterfactual error-propagation waterfall analysis for SOH MedAE.

Input:
results/analysis/error_propagation/counterfactual_summary.csv

Output:
results/figures/main/fig4d/fig4d_error_propagation_waterfall.png
results/figures/main/fig4d/fig4d_error_propagation_waterfall_pure.png
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# 1. Paths
# =============================================================================

INPUT_CSV = os.path.join(
    "results",
    "analysis",
    "error_propagation",
    "counterfactual_summary.csv",
)

SAVE_DIR = os.path.join(
    "results",
    "figures",
    "main",
    "fig4d",
)

SAVE_NAME = "fig4d_error_propagation_waterfall.png"
PURE_SAVE_NAME = "fig4d_error_propagation_waterfall_pure.png"

os.makedirs(
    SAVE_DIR,
    exist_ok=True,
)


# =============================================================================
# 2. Load data
# =============================================================================

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"Input CSV not found: {INPUT_CSV}\n"
        "Please run the error-propagation analysis first."
    )

df = pd.read_csv(INPUT_CSV)


# =============================================================================
# 3. Validate columns
# =============================================================================

required_cols = [
    "experiment",
    "soh_medae",
]

missing_cols = [
    col
    for col in required_cols
    if col not in df.columns
]

if missing_cols:
    raise RuntimeError(
        f"Missing required columns in input CSV: {missing_cols}\n"
        "This figure now requires the SOH MedAE field 'soh_medae'."
    )


# =============================================================================
# 4. Prepare E0-E3 values
# =============================================================================

exp_list = [
    "E0",
    "E1",
    "E2",
    "E3",
]

data = df.set_index(
    "experiment"
)["soh_medae"]

missing_exp = [
    exp
    for exp in exp_list
    if exp not in data.index
]

if missing_exp:
    raise RuntimeError(
        f"Missing required experiments: {missing_exp}"
    )

vals = [
    float(data.loc[exp])
    for exp in exp_list
]

e0 = vals[0]
e1 = vals[1]
e2 = vals[2]
e3 = vals[3]


# =============================================================================
# 5. Prepare waterfall values
# =============================================================================

labels = [
    "E0",
    "E1-E0",
    "E2-E1",
    "E3-E2",
    "E3",
]

deltas = [
    e0,
    e1 - e0,
    e2 - e1,
    e3 - e2,
    e3,
]

bottoms = [
    0.0,
    e0,
    e1,
    e2,
    0.0,
]


# =============================================================================
# 6. Print source values
# =============================================================================

print("\n[SOH MedAE values]")

for exp, value in zip(
    exp_list,
    vals,
):
    print(
        f"{exp}: {value:.6f}%"
    )

print("\n[Waterfall increments]")

print(
    f"E1-E0: {e1 - e0:+.6f}"
)

print(
    f"E2-E1: {e2 - e1:+.6f}"
)

print(
    f"E3-E2: {e3 - e2:+.6f}"
)


# =============================================================================
# 7. Plot configuration
# =============================================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2
plt.rcParams["lines.linewidth"] = 1.5

c_base = "#A67C8E"
c_delta = "#C2A3B0"
c_total = "#5F7D8E"

colors = [
    c_base,
    c_delta,
    c_delta,
    c_delta,
    c_total,
]

axis_color = "#2F3E46"

bar_width = 0.9


# =============================================================================
# 8. Core drawing function
# =============================================================================

def draw_waterfall(
    ax: plt.Axes,
) -> None:
    bars = ax.bar(
        labels,
        deltas,
        bottom=bottoms,
        color=colors,
        edgecolor="none",
        width=bar_width,
    )

    # Connector lines
    for i in range(
        len(deltas) - 1
    ):
        step_y = (
            bottoms[i]
            + deltas[i]
        )

        ax.plot(
            [
                i + bar_width / 2,
                i + 1 - bar_width / 2,
            ],
            [
                step_y,
                step_y,
            ],
            color=axis_color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.6,
        )

    return bars


# =============================================================================
# 9. Academic version
# =============================================================================

def save_academic_version() -> None:
    fig, ax = plt.subplots(
        figsize=(4.2, 3.2)
    )

    bars = draw_waterfall(
        ax=ax,
    )

    # -------------------------------------------------------------------------
    # Axis
    # -------------------------------------------------------------------------
    ax.set_ylabel(
        "SOH MedAE (%)",
        fontsize=9,
        fontweight="bold",
    )

    ax.tick_params(
        axis="x",
        labelsize=8,
    )

    ax.tick_params(
        axis="y",
        labelsize=8,
    )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    ax.spines["left"].set_color(
        "#D1D1D1"
    )

    ax.spines["bottom"].set_color(
        "#D1D1D1"
    )

    # -------------------------------------------------------------------------
    # Value labels
    # -------------------------------------------------------------------------
    y_range = max(vals) - min(
        0.0,
        min(vals),
    )

    text_offset = max(
        0.03,
        y_range * 0.02,
    )

    for i, bar in enumerate(
        bars
    ):
        actual_val = deltas[i]

        if actual_val >= 0:
            y_pos = (
                bottoms[i]
                + actual_val
                + text_offset
            )

            va = "bottom"

        else:
            y_pos = (
                bottoms[i]
                + actual_val
                - text_offset
            )

            va = "top"

        if 0 < i < len(deltas) - 1:
            txt = f"{actual_val:+.2f}"

        else:
            txt = f"{actual_val:.2f}"

        ax.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            y_pos,
            txt,
            ha="center",
            va=va,
            fontsize=8,
            fontweight="bold",
            color=axis_color,
        )

    plt.tight_layout()

    save_path = os.path.join(
        SAVE_DIR,
        SAVE_NAME,
    )

    fig.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[OK] Saved: {save_path}"
    )


# =============================================================================
# 10. Pure version
# =============================================================================

def save_pure_version() -> None:
    fig, ax = plt.subplots(
        figsize=(4.2, 3.2)
    )

    draw_waterfall(
        ax=ax,
    )

    # -------------------------------------------------------------------------
    # Remove all text
    # -------------------------------------------------------------------------
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticks([])
    ax.set_yticks([])

    # -------------------------------------------------------------------------
    # Remove all spines
    # -------------------------------------------------------------------------
    for spine in ax.spines.values():
        spine.set_visible(False)

    # -------------------------------------------------------------------------
    # Remove margins from ticks
    # -------------------------------------------------------------------------
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
    )

    plt.tight_layout(
        pad=0.1
    )

    save_path = os.path.join(
        SAVE_DIR,
        PURE_SAVE_NAME,
    )

    fig.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
    )

    plt.close(fig)

    print(
        f"[OK] Saved: {save_path}"
    )


# =============================================================================
# 11. Main
# =============================================================================

def main() -> None:
    save_academic_version()
    save_pure_version()


if __name__ == "__main__":
    main()
