# -*- coding: utf-8 -*-
"""
suppfig4_u_drop_structure_heatmap.py

Supplementary Figure 4:
Real-data heatmap schematic for the 3-channel structured input before and after
dropping one voltage feature and applying neighbor interpolation.

This script shows:
1. Original 3-channel structured input
2. Input after dropping U_i and imputing it
3. The affected entries in the raw-voltage and differential-voltage channels

Input:
    data/

Output:
    results/figures/supp/suppfig4

Generated file:
    suppfig4_u_drop_structure_realdata_heatmap.png
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle


# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path.cwd().resolve()

DATA_ROOT = PROJECT_ROOT / "data"

SAVE_DIR = PROJECT_ROOT / "results" / "figures" / "supp" / "suppfig4"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PNG = SAVE_DIR / "suppfig4_u_drop_structure_realdata_heatmap.png"


# ============================================================
# Data and figure config
# ============================================================
SOC = 50
PULSE_MS = 100

U_START = 1
U_END = 41

DROP_U_INDEX = 19

N_ROW = 5
N_COL = 8

CMAP = "coolwarm"

RAW_HIGHLIGHT = [(2, 1)]
DELTA_HIGHLIGHT = [(2, 1), (2, 2)]

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
# Data helpers
# ============================================================
def group_to_folder(group: str) -> str:
    material, capacity = group.split("_")
    return f"{capacity} {material}"


def group_to_excel_name(group: str, pulse_ms: int) -> str:
    return f"{group}_W_{pulse_ms}.xlsx"


def find_u_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        [f"U{i}" for i in range(U_START, U_END + 1)],
        [f"U_{i}" for i in range(U_START, U_END + 1)],
        [f"u{i}" for i in range(U_START, U_END + 1)],
        [f"u_{i}" for i in range(U_START, U_END + 1)],
    ]

    for cols in candidates:
        if all(c in df.columns for c in cols):
            return cols

    raise ValueError(
        "Cannot find U1-U41 feature columns. "
        f"Available columns: {list(df.columns)}"
    )


def read_one_real_u41_from_excel() -> tuple[np.ndarray, str]:
    sheet_name = f"SOC{SOC}"

    attempted_files = []

    for group in GROUP_ORDER:
        folder = group_to_folder(group)
        file_name = group_to_excel_name(group, PULSE_MS)
        file_path = DATA_ROOT / folder / file_name

        if not file_path.exists():
            attempted_files.append(str(file_path))
            continue

        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            u_cols = find_u_columns(df)

            if len(df) == 0:
                continue

            u = df[u_cols].iloc[0].to_numpy(dtype=float)

            if u.shape[0] != 41:
                raise ValueError(f"Expected 41 U values, got {u.shape[0]}")

            if not np.isfinite(u).all():
                valid = df[u_cols].replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid) == 0:
                    continue
                u = valid.iloc[0].to_numpy(dtype=float)

            return u, group

        except Exception:
            continue

    raise RuntimeError(
        "No valid U1-U41 sample was loaded. "
        f"Please check DATA_ROOT, SOC={SOC}, and PULSE_MS={PULSE_MS}."
    )


def build_3ch_from_u41(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=float)

    if u.shape[0] != 41:
        raise ValueError(f"Expected 41 U values, got {u.shape[0]}")

    u1 = float(u[0])
    u2_41 = u[1:].astype(float)

    raw_channel = u2_41.reshape(N_ROW, N_COL)

    delta_u = np.empty(40, dtype=float)
    delta_u[0] = u[1] - u[0]
    delta_u[1:] = u[2:] - u[1:-1]
    delta_channel = delta_u.reshape(N_ROW, N_COL)

    ocv_channel = np.full((N_ROW, N_COL), u1, dtype=float)

    return raw_channel, delta_channel, ocv_channel


def impute_after_drop(u: np.ndarray, drop_u_index: int) -> np.ndarray:
    u_imputed = np.asarray(u, dtype=float).copy()
    idx = drop_u_index - 1

    if idx < 0 or idx >= len(u_imputed):
        raise ValueError(f"drop_u_index out of range: {drop_u_index}")

    if idx == 0:
        u_imputed[idx] = u_imputed[idx + 1]
    elif idx == len(u_imputed) - 1:
        u_imputed[idx] = u_imputed[idx - 1]
    else:
        u_imputed[idx] = 0.5 * (u_imputed[idx - 1] + u_imputed[idx + 1])

    return u_imputed


# ============================================================
# Plot helpers
# ============================================================
def add_dashed_boxes(ax, positions, color: str = "#D62728", lw: float = 2.2):
    for row, col in positions:
        rect = Rectangle(
            (col - 0.5, row - 0.5),
            1,
            1,
            fill=False,
            edgecolor=color,
            linewidth=lw,
            linestyle="--",
        )
        ax.add_patch(rect)


def draw_heatmap(
    ax,
    data: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    affected_positions=None,
):
    image = ax.imshow(
        data,
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_title(title, fontsize=10, pad=6)

    ax.set_xticks(range(N_COL))
    ax.set_yticks(range(N_ROW))
    ax.set_xticklabels(range(1, N_COL + 1), fontsize=8)
    ax.set_yticklabels(range(1, N_ROW + 1), fontsize=8)
    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(-0.5, N_COL, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N_ROW, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    if affected_positions:
        add_dashed_boxes(ax, affected_positions)

    return image


def compute_channel_limits(original_channel: np.ndarray, imputed_channel: np.ndarray):
    vmin = min(np.nanmin(original_channel), np.nanmin(imputed_channel))
    vmax = max(np.nanmax(original_channel), np.nanmax(imputed_channel))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Non-finite channel limits detected.")

    if np.isclose(vmin, vmax):
        vmin -= 1e-6
        vmax += 1e-6

    return vmin, vmax


def make_figure(
    original_raw: np.ndarray,
    original_delta: np.ndarray,
    original_ocv: np.ndarray,
    imputed_raw: np.ndarray,
    imputed_delta: np.ndarray,
    imputed_ocv: np.ndarray,
    material_label: str,
):
    raw_vmin, raw_vmax = compute_channel_limits(original_raw, imputed_raw)
    delta_vmin, delta_vmax = compute_channel_limits(original_delta, imputed_delta)
    ocv_vmin, ocv_vmax = compute_channel_limits(original_ocv, imputed_ocv)

    fig = plt.figure(figsize=(8.8, 6.8))

    grid_spec = fig.add_gridspec(
        nrows=6,
        ncols=3,
        height_ratios=[1.0, 0.20, 0.18, 1.0, 0.12, 0.22],
        hspace=0.42,
        wspace=0.34,
    )

    ax_top1 = fig.add_subplot(grid_spec[0, 0])
    ax_top2 = fig.add_subplot(grid_spec[0, 1])
    ax_top3 = fig.add_subplot(grid_spec[0, 2])

    im_raw_top = draw_heatmap(
        ax=ax_top1,
        data=original_raw,
        title="Channel 1: raw voltage\n$U_2$-$U_{41}$",
        vmin=raw_vmin,
        vmax=raw_vmax,
    )

    im_delta_top = draw_heatmap(
        ax=ax_top2,
        data=original_delta,
        title="Channel 2: differential voltage\n$\\Delta U$",
        vmin=delta_vmin,
        vmax=delta_vmax,
    )

    im_ocv_top = draw_heatmap(
        ax=ax_top3,
        data=original_ocv,
        title="Channel 3: OCV baseline\n$U_1$",
        vmin=ocv_vmin,
        vmax=ocv_vmax,
    )

    fig.text(
        0.5,
        0.965,
        f"3-structured input (original real sample, {material_label})",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    ax_arrow = fig.add_subplot(grid_spec[1:3, :])
    ax_arrow.axis("off")

    arrow = FancyArrowPatch(
        (0.5, 0.85),
        (0.5, 0.15),
        transform=ax_arrow.transAxes,
        arrowstyle="simple",
        mutation_scale=35,
        linewidth=0,
        color="#4D4D4D",
    )
    ax_arrow.add_patch(arrow)

    ax_arrow.text(
        0.56,
        0.50,
        f"Drop $U_{{{DROP_U_INDEX}}}$",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    ax_bot1 = fig.add_subplot(grid_spec[3, 0])
    ax_bot2 = fig.add_subplot(grid_spec[3, 1])
    ax_bot3 = fig.add_subplot(grid_spec[3, 2])

    draw_heatmap(
        ax=ax_bot1,
        data=imputed_raw,
        title="After imputation\n1 affected entry",
        vmin=raw_vmin,
        vmax=raw_vmax,
        affected_positions=RAW_HIGHLIGHT,
    )

    draw_heatmap(
        ax=ax_bot2,
        data=imputed_delta,
        title="After imputation\n2 affected entries",
        vmin=delta_vmin,
        vmax=delta_vmax,
        affected_positions=DELTA_HIGHLIGHT,
    )

    draw_heatmap(
        ax=ax_bot3,
        data=imputed_ocv,
        title="After imputation\nunchanged",
        vmin=ocv_vmin,
        vmax=ocv_vmax,
        affected_positions=[],
    )

    cax1 = fig.add_subplot(grid_spec[4, 0])
    cax2 = fig.add_subplot(grid_spec[4, 1])
    cax3 = fig.add_subplot(grid_spec[4, 2])

    cbar1 = fig.colorbar(im_raw_top, cax=cax1, orientation="horizontal")
    cbar1.set_label("Raw voltage", fontsize=8.5, labelpad=2)
    cbar1.ax.tick_params(labelsize=7.5, length=2)
    cbar1.outline.set_linewidth(0.6)

    cbar2 = fig.colorbar(im_delta_top, cax=cax2, orientation="horizontal")
    cbar2.set_label("Differential voltage", fontsize=8.5, labelpad=2)
    cbar2.ax.tick_params(labelsize=7.5, length=2)
    cbar2.outline.set_linewidth(0.6)

    cbar3 = fig.colorbar(im_ocv_top, cax=cax3, orientation="horizontal")
    cbar3.set_label("OCV baseline", fontsize=8.5, labelpad=2)
    cbar3.ax.tick_params(labelsize=7.5, length=2)
    cbar3.outline.set_linewidth(0.6)

    legend_patch = Patch(
        facecolor="none",
        edgecolor="#D62728",
        linewidth=2.0,
        linestyle="--",
        label="Affected entry after dropping $U_i$",
    )

    fig.legend(
        handles=[legend_patch],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        frameon=False,
        fontsize=9,
        ncol=1,
    )

    fig.text(
        0.5,
        0.02,
        "Color maps voltage magnitude within each channel; blue indicates lower values and red indicates higher values.",
        ha="center",
        va="center",
        fontsize=8.8,
    )

    return fig


# ============================================================
# Main
# ============================================================
def main():
    print("[INFO] Generating Supplementary Figure 4...")

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.linewidth": 1.0,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    u_original, material_label = read_one_real_u41_from_excel()
    u_imputed = impute_after_drop(u_original, DROP_U_INDEX)

    original_raw, original_delta, original_ocv = build_3ch_from_u41(u_original)
    imputed_raw, imputed_delta, imputed_ocv = build_3ch_from_u41(u_imputed)

    fig = make_figure(
        original_raw=original_raw,
        original_delta=original_delta,
        original_ocv=original_ocv,
        imputed_raw=imputed_raw,
        imputed_delta=imputed_delta,
        imputed_ocv=imputed_ocv,
        material_label=material_label,
    )

    fig.savefig(SAVE_PNG, bbox_inches="tight", dpi=600)
    plt.close(fig)

    print("[DONE] Supplementary Figure 4 generated.")
    print(f"[SAVED] Figure: {SAVE_PNG}")


if __name__ == "__main__":
    main()