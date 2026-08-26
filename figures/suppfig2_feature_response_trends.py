# -*- coding: utf-8 -*-
"""
suppfig2_feature_response_trends.py

Supplementary Figure 2:
Voltage-response trend plots for all material-capacity groups.

This script reads raw Excel data directly and generates, for each group:
1. U vs SOC
2. U vs SOH
3. DeltaU vs SOC
4. DeltaU vs SOH

Input:
    data/

Output:
    results/figures/supp/suppfig2

Generated files:
    For each material-capacity group:
        <group>_U_SOC.png
        <group>_U_SOH.png
        <group>_DeltaU_SOC.png
        <group>_DeltaU_SOH.png
        cache_feat_<group>.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats


# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path.cwd().resolve()

DATA_ROOT = PROJECT_ROOT / "data"
SAVE_ROOT = PROJECT_ROOT / "results" / "figures" / "supp" / "suppfig2"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Data config
# ============================================================
SOC_COL = "SOC"
SOH_COL = "SOH"

U_START = 1
U_END = 41

TARGET_MATERIALS = [
    "LFP_35Ah",
    "LFP_68Ah",
    "LMO_10Ah",
    "LMO_24Ah",
    "LMO_25Ah",
    "LMO_26Ah",
    "NMC_15Ah",
    "NMC_21Ah",
]

SELECTED_SOC_LIST = [20, 30, 50, 80, 90]
SELECTED_PULSE_WIDTHS_MS = [100, 500, 1000]

MAX_SAMPLES_PER_GROUP = 5000
RANDOM_SEED = 42


# ============================================================
# Plot style
# ============================================================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})


# ============================================================
# Path helpers
# ============================================================
def group_to_folder(group: str) -> str:
    material, capacity = group.split("_")
    return f"{capacity} {material}"


def group_to_excel_name(group: str, pulse_width_ms: int) -> str:
    return f"{group}_W_{pulse_width_ms}.xlsx"


def get_material_save_dir(group: str) -> Path:
    save_dir = SAVE_ROOT / group
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def get_cache_csv_path(group: str) -> Path:
    return get_material_save_dir(group) / f"cache_feat_{group}.csv"


# ============================================================
# Data helpers
# ============================================================
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


def find_column(df: pd.DataFrame, target: str):
    for col in df.columns:
        if str(col).strip() == target:
            return col

    target_norm = target.lower().replace(" ", "")

    for col in df.columns:
        col_norm = str(col).strip().lower().replace(" ", "")
        if col_norm in [
            target_norm,
            f"{target_norm}(%)",
            f"{target_norm}/%",
            f"{target_norm}_percent",
            f"{target_norm}percent",
        ]:
            return col

    return None


def parse_soc_from_sheet_name(sheet_name: str):
    match = re.match(r"^\s*SOC\s*(\d+(?:\.\d+)?)\s*$", str(sheet_name), re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def normalize_soc_to_percent(values) -> np.ndarray:
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

    finite_values = values[np.isfinite(values)]
    if len(finite_values) > 0 and np.nanmax(finite_values) <= 1.5:
        values = values * 100.0

    return values


def normalize_soh_to_fraction(values) -> np.ndarray:
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

    finite_values = values[np.isfinite(values)]
    if len(finite_values) > 0 and np.nanmax(finite_values) > 1.5:
        values = values / 100.0

    return values


def read_one_sheet_features(
    file_path: Path,
    sheet_name: str,
    group: str,
) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    u_cols = find_u_columns(df)

    soc_col = find_column(df, SOC_COL)
    soh_col = find_column(df, SOH_COL)

    if soh_col is None:
        raise ValueError(
            f"{file_path} | sheet={sheet_name}: missing SOH column. "
            f"Available columns: {list(df.columns)}"
        )

    soc_from_sheet = parse_soc_from_sheet_name(sheet_name)

    out = df[u_cols].copy()
    out.columns = [f"U{i}" for i in range(U_START, U_END + 1)]

    if soc_col is not None:
        out[SOC_COL] = normalize_soc_to_percent(df[soc_col])
    elif soc_from_sheet is not None:
        out[SOC_COL] = float(soc_from_sheet)
    else:
        raise ValueError(
            f"{file_path} | sheet={sheet_name}: missing SOC column and cannot parse SOC from sheet name."
        )

    out[SOH_COL] = normalize_soh_to_fraction(df[soh_col])
    out["material_capacity"] = group
    out["pulse_ms"] = int(re.search(r"_W_(\d+)", file_path.stem).group(1))

    required_cols = [f"U{i}" for i in range(U_START, U_END + 1)] + [SOC_COL, SOH_COL]
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=required_cols).copy()

    return out


def get_material_data(group: str, max_samples: int = MAX_SAMPLES_PER_GROUP):
    cache_csv = get_cache_csv_path(group)

    if cache_csv.exists():
        df = pd.read_csv(cache_csv)
        u_cols = [f"U{i}" for i in range(U_START, U_END + 1)]

        x = df[u_cols].to_numpy(dtype=float)
        soc = df[SOC_COL].to_numpy(dtype=float)
        soh = df[SOH_COL].to_numpy(dtype=float)

        return x, soc, soh

    all_dfs = []
    folder = group_to_folder(group)

    for soc_value in SELECTED_SOC_LIST:
        sheet_name = f"SOC{soc_value}"

        for pulse_width_ms in SELECTED_PULSE_WIDTHS_MS:
            file_name = group_to_excel_name(group, pulse_width_ms)
            file_path = DATA_ROOT / folder / file_name

            if not file_path.exists():
                continue

            try:
                df_one = read_one_sheet_features(
                    file_path=file_path,
                    sheet_name=sheet_name,
                    group=group,
                )
            except Exception:
                continue

            if len(df_one) > 0:
                all_dfs.append(df_one)

    if not all_dfs:
        return None, None, None

    df_all = pd.concat(all_dfs, axis=0, ignore_index=True)
    df_all = df_all.drop_duplicates().reset_index(drop=True)

    if len(df_all) > max_samples:
        df_all = df_all.sample(
            n=max_samples,
            replace=False,
            random_state=RANDOM_SEED,
        ).reset_index(drop=True)

    u_cols = [f"U{i}" for i in range(U_START, U_END + 1)]

    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df_all[u_cols + [SOC_COL, SOH_COL, "material_capacity", "pulse_ms"]].to_csv(
        cache_csv,
        index=False,
        encoding="utf-8-sig",
    )

    x = df_all[u_cols].to_numpy(dtype=float)
    soc = df_all[SOC_COL].to_numpy(dtype=float)
    soh = df_all[SOH_COL].to_numpy(dtype=float)

    return x, soc, soh


# ============================================================
# Plot helpers
# ============================================================
def save_figure(fig, save_path: Path):
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def add_filter_note(ax, text: str):
    ax.text(
        0.02,
        0.96,
        text,
        transform=ax.transAxes,
        fontsize=10.5,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.78,
        },
    )


def get_soc_soh_groups(soc: np.ndarray, soh: np.ndarray):
    soc_rounded = np.round(soc / 10.0) * 10.0
    unique_socs = np.unique(soc_rounded[np.isfinite(soc_rounded)])

    if len(unique_socs) >= 3:
        target_socs = [
            unique_socs[0],
            unique_socs[len(unique_socs) // 2],
            unique_socs[-1],
        ]
    else:
        target_socs = unique_socs.tolist()

    healthy_soh_threshold = np.percentile(soh[np.isfinite(soh)], 85)

    if len(unique_socs) > 0:
        best_soc = float(stats.mode(soc_rounded, keepdims=False)[0])
    else:
        best_soc = float(np.nanmedian(soc))

    mask_best_soc = np.abs(soc - best_soc) <= 10.0

    valid_soh = soh[mask_best_soc]
    valid_soh = valid_soh[np.isfinite(valid_soh)]

    if len(valid_soh) > 0:
        target_sohs = [
            np.percentile(valid_soh, 5),
            np.percentile(valid_soh, 50),
            np.percentile(valid_soh, 95),
        ]
    else:
        target_sohs = []

    return target_socs, healthy_soh_threshold, best_soc, mask_best_soc, target_sohs


# ============================================================
# Figure panels
# ============================================================
def draw_u_evolution_split(
    x: np.ndarray,
    soc: np.ndarray,
    soh: np.ndarray,
    group: str,
):
    save_dir = get_material_save_dir(group)
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)

    x_axis = np.arange(1, 42)

    target_socs, healthy_soh_threshold, best_soc, mask_best_soc, target_sohs = (
        get_soc_soh_groups(soc, soh)
    )

    filter_note = f"Top 15% SOH samples: SOH >= {healthy_soh_threshold * 100:.1f}%"

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    colors_soc = sns.color_palette("Blues", max(3, len(target_socs)))

    for i, target_soc in enumerate(target_socs):
        mask = (soh >= healthy_soh_threshold) & (np.abs(soc - target_soc) <= 5.0)

        if np.sum(mask) > 0:
            ax1.plot(
                x_axis,
                np.mean(x[mask], axis=0),
                color=colors_soc[i],
                linewidth=5,
                label=f"SOC ≈ {target_soc:.0f}%",
            )

    ax1.set_title(f"Voltage baseline shift with SOC ({group})", fontweight="bold")
    ax1.set_xlabel("Feature index (U1-U41)")
    ax1.set_ylabel("Absolute voltage (V)")
    ax1.legend(frameon=False)
    ax1.grid(True, linestyle="--", alpha=0.4)

    for v in [1.5, 9.5, 17.5, 25.5, 33.5]:
        ax1.axvline(x=v, color="gray", linestyle=":", alpha=0.3)

    add_filter_note(ax1, filter_note)

    save_figure(fig1, save_dir / f"{group}_U_SOC.png")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    colors_soh = sns.color_palette("Reds_r", max(3, len(target_sohs) if target_sohs else 3))

    for i, target_soh in enumerate(target_sohs):
        mask = mask_best_soc & (np.abs(soh - target_soh) <= 0.05)

        if np.sum(mask) > 0:
            mean_u = np.mean(x[mask], axis=0)

            ax2.plot(
                x_axis,
                mean_u - mean_u[0],
                color=colors_soh[i],
                linewidth=5,
                label=f"SOH ≈ {target_soh * 100:.0f}%",
            )

    ax2.set_title(f"Polarization expansion ({group}, SOC≈{best_soc:.0f}%)", fontweight="bold")
    ax2.set_xlabel("Feature index (U1-U41)")
    ax2.set_ylabel("Voltage response ΔU (V)")
    ax2.legend(frameon=False)
    ax2.grid(True, linestyle="--", alpha=0.4)

    for v in [1.5, 9.5, 17.5, 25.5, 33.5]:
        ax2.axvline(x=v, color="gray", linestyle=":", alpha=0.3)

    save_figure(fig2, save_dir / f"{group}_U_SOH.png")


def draw_delta_u_evolution_split(
    x: np.ndarray,
    soc: np.ndarray,
    soh: np.ndarray,
    group: str,
):
    save_dir = get_material_save_dir(group)
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)

    x_delta = x[:, 1:] - x[:, :-1]
    x_axis = np.arange(2, 42)

    target_socs, healthy_soh_threshold, best_soc, mask_best_soc, target_sohs = (
        get_soc_soh_groups(soc, soh)
    )

    filter_note = f"Top 15% SOH samples: SOH >= {healthy_soh_threshold * 100:.1f}%"

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    colors_soc = sns.color_palette("Blues", max(3, len(target_socs)))

    for i, target_soc in enumerate(target_socs):
        mask = (soh >= healthy_soh_threshold) & (np.abs(soc - target_soc) <= 5.0)

        if np.sum(mask) > 0:
            ax1.plot(
                x_axis,
                np.mean(x_delta[mask], axis=0),
                color=colors_soc[i],
                linewidth=5,
                label=f"SOC ≈ {target_soc:.0f}%",
            )

    ax1.set_title(f"Differential voltage response with SOC ({group})", fontweight="bold")
    ax1.set_xlabel("Feature index (ΔU2-ΔU41)")
    ax1.set_ylabel("ΔU (V)")
    ax1.legend(frameon=False)
    ax1.grid(True, linestyle="--", alpha=0.4)

    add_filter_note(ax1, filter_note)

    save_figure(fig1, save_dir / f"{group}_DeltaU_SOC.png")

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    colors_soh = sns.color_palette("Reds_r", max(3, len(target_sohs) if target_sohs else 3))

    for i, target_soh in enumerate(target_sohs):
        mask = mask_best_soc & (np.abs(soh - target_soh) <= 0.05)

        if np.sum(mask) > 0:
            ax2.plot(
                x_axis,
                np.mean(x_delta[mask], axis=0),
                color=colors_soh[i],
                linewidth=5,
                label=f"SOH ≈ {target_soh * 100:.0f}%",
            )

    ax2.set_title(f"Differential polarization expansion ({group}, SOC≈{best_soc:.0f}%)", fontweight="bold")
    ax2.set_xlabel("Feature index (ΔU2-ΔU41)")
    ax2.set_ylabel("ΔU (V)")
    ax2.legend(frameon=False)
    ax2.grid(True, linestyle="--", alpha=0.4)

    save_figure(fig2, save_dir / f"{group}_DeltaU_SOH.png")


# ============================================================
# Main
# ============================================================
def main():
    print("[INFO] Generating Supplementary Figure 2...")

    generated_groups = []
    skipped_groups = []

    for group in TARGET_MATERIALS:
        x, soc, soh = get_material_data(group, max_samples=MAX_SAMPLES_PER_GROUP)

        if x is None:
            skipped_groups.append(group)
            continue

        draw_u_evolution_split(x, soc, soh, group)
        draw_delta_u_evolution_split(x, soc, soh, group)

        generated_groups.append(group)

    print("[DONE] Supplementary Figure 2 generated.")
    print(f"[SAVED] Output directory: {SAVE_ROOT}")
    print(f"[INFO] Generated groups: {len(generated_groups)}")
    print(f"[INFO] Skipped groups: {len(skipped_groups)}")

    if skipped_groups:
        print(f"[WARN] Skipped: {', '.join(skipped_groups)}")


if __name__ == "__main__":
    main()