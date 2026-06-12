# -*- coding: utf-8 -*-
"""
suppfig3_feature_correlation_contours.py

Supplementary Figure 3:
Contour correlation maps for voltage-response features.

This script reads raw Excel data directly and generates, for each group:
1. Contour correlation map of polarization features

Input:
    data/

Output:
    results/figures/supp/suppfig3

Generated files:
    For each material-capacity group:
        <group>_Contour_correlation.png
        cache_feat_<group>.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.ndimage import gaussian_filter


# ============================================================
# Project paths
# ============================================================
PROJECT_ROOT = Path.cwd().resolve()

DATA_ROOT = PROJECT_ROOT / "data"
SAVE_ROOT = PROJECT_ROOT / "results" / "figures" / "supp" / "suppfig3"
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
CONTOUR_SMOOTH_SIGMA = 1.0


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


def draw_contour_correlation(x: np.ndarray, group: str):
    save_dir = get_material_save_dir(group)

    x_delta_corr = x[:, 1:] - x[:, [0]]
    x_delta_corr = x_delta_corr + np.random.default_rng(RANDOM_SEED).normal(
        0,
        1e-9,
        x_delta_corr.shape,
    )

    corr_matrix = np.corrcoef(x_delta_corr, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

    corr_smooth = gaussian_filter(corr_matrix, sigma=CONTOUR_SMOOTH_SIGMA)

    n = corr_smooth.shape[0]
    xx = np.arange(n)
    yy = np.arange(n)
    x_grid, y_grid = np.meshgrid(xx, yy)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.set_context("paper", font_scale=1.2)

    contour = ax.contourf(
        x_grid,
        y_grid,
        corr_smooth,
        levels=np.linspace(-1, 1, 17),
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )

    ax.contour(
        x_grid,
        y_grid,
        corr_smooth,
        levels=np.linspace(-1, 1, 9),
        colors="k",
        linewidths=0.45,
        alpha=0.45,
    )

    cbar = fig.colorbar(contour, ax=ax, shrink=0.82)
    cbar.set_label("Pearson correlation coefficient (r)")

    ticks = [0, 8, 16, 24, 32, 39]
    labels = ["U2", "U10", "U18", "U26", "U34", "U41"]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for v in [8, 16, 24, 32]:
        ax.axvline(v, color="white", linewidth=1.2, alpha=0.9)
        ax.axhline(v, color="white", linewidth=1.2, alpha=0.9)

    ax.set_title(f"Feature correlation structure for {group}", fontweight="bold", pad=15)
    ax.set_xlabel("Polarization feature index")
    ax.set_ylabel("Polarization feature index")

    save_figure(fig, save_dir / f"{group}_Contour_correlation.png")


# ============================================================
# Main
# ============================================================
def main():
    print("[INFO] Generating Supplementary Figure 3...")

    generated_groups = []
    skipped_groups = []

    for group in TARGET_MATERIALS:
        x, soc, soh = get_material_data(group, max_samples=MAX_SAMPLES_PER_GROUP)

        if x is None:
            skipped_groups.append(group)
            continue

        draw_contour_correlation(x, group)
        generated_groups.append(group)

    print("[DONE] Supplementary Figure 3 generated.")
    print(f"[SAVED] Output directory: {SAVE_ROOT}")
    print(f"[INFO] Generated groups: {len(generated_groups)}")
    print(f"[INFO] Skipped groups: {len(skipped_groups)}")

    if skipped_groups:
        print(f"[WARN] Skipped: {', '.join(skipped_groups)}")


if __name__ == "__main__":
    main()