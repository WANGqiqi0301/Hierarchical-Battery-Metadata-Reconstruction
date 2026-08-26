# -*- coding: utf-8 -*-
"""
plot_fig2f_u_evolution.py

Figure 2f:
U1-U41 evolution with SOC and SOH.

Only standard annotated figures are saved.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# =============================================================================
# Project path
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import proposed_framework.run_proposed_framework as M


# =============================================================================
# Configuration
# =============================================================================
DATA_ROOT = str(PROJECT_ROOT.parent / "data")
SAVE_DIR = r"results/figures/main/fig2f"

TARGET_MATERIAL = "LMO_24Ah"
CACHE_CSV = os.path.join(SAVE_DIR, f"cache_feat_{TARGET_MATERIAL}.csv")

SOC_COL = "SOC"
SOH_COL = "SOH"

SOC_LIST = [20, 30, 50, 80, 90]
PULSE_LIST = [100, 500, 1000]

MAX_SAMPLES = 5000
RANDOM_SEED = 42
DPI = 600

os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# Data
# =============================================================================
def load_material_data_from_cache(cache_csv: str):
    df = pd.read_csv(cache_csv)
    feature_cols = [f"U{i}" for i in range(1, 42)]

    missing = [c for c in feature_cols + [SOC_COL, SOH_COL] if c not in df.columns]
    if missing:
        raise RuntimeError(f"Cache is missing columns: {missing}")

    X = df[feature_cols].to_numpy(dtype=float)
    soc = df[SOC_COL].to_numpy(dtype=float)
    soh = df[SOH_COL].to_numpy(dtype=float)

    return X, soc, soh


def save_material_cache(cache_csv: str, X: np.ndarray, soc: np.ndarray, soh: np.ndarray) -> None:
    feature_cols = [f"U{i}" for i in range(1, 42)]

    df = pd.DataFrame(X, columns=feature_cols)
    df[SOC_COL] = soc
    df[SOH_COL] = soh

    os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
    df.to_csv(cache_csv, index=False, encoding="utf-8-sig")


def get_material_data(target_material: str):
    if os.path.exists(CACHE_CSV):
        print(f"[CACHE] Loading: {CACHE_CSV}")
        return load_material_data_from_cache(CACHE_CSV)

    print(f"[DATA] Loading raw data for {target_material}")

    out = M.build_train_mix_soc_mix_pt(
        data_root=DATA_ROOT,
        soc_list=list(map(int, SOC_LIST)),
        pulse_list=list(map(int, PULSE_LIST)),
        u_start=1,
        u_end=41,
        drop_first_class=True,
    )

    X_all = out[0]
    y_all = out[1]
    meta_all = out[2]

    if len(y_all) == 0:
        raise RuntimeError("No data loaded from build_train_mix_soc_mix_pt.")

    mask_material = y_all == target_material

    if not np.any(mask_material):
        available_labels = sorted(pd.Series(y_all).astype(str).unique().tolist())
        raise RuntimeError(
            f"No data found for {target_material}. "
            f"Available labels: {available_labels}"
        )

    X_mat = X_all[mask_material]
    meta_mat = meta_all.loc[mask_material].reset_index(drop=True)

    mask_valid = (
        np.isfinite(X_mat).all(axis=1)
        & np.isfinite(meta_mat[SOC_COL].astype(float).to_numpy())
        & np.isfinite(meta_mat[SOH_COL].astype(float).to_numpy())
    )

    X_mat = X_mat[mask_valid]
    soc_mat = meta_mat.loc[mask_valid, SOC_COL].astype(float).to_numpy()
    soh_mat = meta_mat.loc[mask_valid, SOH_COL].astype(float).to_numpy()

    if len(X_mat) == 0:
        raise RuntimeError(f"No valid finite data found for {target_material}")

    if len(X_mat) > MAX_SAMPLES:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(X_mat), MAX_SAMPLES, replace=False)

        X_mat = X_mat[idx]
        soc_mat = soc_mat[idx]
        soh_mat = soh_mat[idx]

    save_material_cache(CACHE_CSV, X_mat, soc_mat, soh_mat)

    print(f"[CACHE] Saved: {CACHE_CSV}")
    print(f"[DATA] Samples: {len(X_mat)}")

    return X_mat, soc_mat, soh_mat


# =============================================================================
# Plot helpers
# =============================================================================
def add_feature_boundaries(ax) -> None:
    for x in [1.5, 9.5, 17.5, 25.5, 33.5]:
        ax.axvline(x=x, color="gray", linestyle=":", alpha=0.3)


def save_standard(fig, save_name: str) -> None:
    save_path = os.path.join(SAVE_DIR, save_name)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {save_path}")


# =============================================================================
# Plots
# =============================================================================
def plot_soc_evolution(X: np.ndarray, soc: np.ndarray, soh: np.ndarray) -> None:
    x_axis = np.arange(1, 42)
    soc_rounded = np.round(soc / 10.0) * 10.0

    unique_socs = np.unique(soc_rounded)

    if len(unique_socs) >= 3:
        target_socs = [
            unique_socs[0],
            unique_socs[len(unique_socs) // 2],
            unique_socs[-1],
        ]
    else:
        target_socs = unique_socs

    healthy_soh_threshold = np.percentile(soh, 85)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = sns.color_palette("Blues", 3)

    for i, target_soc in enumerate(target_socs):
        mask = (soh >= healthy_soh_threshold) & (np.abs(soc - target_soc) <= 5)

        if np.sum(mask) > 0:
            ax.plot(
                x_axis,
                np.mean(X[mask], axis=0),
                color=colors[i % 3],
                linewidth=5,
                label=f"SOC approx. {target_soc:.0f}%",
            )

    ax.set_title("A. Voltage Baseline Shift with SOC", fontweight="bold")
    ax.set_xlabel("Feature Index (U1 - U41)")
    ax.set_ylabel("Absolute Voltage (V)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    add_feature_boundaries(ax)

    save_standard(fig, "Fig2f_SOC_Evolution_Standard.png")


def plot_soh_evolution(X: np.ndarray, soc: np.ndarray, soh: np.ndarray) -> None:
    x_axis = np.arange(1, 42)
    soc_rounded = np.round(soc / 10.0) * 10.0

    best_soc = float(stats.mode(soc_rounded, keepdims=False)[0])
    mask_best_soc = np.abs(soc - best_soc) <= 10
    valid_soh = soh[mask_best_soc]

    if len(valid_soh) == 0:
        raise RuntimeError("No valid SOH values for SOH evolution plot.")

    target_sohs = [
        np.percentile(valid_soh, 5),
        np.percentile(valid_soh, 50),
        np.percentile(valid_soh, 95),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = sns.color_palette("Reds_r", 3)

    for i, target_soh in enumerate(target_sohs):
        mask = mask_best_soc & (np.abs(soh - target_soh) <= 0.05)

        if np.sum(mask) > 0:
            mean_u = np.mean(X[mask], axis=0)

            ax.plot(
                x_axis,
                mean_u - mean_u[0],
                color=colors[i % 3],
                linewidth=5,
                label=f"SOH approx. {target_soh * 100:.0f}%",
            )

    ax.set_title(
        f"B. Polarization Expansion (SOC approx. {best_soc:.0f}%)",
        fontweight="bold",
    )
    ax.set_xlabel("Feature Index (U1 - U41)")
    ax.set_ylabel(r"Voltage Response $\Delta U$ (V)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    add_feature_boundaries(ax)

    save_standard(fig, "Fig2f_SOH_Evolution_Standard.png")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)

    X, soc, soh = get_material_data(TARGET_MATERIAL)

    plot_soc_evolution(X, soc, soh)
    plot_soh_evolution(X, soc, soh)

    print("[DONE] Figure 2f standard plots generated.")


if __name__ == "__main__":
    main()