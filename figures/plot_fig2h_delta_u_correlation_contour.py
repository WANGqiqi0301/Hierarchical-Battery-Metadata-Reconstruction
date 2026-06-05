# -*- coding: utf-8 -*-
"""
plot_fig2h_delta_u_correlation_contour.py

Figure 2h:
Contour map of Pearson correlations among Delta-U polarization features.

The feature matrix is constructed as:
    [U2 - U1, U3 - U1, ..., U41 - U1]

Only the contour map is generated.

Default outputs:
    results/figures/main/fig2h/fig2h_delta_u_correlation_contour.png
    results/figures/main/fig2h/corr_matrix_u2_u41_minus_u1.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import gaussian_filter


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
SAVE_DIR = r"results/figures/main/fig2h"

TARGET_MATERIAL = "LMO_24Ah"
CACHE_CSV = os.path.join(SAVE_DIR, f"cache_feat_deltaU_{TARGET_MATERIAL}.csv")

SOC_COL = "SOC"
SOH_COL = "SOH"

SOC_LIST = [20, 30, 50, 80, 90]
PULSE_LIST = [100, 500, 1000]

MAX_SAMPLES = 5000
RANDOM_SEED = 42
SMOOTH_SIGMA = 1.0
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
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(X_mat), MAX_SAMPLES, replace=False)

        X_mat = X_mat[idx]
        soc_mat = soc_mat[idx]
        soh_mat = soh_mat[idx]

    save_material_cache(CACHE_CSV, X_mat, soc_mat, soh_mat)

    print(f"[CACHE] Saved: {CACHE_CSV}")
    print(f"[DATA] Samples: {len(X_mat)}")

    return X_mat, soc_mat, soh_mat


# =============================================================================
# Correlation
# =============================================================================
def build_delta_features(X: np.ndarray) -> np.ndarray:
    """
    Build Delta-U features:
        U2-U1, U3-U1, ..., U41-U1

    Output shape:
        (N, 40)
    """
    if X.shape[1] != 41:
        raise ValueError(f"Expected X with 41 columns, got shape {X.shape}")

    rng = np.random.default_rng(RANDOM_SEED)
    X_feat = X[:, 1:] - X[:, [0]]
    X_feat = X_feat + rng.normal(0, 1e-9, X_feat.shape)

    return X_feat


def compute_corr_matrix(X_feat: np.ndarray) -> np.ndarray:
    corr_matrix = np.corrcoef(X_feat, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    return corr_matrix


def save_corr_matrix(corr_matrix: np.ndarray) -> None:
    labels = [f"U{i}-U1" for i in range(2, 42)]

    corr_df = pd.DataFrame(
        corr_matrix,
        index=labels,
        columns=labels,
    )

    save_path = os.path.join(SAVE_DIR, "corr_matrix_u2_u41_minus_u1.csv")
    corr_df.to_csv(save_path, encoding="utf-8-sig")

    print(f"[OK] Saved: {save_path}")


# =============================================================================
# Plot
# =============================================================================
def plot_contour_map(
    corr_matrix: np.ndarray,
    target_material: str,
    sigma: float = 1.0,
) -> None:
    sns.set_theme(style="white", context="paper", font_scale=1.2)

    corr_smooth = gaussian_filter(corr_matrix, sigma=sigma)

    n = corr_smooth.shape[0]
    x = np.arange(n)
    y = np.arange(n)
    Xg, Yg = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8.6, 7.6))

    contour = ax.contourf(
        Xg,
        Yg,
        corr_smooth,
        levels=np.linspace(-1, 1, 17),
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )

    ax.contour(
        Xg,
        Yg,
        corr_smooth,
        levels=np.linspace(-1, 1, 9),
        colors="k",
        linewidths=0.45,
        alpha=0.45,
    )

    cbar = fig.colorbar(contour, ax=ax, shrink=0.82)
    cbar.set_label("Pearson Correlation Coefficient ($r$)")

    ticks = [0, 8, 16, 24, 32, 39]
    tick_labels = ["U2", "U10", "U18", "U26", "U34", "U41"]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    for value in [8, 16, 24, 32]:
        ax.axvline(value, color="white", linewidth=1.2, alpha=0.9)
        ax.axhline(value, color="white", linewidth=1.2, alpha=0.9)

    ax.set_title(
        f"Contour Map of Feature Correlation ({target_material})",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Polarization Feature Index")
    ax.set_ylabel("Polarization Feature Index")

    save_path = os.path.join(SAVE_DIR, "fig2h_delta_u_correlation_contour.png")
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    print("[START] Generating Figure 2h contour map.")

    X, _, _ = get_material_data(TARGET_MATERIAL)

    X_feat = build_delta_features(X)
    corr_matrix = compute_corr_matrix(X_feat)

    save_corr_matrix(corr_matrix)
    plot_contour_map(
        corr_matrix=corr_matrix,
        target_material=TARGET_MATERIAL,
        sigma=SMOOTH_SIGMA,
    )

    print("[DONE] Figure 2h contour map generated.")


if __name__ == "__main__":
    main()