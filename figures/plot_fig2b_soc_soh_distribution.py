# -*- coding: utf-8 -*-
"""
plot_fig2b_soc_soh_distribution.py

Figure 2b:
Train/test SOC and SOH distribution plots.

This script supports two modes:
1) If cache files exist:
   - read draw_figures/SOC/soc_train_test_cache.npz
   - read draw_figures/SOH/soh_train_test_cache.npz

2) If cache files do not exist:
   - rebuild train/test data using proposed_framework.run_proposed_framework
   - reproduce the same ID-level split logic
   - save SOC/SOH cache files automatically
   - then plot Figure 2b

Default outputs:
    results/figures/main/fig2b/fig2b_soc_distribution.png
    results/figures/main/fig2b/fig2b_soh_distribution.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# =============================================================================
# Project import
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import proposed_framework.run_proposed_framework as M


# =============================================================================
# Visual settings: keep exactly the same as the original code
# =============================================================================
COLOR_TR = "C0"
COLOR_TE = "C1"
LINE_WIDTH = 4.0
GLOBAL_ALPHA = 0.3


# =============================================================================
# Default paths and config
# =============================================================================
DEFAULT_DATA_ROOT = getattr(M, "DATA_ROOT", str(PROJECT_ROOT.parent / "data"))
DEFAULT_EXP_DIR = getattr(M, "EXP_DIR", r"results/proposed_framework")

DEFAULT_SOC_CACHE = r"draw_figures/SOC/soc_train_test_cache.npz"
DEFAULT_SOH_CACHE = r"draw_figures/SOH/soh_train_test_cache.npz"

DEFAULT_OUTPUT_DIR = r"results/figures/main/fig2b"
DEFAULT_SOC_OUTPUT_NAME = "fig2b_soc_distribution.png"
DEFAULT_SOH_OUTPUT_NAME = "fig2b_soh_distribution.png"

SOC_LIST = list(range(5, 90, 5))
PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

SOC_COL = "SOC"
SOH_COL = "SOH"
ID_COL = "ID"

SEED = 42
TEST_ID_FRAC = 0.2
TEST_ID_COUNT = 0


# =============================================================================
# Argument parser
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 2b SOC/SOH train-test distribution plots."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Original data root directory.",
    )

    parser.add_argument(
        "--exp_dir",
        type=str,
        default=DEFAULT_EXP_DIR,
        help="Experiment directory.",
    )

    parser.add_argument(
        "--soc_cache",
        type=str,
        default=DEFAULT_SOC_CACHE,
        help="Path to SOC cache npz.",
    )

    parser.add_argument(
        "--soh_cache",
        type=str,
        default=DEFAULT_SOH_CACHE,
        help="Path to SOH cache npz.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output figures.",
    )

    parser.add_argument(
        "--soc_output_name",
        type=str,
        default=DEFAULT_SOC_OUTPUT_NAME,
        help="Filename for SOC distribution plot.",
    )

    parser.add_argument(
        "--soh_output_name",
        type=str,
        default=DEFAULT_SOH_OUTPUT_NAME,
        help="Filename for SOH distribution plot.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure resolution.",
    )

    parser.add_argument(
        "--force_rebuild_cache",
        action="store_true",
        help="Force rebuilding SOC/SOH cache from proposed_framework data.",
    )

    parser.add_argument(
        "--hide_soc_legend",
        action="store_true",
        help="Hide legend for SOC plot.",
    )

    parser.add_argument(
        "--show_soh_legend",
        action="store_true",
        help="Show legend for SOH plot.",
    )

    return parser.parse_args()


# =============================================================================
# Cache utilities
# =============================================================================
def cache_exists(soc_cache: str, soh_cache: str) -> bool:
    return os.path.exists(soc_cache) and os.path.exists(soh_cache)


def load_soc_cache(cache_path: str) -> tuple[np.ndarray, np.ndarray]:
    obj = np.load(cache_path)
    return obj["soc_tr"].astype(np.float64), obj["soc_te"].astype(np.float64)


def load_soh_cache(cache_path: str) -> tuple[np.ndarray, np.ndarray]:
    obj = np.load(cache_path)
    return obj["soh_tr"].astype(np.float64), obj["soh_te"].astype(np.float64)


def save_distribution_cache(
    soc_cache: str,
    soh_cache: str,
    soc_tr: np.ndarray,
    soc_te: np.ndarray,
    soh_tr: np.ndarray,
    soh_te: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(soc_cache), exist_ok=True)
    os.makedirs(os.path.dirname(soh_cache), exist_ok=True)

    np.savez_compressed(
        soc_cache,
        soc_tr=np.asarray(soc_tr, dtype=np.float32),
        soc_te=np.asarray(soc_te, dtype=np.float32),
    )

    np.savez_compressed(
        soh_cache,
        soh_tr=np.asarray(soh_tr, dtype=np.float32),
        soh_te=np.asarray(soh_te, dtype=np.float32),
    )

    print(f"[CACHE] Saved SOC cache: {soc_cache}")
    print(f"[CACHE] Saved SOH cache: {soh_cache}")


# =============================================================================
# Build cache from proposed_framework
# =============================================================================
def build_train_test_soc_soh_from_framework(
    data_root: str,
    exp_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rebuild train/test SOC and SOH arrays using the same split logic as
    run_proposed_framework / run_further_analysis_proposed.
    """
    cache_dir = os.path.join(exp_dir, "cache")

    train_kwargs = dict(
        data_root=data_root,
        soc_list=SOC_LIST,
        pulse_list=list(map(int, PULSE_LIST)),
        u_start=1,
        u_end=41,
        drop_first_class=True,
    )

    test_kwargs = dict(
        data_root=data_root,
        pulse_list=list(map(int, PULSE_LIST)),
        u_start=1,
        u_end=41,
        drop_first_class=True,
    )

    Xtr_raw, ytr_raw, mtr_raw, _, _ = M.load_or_build_cache(
        cache_dir,
        "raw_train",
        M.build_train_mix_soc_mix_pt,
        train_kwargs,
    )

    Xte_raw, yte_raw, mte_raw, _, _ = M.load_or_build_cache(
        cache_dir,
        "raw_test",
        M.build_test_random_mix_pt,
        test_kwargs,
    )

    Xtr_raw, ytr_raw, mtr_raw = M.drop_nan_inf_rows(
        Xtr_raw,
        ytr_raw,
        mtr_raw,
        "RAW_TRAIN_FIG2B",
    )

    Xte_raw, yte_raw, mte_raw = M.drop_nan_inf_rows(
        Xte_raw,
        yte_raw,
        mte_raw,
        "RAW_TEST_FIG2B",
    )

    if ID_COL not in mtr_raw.columns or ID_COL not in mte_raw.columns:
        raise RuntimeError("Meta must contain ID column for train/test split.")

    all_ids = np.concatenate([
        mtr_raw[ID_COL].astype(str).to_numpy(),
        mte_raw[ID_COL].astype(str).to_numpy(),
    ])

    test_ids = M.pick_test_ids(
        all_ids,
        test_id_frac=TEST_ID_FRAC,
        test_id_count=TEST_ID_COUNT,
        seed=SEED,
    )

    Xtr, ytr, mtr, Xte, yte, mte = M.apply_id_split(
        Xtr_raw,
        ytr_raw,
        mtr_raw,
        Xte_raw,
        yte_raw,
        mte_raw,
        test_ids=test_ids,
    )

    if SOC_COL not in mtr.columns or SOC_COL not in mte.columns:
        raise RuntimeError(f"Missing SOC column: {SOC_COL}")

    if SOH_COL not in mtr.columns or SOH_COL not in mte.columns:
        raise RuntimeError(f"Missing SOH column: {SOH_COL}")

    soc_tr = mtr[SOC_COL].astype(float).to_numpy(dtype=np.float64)
    soc_te = mte[SOC_COL].astype(float).to_numpy(dtype=np.float64)

    soh_tr = mtr[SOH_COL].astype(float).to_numpy(dtype=np.float64)
    soh_te = mte[SOH_COL].astype(float).to_numpy(dtype=np.float64)

    print("[DATA] Rebuilt Figure 2b distribution data:")
    print(f"       SOC train/test: {soc_tr.shape} / {soc_te.shape}")
    print(f"       SOH train/test: {soh_tr.shape} / {soh_te.shape}")

    return soc_tr, soc_te, soh_tr, soh_te


def get_or_build_distribution_data(args):
    if cache_exists(args.soc_cache, args.soh_cache) and not args.force_rebuild_cache:
        print("[CACHE] Loading existing SOC/SOH distribution caches.")
        soc_tr, soc_te = load_soc_cache(args.soc_cache)
        soh_tr, soh_te = load_soh_cache(args.soh_cache)
        return soc_tr, soc_te, soh_tr, soh_te

    print("[CACHE] SOC/SOH cache not found or force rebuild enabled.")
    print("[DATA] Rebuilding SOC/SOH distributions from proposed_framework data.")

    soc_tr, soc_te, soh_tr, soh_te = build_train_test_soc_soh_from_framework(
        data_root=args.data_root,
        exp_dir=args.exp_dir,
    )

    save_distribution_cache(
        soc_cache=args.soc_cache,
        soh_cache=args.soh_cache,
        soc_tr=soc_tr,
        soc_te=soc_te,
        soh_tr=soh_tr,
        soh_te=soh_te,
    )

    return soc_tr, soc_te, soh_tr, soh_te


# =============================================================================
# Plotting
# =============================================================================
def plot_soc_distribution(
    soc_tr: np.ndarray,
    soc_te: np.ndarray,
    save_path: str,
    dpi: int = 300,
    show_legend: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=dpi)
    xlim = (0, 100)

    hist_bins = np.linspace(xlim[0], xlim[1], 41)

    ax.hist(
        soc_te,
        bins=hist_bins,
        density=True,
        alpha=GLOBAL_ALPHA,
        color=COLOR_TE,
        edgecolor="none",
        label="Test SOC",
    )

    xs = np.linspace(xlim[0], xlim[1], 400)
    kde = gaussian_kde(soc_te)

    ax.plot(
        xs,
        kde(xs),
        color=COLOR_TE,
        linewidth=LINE_WIDTH,
        label="Test KDE",
    )

    uniq, cnt = np.unique(soc_tr, return_counts=True)
    dens = cnt / (len(soc_tr) * 5.0)

    ax.bar(
        uniq,
        dens,
        width=3.2,
        alpha=GLOBAL_ALPHA,
        color=COLOR_TR,
        edgecolor="none",
        label="Train SOC (Grid)",
    )

    ax.set_xlim(xlim)
    ax.set_xlabel("SOC (%)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.2)

    if show_legend:
        ax.legend(frameon=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", transparent=False, pad_inches=0)
    plt.close(fig)


def plot_soh_distribution(
    soh_tr: np.ndarray,
    soh_te: np.ndarray,
    save_path: str,
    dpi: int = 300,
    show_legend: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=dpi)

    lo = min(soh_tr.min(), soh_te.min())
    hi = max(soh_tr.max(), soh_te.max())
    hist_bins = np.linspace(lo, hi, 41)

    ax.hist(
        soh_tr,
        bins=hist_bins,
        density=True,
        alpha=GLOBAL_ALPHA,
        color=COLOR_TR,
        edgecolor="none",
        label="Train SOH",
    )

    ax.hist(
        soh_te,
        bins=hist_bins,
        density=True,
        alpha=GLOBAL_ALPHA,
        color=COLOR_TE,
        edgecolor="none",
        label="Test SOH",
    )

    xs = np.linspace(lo, hi, 400)
    kde_tr = gaussian_kde(soh_tr)
    kde_te = gaussian_kde(soh_te)

    ax.plot(
        xs,
        kde_tr(xs),
        color=COLOR_TR,
        linewidth=LINE_WIDTH,
        label="Train KDE",
    )

    ax.plot(
        xs,
        kde_te(xs),
        color=COLOR_TE,
        linewidth=LINE_WIDTH,
        label="Test KDE",
    )

    ax.set_xlabel("SOH")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.2)

    if show_legend:
        ax.legend(frameon=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", transparent=False, pad_inches=0)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()

    soc_tr, soc_te, soh_tr, soh_te = get_or_build_distribution_data(args)

    soc_save_path = os.path.join(args.output_dir, args.soc_output_name)
    soh_save_path = os.path.join(args.output_dir, args.soh_output_name)

    plot_soc_distribution(
        soc_tr=soc_tr,
        soc_te=soc_te,
        save_path=soc_save_path,
        dpi=args.dpi,
        show_legend=not args.hide_soc_legend,
    )

    plot_soh_distribution(
        soh_tr=soh_tr,
        soh_te=soh_te,
        save_path=soh_save_path,
        dpi=args.dpi,
        show_legend=args.show_soh_legend,
    )

    print("[OK] Figure 2b distribution plots saved:")
    print(f"  {soc_save_path}")
    print(f"  {soh_save_path}")


if __name__ == "__main__":
    main()