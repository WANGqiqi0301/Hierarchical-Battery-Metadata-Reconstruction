# -*- coding: utf-8 -*-
"""
plot_fig2d_soc_soh_joint_distribution.py

Figure 2d:
SOC-SOH joint distribution for fixed-SOC and random-SOC samples.

This script is the organized version of the original Figure 2d code.

Original visual outputs are preserved:
1) Jointplot with marginal KDEs
2) Transparent scatter-only panel
3) SOH marginal KDE-only panel
4) SOC marginal KDE-only panel

For each data type:
    Fixed_SOC
    Random_SOC

Default output directory:
    results/figures/main/fig2d

This version uses the reorganized proposed_framework data builders and does
not depend on old a1_dataloader.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Project import
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import proposed_framework.run_proposed_framework as M


# =============================================================================
# Default configuration
# =============================================================================
DEFAULT_DATA_ROOT = getattr(M, "DATA_ROOT", str(PROJECT_ROOT.parent / "data"))
DEFAULT_OUTPUT_DIR = r"results/figures/main/fig2d"
DEFAULT_CACHE_FILE = "fig2d_soc_soh_data_cache.csv"

SOC_COL = "SOC"
SOH_COL = "SOH"

SOC_FIXED_LIST = list(range(5, 95, 5))
PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

DPI = 600


# =============================================================================
# Visual settings: keep original style
# =============================================================================
PROFESSIONAL_COLORS = [
    "#1F77B4",
    "#D62728",
    "#2CA02C",
    "#FF7F0E",
    "#9467BD",
    "#8C564B",
    "#00CED1",
    "#333333",
]

SCATTER_ALPHA = 0.1
SCATTER_SIZE = 60
KDE_ALPHA = 0.25


# =============================================================================
# Argument parser
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 2d SOC-SOH joint distribution plots."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Original data root directory.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save Figure 2d outputs.",
    )

    parser.add_argument(
        "--cache_file",
        type=str,
        default=DEFAULT_CACHE_FILE,
        help="Cache CSV filename under output_dir.",
    )

    parser.add_argument(
        "--force_rebuild_cache",
        action="store_true",
        help="Force rebuilding the SOC-SOH data cache.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI,
        help="Output figure resolution.",
    )

    return parser.parse_args()


# =============================================================================
# Utilities
# =============================================================================
def build_color_map(material_series: pd.Series) -> dict:
    """Build material-color mapping using original color list."""
    unique_materials = sorted(material_series.dropna().unique())

    return {
        material: PROFESSIONAL_COLORS[i % len(PROFESSIONAL_COLORS)]
        for i, material in enumerate(unique_materials)
    }


def ensure_required_columns(meta: pd.DataFrame, name: str) -> None:
    """Ensure metadata contains SOC and SOH columns."""
    missing = [col for col in [SOC_COL, SOH_COL] if col not in meta.columns]

    if missing:
        raise RuntimeError(
            f"{name} metadata is missing required columns: {missing}"
        )


# =============================================================================
# Data loading
# =============================================================================
def rebuild_combined_data(data_root: str) -> pd.DataFrame:
    """
    Rebuild Figure 2d data from the original data folder using proposed_framework.

    Fixed_SOC:
        build_train_mix_soc_mix_pt with SOC_FIXED_LIST and PULSE_LIST.

    Random_SOC:
        build_test_random_mix_pt with PULSE_LIST.
    """
    results = []

    print("[DATA] Rebuilding Fixed_SOC data from original data folder.")

    out_fixed = M.build_train_mix_soc_mix_pt(
        data_root=data_root,
        soc_list=SOC_FIXED_LIST,
        pulse_list=list(map(int, PULSE_LIST)),
        u_start=1,
        u_end=41,
        drop_first_class=True,
    )

    X_fixed, y_fixed, meta_fixed = out_fixed[0], out_fixed[1], out_fixed[2]

    if len(y_fixed) > 0:
        mask = np.isfinite(X_fixed).all(axis=1)
        y_fixed = y_fixed[mask]
        meta_fixed = meta_fixed.loc[mask].reset_index(drop=True)
        ensure_required_columns(meta_fixed, "Fixed_SOC")

        results.append(
            pd.DataFrame(
                {
                    "SOC": meta_fixed[SOC_COL].astype(float).values,
                    "SOH": meta_fixed[SOH_COL].astype(float).values,
                    "Material": y_fixed,
                    "Type": "Fixed_SOC",
                }
            )
        )

        print(f"[DATA] Fixed_SOC rows: {len(y_fixed)}")

    print("[DATA] Rebuilding Random_SOC data from original data folder.")

    out_random = M.build_test_random_mix_pt(
        data_root=data_root,
        pulse_list=list(map(int, PULSE_LIST)),
        u_start=1,
        u_end=41,
        drop_first_class=True,
    )

    X_random, y_random, meta_random = out_random[0], out_random[1], out_random[2]

    if len(y_random) > 0:
        mask = np.isfinite(X_random).all(axis=1)
        y_random = y_random[mask]
        meta_random = meta_random.loc[mask].reset_index(drop=True)
        ensure_required_columns(meta_random, "Random_SOC")

        results.append(
            pd.DataFrame(
                {
                    "SOC": meta_random[SOC_COL].astype(float).values,
                    "SOH": meta_random[SOH_COL].astype(float).values,
                    "Material": y_random,
                    "Type": "Random_SOC",
                }
            )
        )

        print(f"[DATA] Random_SOC rows: {len(y_random)}")

    if not results:
        raise RuntimeError("No data loaded for Figure 2d.")

    df = pd.concat(results, ignore_index=True)

    print("[DATA] Combined Figure 2d data:")
    print(df.groupby(["Type", "Material"]).size())

    return df


def get_combined_data(
    data_root: str,
    output_dir: str,
    cache_file: str,
    force_rebuild_cache: bool = False,
) -> pd.DataFrame:
    """Load cached combined data or rebuild it."""
    os.makedirs(output_dir, exist_ok=True)

    cache_path = os.path.join(output_dir, cache_file)

    if os.path.exists(cache_path) and not force_rebuild_cache:
        print(f"[CACHE] Loading Figure 2d data cache: {cache_path}")
        return pd.read_csv(cache_path)

    print("[CACHE] Cache missing or force rebuild enabled.")
    df = rebuild_combined_data(data_root=data_root)

    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"[CACHE] Saved Figure 2d data cache: {cache_path}")

    return df


# =============================================================================
# Plotting
# =============================================================================
def draw_four_versions(
    df: pd.DataFrame,
    data_type: str,
    prefix: str,
    save_dir: str,
    dpi: int = DPI,
) -> None:
    """
    Draw four output versions for one data type.

    Preserves original plotting logic:
    1) jointplot
    2) transparent scatter
    3) SOH KDE
    4) SOC KDE
    """
    subset = df[df["Type"] == data_type].copy()

    if subset.empty:
        print(f"[WARN] Empty subset for {data_type}. Skipped.")
        return

    print(f"[PLOT] Drawing {prefix} | N={len(subset)}")

    color_map = build_color_map(subset["Material"])

    save_base = os.path.join(save_dir, prefix)

    # -------------------------------------------------------------------------
    # 1) Jointplot
    # -------------------------------------------------------------------------
    g = sns.jointplot(
        data=subset,
        x="SOC",
        y="SOH",
        hue="Material",
        palette=color_map,
        height=8,
        alpha=0.25,
        s=10,
        linewidth=0,
        marginal_kws=dict(fill=True, alpha=KDE_ALPHA),
    )

    g.set_axis_labels("SOC (%)", "SOH")
    sns.move_legend(g.ax_joint, "upper left", bbox_to_anchor=(1.15, 1))

    plt.savefig(
        save_base + "_full.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close()

    # -------------------------------------------------------------------------
    # 2) Transparent scatter
    # -------------------------------------------------------------------------
    plt.figure(figsize=(6, 6))

    for material, color in color_map.items():
        sub = subset[subset["Material"] == material]

        plt.scatter(
            sub["SOC"],
            sub["SOH"],
            c=color,
            alpha=SCATTER_ALPHA,
            s=SCATTER_SIZE,
            edgecolors="none",
            rasterized=True,
        )

    plt.axis("off")

    plt.savefig(
        save_base + "_scatter.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close()

    # -------------------------------------------------------------------------
    # 3) SOH KDE
    # -------------------------------------------------------------------------
    plt.figure(figsize=(4, 6))

    for material, color in color_map.items():
        sub = subset[subset["Material"] == material]

        sns.kdeplot(
            y=sub["SOH"],
            fill=True,
            alpha=KDE_ALPHA,
            color=color,
            linewidth=0,
        )

    plt.axis("off")

    plt.savefig(
        save_base + "_SOH.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close()

    # -------------------------------------------------------------------------
    # 4) SOC KDE
    # -------------------------------------------------------------------------
    plt.figure(figsize=(6, 4))

    for material, color in color_map.items():
        sub = subset[subset["Material"] == material]

        sns.kdeplot(
            x=sub["SOC"],
            fill=True,
            alpha=KDE_ALPHA,
            color=color,
            linewidth=0,
        )

    plt.axis("off")

    plt.savefig(
        save_base + "_SOC.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = get_combined_data(
        data_root=args.data_root,
        output_dir=args.output_dir,
        cache_file=args.cache_file,
        force_rebuild_cache=args.force_rebuild_cache,
    )

    draw_four_versions(
        df=df,
        data_type="Fixed_SOC",
        prefix="Fig2d_Fixed",
        save_dir=args.output_dir,
        dpi=args.dpi,
    )

    draw_four_versions(
        df=df,
        data_type="Random_SOC",
        prefix="Fig2d_Random",
        save_dir=args.output_dir,
        dpi=args.dpi,
    )

    print("[DONE] Figure 2d generation complete.")


if __name__ == "__main__":
    main()