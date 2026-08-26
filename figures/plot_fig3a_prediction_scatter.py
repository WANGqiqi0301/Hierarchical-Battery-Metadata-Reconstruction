# -*- coding: utf-8 -*-
"""
figures/plot_fig3a_prediction_scatter.py

Generate Figure 3a directly from further-analysis prediction CSV files.

Inputs:
    results/proposed_framework/further_analysis/tables/
        train_predictions_for_scatter.csv
        test_predictions_per_sample.csv

Required columns:
    soc_true, soc_pred, soh_true, soh_pred

Outputs:
    results/figures/main/fig3a/
        fig3a_soh_train_full.png/pdf/svg
        fig3a_soh_train_pure.png/pdf/svg
        fig3a_soh_test_full.png/pdf/svg
        fig3a_soh_test_pure.png/pdf/svg
        fig3a_soc_train_full.png/pdf/svg
        fig3a_soc_train_pure.png/pdf/svg
        fig3a_soc_test_full.png/pdf/svg
        fig3a_soc_test_pure.png/pdf/svg
        fig3a_prediction_scatter_2x2.png/pdf/svg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TRAIN_CSV = (
    PROJECT_ROOT
    / "results"
    / "proposed_framework"
    / "further_analysis"
    / "tables"
    / "train_predictions_for_scatter.csv"
)

DEFAULT_TEST_CSV = (
    PROJECT_ROOT
    / "results"
    / "proposed_framework"
    / "further_analysis"
    / "tables"
    / "test_predictions_per_sample.csv"
)

DEFAULT_FIG_OUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "main"
    / "fig3a"
)

PLOT_CONFIG = {
    "hex_colors": ["#732C7C", "#4B7DA6", "#65A5D9", "#41C28A", "#FFEF30"],
    "dot_size": 120,
    "font": "Arial",
    "bw_method": 0.1,
}

REQUIRED_COLUMNS = ["soc_true", "soc_pred", "soh_true", "soh_pred"]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(
    fig: plt.Figure,
    out_base: str | Path,
    dpi: int = 300,
    transparent: bool = False,
) -> None:
    out_base = Path(out_base)
    ensure_dir(out_base.parent)

    fig.savefig(
        str(out_base) + ".png",
        dpi=dpi,
        bbox_inches="tight",
        transparent=transparent,
    )
    fig.savefig(
        str(out_base) + ".pdf",
        bbox_inches="tight",
        transparent=transparent,
    )
    fig.savefig(
        str(out_base) + ".svg",
        bbox_inches="tight",
        transparent=transparent,
    )


def load_prediction_csv(path: str | Path, split: str) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{split} prediction CSV not found:\n{path}"
        )

    df = pd.read_csv(path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError(
            f"{split} prediction CSV is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    removed = before - len(df)

    if len(df) == 0:
        raise RuntimeError(
            f"No valid prediction rows remain in {split} CSV after dropping NaN/invalid values."
        )

    print(f"[LOAD] {split.upper()} CSV: {path}")
    print(f"[LOAD] {split.upper()} rows: {len(df)}")
    if removed > 0:
        print(f"[WARN] {split.upper()} removed invalid rows: {removed}")

    return df


def build_predictions(
    train_csv: str | Path,
    test_csv: str | Path,
) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    df_train = load_prediction_csv(train_csv, split="train")
    df_test = load_prediction_csv(test_csv, split="test")

    return {
        ("soc", "train"): (
            df_train["soc_true"].to_numpy(dtype=np.float64),
            df_train["soc_pred"].to_numpy(dtype=np.float64),
        ),
        ("soc", "test"): (
            df_test["soc_true"].to_numpy(dtype=np.float64),
            df_test["soc_pred"].to_numpy(dtype=np.float64),
        ),
        ("soh", "train"): (
            df_train["soh_true"].to_numpy(dtype=np.float64),
            df_train["soh_pred"].to_numpy(dtype=np.float64),
        ),
        ("soh", "test"): (
            df_test["soh_true"].to_numpy(dtype=np.float64),
            df_test["soh_pred"].to_numpy(dtype=np.float64),
        ),
    }


def convert_soh_to_percent_if_needed(
    t: np.ndarray,
    p: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    t2 = t.astype(np.float64).copy()
    p2 = p.astype(np.float64).copy()

    max_true = np.nanmax(np.abs(t2))
    max_pred = np.nanmax(np.abs(p2))

    if max_true <= 2.0 and max_pred <= 2.0:
        t2 *= 100.0
        p2 *= 100.0
        print("[PLOT] SOH appears to be in 0-1 scale. Converted to percentage.")
    else:
        print("[PLOT] SOH appears to already be in percentage scale. No conversion.")

    return t2, p2


def compute_metrics(t: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    ape = np.abs((t - p) / np.maximum(np.abs(t), 1e-8)) * 100.0
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    medape = float(np.median(ape))
    return {"rmse": rmse, "medape": medape}


def compute_kde_density(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    target: str,
    cfg: dict,
) -> np.ndarray:
    t = true_values.astype(np.float64).copy()
    p = pred_values.astype(np.float64).copy()

    if target.lower() == "soh":
        t, p = convert_soh_to_percent_if_needed(t, p)

    xy = np.vstack([t, p])

    if xy.shape[1] < 3:
        raise RuntimeError(f"Not enough points for KDE: {xy.shape[1]}")

    z = gaussian_kde(
        xy,
        bw_method=cfg["bw_method"],
    )(xy)

    return z


def compute_global_kde_range(
    predictions: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
    cfg: dict,
) -> Tuple[float, float]:
    all_z = []

    for target in ["soh", "soc"]:
        for split in ["train", "test"]:
            t, p = predictions[(target, split)]

            z = compute_kde_density(
                true_values=t,
                pred_values=p,
                target=target,
                cfg=cfg,
            )

            all_z.append(z)

            print(
                f"[KDE RANGE] {target.upper()} {split}: "
                f"min={z.min():.6e}, max={z.max():.6e}"
            )

    all_z = np.concatenate(all_z)

    global_z_min = float(np.min(all_z))
    global_z_max = float(np.max(all_z))

    print(
        f"[KDE GLOBAL RANGE] "
        f"min={global_z_min:.6e}, "
        f"max={global_z_max:.6e}"
    )

    return global_z_min, global_z_max


def kde_scatter_on_ax(
    ax: plt.Axes,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    target: str,
    split: str,
    cfg: dict,
    global_z_min: float,
    global_z_max: float,
    show_title: bool = True,
    show_labels: bool = True,
):
    target = target.lower()
    split = split.lower()

    t = true_values.astype(np.float64).copy()
    p = pred_values.astype(np.float64).copy()

    if target == "soh":
        t, p = convert_soh_to_percent_if_needed(t, p)
        xlabel = "Measured SOH (%)"
        ylabel = "Predicted SOH (%)"

        vmin = min(float(np.nanmin(t)), float(np.nanmin(p)))
        vmax = max(float(np.nanmax(t)), float(np.nanmax(p)))
        margin = (vmax - vmin) * 0.05 if vmax > vmin else 1.0
        lims = [vmin - margin, vmax + margin]

    elif target == "soc":
        xlabel = "Measured SOC (%)"
        ylabel = "Predicted SOC (%)"
        lims = [0, 100]

    else:
        raise ValueError("target must be 'soc' or 'soh'.")

    metrics = compute_metrics(t, p)

    print(
        f"[METRIC] {target.upper()} {split}: "
        f"Median APE = {metrics['medape']:.4f}% | "
        f"RMSE = {metrics['rmse']:.4f}"
    )

    xy = np.vstack([t, p])

    if xy.shape[1] < 3:
        raise RuntimeError(f"Not enough points for KDE: {xy.shape[1]}")

    print(f"[KDE] {target.upper()} {split}: {xy.shape[1]} points")
    z = gaussian_kde(xy, bw_method=cfg["bw_method"])(xy)

    idx = z.argsort()
    t_sorted = t[idx]
    p_sorted = p[idx]
    z_sorted = z[idx]

    z_norm = (
        (z_sorted - global_z_min)
        / (global_z_max - global_z_min + 1e-10)
    )
    z_norm = np.clip(z_norm, 0.0, 1.0)

    cmap_custom = mcolors.LinearSegmentedColormap.from_list(
        f"{target}_{split}_cmap",
        cfg["hex_colors"],
    )

    colors_rgba = cmap_custom(z_norm)
    colors_rgba[:, 3] = 0.7 * z_norm + 0.2

    ax.scatter(
        t_sorted,
        p_sorted,
        c=colors_rgba,
        s=cfg["dot_size"],
        edgecolors="none",
    )

    ax.plot(
        lims,
        lims,
        color="#333333",
        linestyle=":",
        alpha=0.7,
        linewidth=1.5,
        zorder=0,
    )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, linestyle=":", alpha=0.25)
    ax.tick_params(axis="both", which="major", labelsize=11)

    if show_labels:
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

    if show_title:
        ax.set_title(
            f"{target.upper()} {split.capitalize()}\n"
            f"Median APE = {metrics['medape']:.3f}%",
            fontsize=12,
            pad=10,
        )

    return metrics, cmap_custom


def plot_single_full_and_pure(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    target: str,
    split: str,
    fig_out_dir: Path,
    cfg: dict,
    dpi: int,
    global_z_min: float,
    global_z_max: float,
) -> None:
    plt.rcParams["font.sans-serif"] = [cfg["font"]]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7.8, 6.2))

    _, cmap_custom = kde_scatter_on_ax(
        ax=ax,
        true_values=true_values,
        pred_values=pred_values,
        target=target,
        split=split,
        cfg=cfg,
        global_z_min=global_z_min,
        global_z_max=global_z_max,
        show_title=True,
        show_labels=True,
    )

    sm = plt.cm.ScalarMappable(
        cmap=cmap_custom,
        norm=plt.Normalize(
            vmin=global_z_min,
            vmax=global_z_max,
        ),
    )
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Point density", rotation=270, labelpad=15, fontsize=11)

    full_base = fig_out_dir / f"fig3a_{target}_{split}_full"
    save_figure(fig, full_base, dpi=dpi, transparent=False)
    plt.close(fig)

    fig_pure, ax_pure = plt.subplots(figsize=(6, 6))

    kde_scatter_on_ax(
        ax=ax_pure,
        true_values=true_values,
        pred_values=pred_values,
        target=target,
        split=split,
        cfg=cfg,
        global_z_min=global_z_min,
        global_z_max=global_z_max,
        show_title=False,
        show_labels=False,
    )

    ax_pure.set_axis_off()

    pure_base = fig_out_dir / f"fig3a_{target}_{split}_pure"
    save_figure(fig_pure, pure_base, dpi=dpi, transparent=True)
    plt.close(fig_pure)

    print(f"[SAVE] {target.upper()} {split} figures saved.")


def plot_combined_2x2(
    predictions: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
    fig_out_dir: Path,
    cfg: dict,
    dpi: int,
    global_z_min: float,
    global_z_max: float,
) -> None:
    plt.rcParams["font.sans-serif"] = [cfg["font"]]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    layout = [
        ("soh", "train", axes[0, 0]),
        ("soh", "test", axes[0, 1]),
        ("soc", "train", axes[1, 0]),
        ("soc", "test", axes[1, 1]),
    ]

    for target, split, ax in layout:
        t, p = predictions[(target, split)]
        kde_scatter_on_ax(
            ax=ax,
            true_values=t,
            pred_values=p,
            target=target,
            split=split,
            cfg=cfg,
            global_z_min=global_z_min,
            global_z_max=global_z_max,
            show_title=True,
            show_labels=True,
        )

    fig.tight_layout()

    out_base = fig_out_dir / "fig3a_prediction_scatter_2x2"
    save_figure(fig, out_base, dpi=dpi, transparent=False)
    plt.close(fig)

    print("[SAVE] Combined 2x2 figure saved.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Figure 3a directly from further-analysis prediction CSVs."
    )

    parser.add_argument(
        "--train_csv",
        type=str,
        default=str(DEFAULT_TRAIN_CSV),
        help="Path to train_predictions_for_scatter.csv.",
    )

    parser.add_argument(
        "--test_csv",
        type=str,
        default=str(DEFAULT_TEST_CSV),
        help="Path to test_predictions_per_sample.csv.",
    )

    parser.add_argument(
        "--fig_out_dir",
        type=str,
        default=str(DEFAULT_FIG_OUT_DIR),
        help="Directory to save Figure 3a outputs.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure dpi.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    fig_out_dir = ensure_dir(args.fig_out_dir)

    print("[FIGURE] Figure 3a prediction scatter")
    print("[MODE] Direct CSV plotting; no model inference")
    print(f"[TRAIN CSV] {train_csv}")
    print(f"[TEST CSV]  {test_csv}")
    print(f"[FIGURES]  {fig_out_dir}")

    predictions = build_predictions(
        train_csv=train_csv,
        test_csv=test_csv,
    )

    global_z_min, global_z_max = compute_global_kde_range(
        predictions=predictions,
        cfg=PLOT_CONFIG,
    )

    for target in ["soh", "soc"]:
        for split in ["train", "test"]:
            true_values, pred_values = predictions[(target, split)]

            plot_single_full_and_pure(
                true_values=true_values,
                pred_values=pred_values,
                target=target,
                split=split,
                fig_out_dir=fig_out_dir,
                cfg=PLOT_CONFIG,
                dpi=int(args.dpi),
                global_z_min=global_z_min,
                global_z_max=global_z_max,
            )

    plot_combined_2x2(
        predictions=predictions,
        fig_out_dir=fig_out_dir,
        cfg=PLOT_CONFIG,
        dpi=int(args.dpi),
        global_z_min=global_z_min,
        global_z_max=global_z_max,
    )

    print("\n[DONE] Figure 3a prediction scatter plots finished.")
    print(f"[FIGURES] {fig_out_dir}")


if __name__ == "__main__":
    main()
