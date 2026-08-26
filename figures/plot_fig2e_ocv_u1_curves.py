# -*- coding: utf-8 -*-
"""
plot_fig2e_ocv_u1_curves.py

Figure 2e:
OCV-like U1 curves across SOC for all material-capacity groups.

The script reads the first Excel file in each material-capacity folder,
extracts U1 values from SOC sheets for one representative cell ID, and
plots U1 as a function of SOC.

Default input:
    data/

Default output:
    results/figures/main/fig2e/fig2e_ocv_u1_curves.png

Optional clean output:
    results/figures/main/fig2e/fig2e_ocv_u1_curves_clean.png
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Default configuration
# =============================================================================
DEFAULT_DATA_ROOT = "data"
DEFAULT_OUTPUT_DIR = "results/figures/main/fig2e"

FOLDERS = [
    "10Ah LMO",
    "15Ah NMC",
    "21Ah NMC",
    "24Ah LMO",
    "25Ah LMO",
    "26Ah LMO",
    "35Ah LFP",
    "68Ah LFP",
]

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

LINE_STYLES = ["-", "-", "-", "-", "--", "--", "--", "--"]


# =============================================================================
# Style
# =============================================================================
def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica"],
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "lines.linewidth": 2.2,
            "savefig.dpi": 600,
            "savefig.format": "png",
        }
    )


# =============================================================================
# Argument parser
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 2e OCV/U1 curves."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing material-capacity folders.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output figures.",
    )

    parser.add_argument(
        "--save_clean",
        action="store_true",
        help="Also save a clean line-only transparent version.",
    )

    return parser.parse_args()


# =============================================================================
# Data extraction
# =============================================================================
def parse_soc_sheets(sheet_names: List[str]) -> dict[int, str]:
    """Map SOC value to sheet name."""
    soc_map = {}

    for sheet in sheet_names:
        match = re.search(r"SOC\s*(\d+)", sheet, re.I)
        if match:
            soc_map[int(match.group(1))] = sheet

    return soc_map


def find_representative_id(excel_path: str, soc_map: dict[int, str]):
    """Find one valid ID from the highest available SOC sheet."""
    for soc_value in sorted(soc_map.keys(), reverse=True):
        df_check = pd.read_excel(excel_path, sheet_name=soc_map[soc_value])

        if not df_check.empty and "ID" in df_check.columns:
            return df_check["ID"].iloc[0]

    return None


def extract_folder_curve(
    data_root: str,
    folder: str,
    folder_index: int,
) -> Tuple[List[int], List[float], str, int] | None:
    """
    Extract one U1-SOC curve from one material-capacity folder.
    """
    folder_path = os.path.join(data_root, folder)

    if not os.path.exists(folder_path):
        print(f"[WARN] Missing folder: {folder_path}")
        return None

    files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]

    if not files:
        print(f"[WARN] No Excel file found in: {folder_path}")
        return None

    excel_path = os.path.join(folder_path, files[0])

    try:
        excel_file = pd.ExcelFile(excel_path)
        soc_map = parse_soc_sheets(excel_file.sheet_names)

        if not soc_map:
            print(f"[WARN] No SOC sheets found in: {excel_path}")
            return None

        target_id = find_representative_id(excel_path, soc_map)

        if target_id is None:
            print(f"[WARN] No valid ID found in: {excel_path}")
            return None

        x_data = []
        y_data = []

        for soc_value in sorted(soc_map.keys()):
            df = pd.read_excel(excel_path, sheet_name=soc_map[soc_value])

            if "ID" not in df.columns or "U1" not in df.columns:
                continue

            row = df[df["ID"] == target_id]

            if not row.empty:
                x_data.append(soc_value)
                y_data.append(float(row["U1"].iloc[0]))

        if not x_data:
            print(f"[WARN] No U1 data extracted from: {excel_path}")
            return None

        return x_data, y_data, folder, folder_index

    except Exception as exc:
        print(f"[WARN] Skipping {folder}: {exc}")
        return None


def load_ocv_curves(data_root: str):
    """Load all OCV/U1 curves."""
    all_plot_data = []

    for idx, folder in enumerate(FOLDERS):
        item = extract_folder_curve(
            data_root=data_root,
            folder=folder,
            folder_index=idx,
        )

        if item is not None:
            all_plot_data.append(item)

    if not all_plot_data:
        raise RuntimeError("No valid OCV/U1 data extracted. Please check data path and Excel sheets.")

    print(f"[DATA] Loaded {len(all_plot_data)} OCV/U1 curves.")
    return all_plot_data


# =============================================================================
# Plotting
# =============================================================================
def plot_ocv_full(all_plot_data, output_dir: str) -> None:
    """Save the full annotated OCV/U1 curve figure."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 3.8))

    for x, y, label, i in all_plot_data:
        ax.plot(
            x,
            y,
            color=PROFESSIONAL_COLORS[i],
            linestyle=LINE_STYLES[i],
            label=label,
        )

    ax.set_xlabel("State of Charge (%)", fontweight="bold")
    ax.set_ylabel(r"Potential $U_1$ (V)", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=7,
        frameon=False,
    )

    save_path = os.path.join(output_dir, "fig2e_ocv_u1_curves.png")
    fig.savefig(
        save_path,
        bbox_inches="tight",
        transparent=False,
        facecolor="white",
    )
    plt.close(fig)

    print(f"[OK] Saved: {save_path}")


def plot_ocv_clean(all_plot_data, output_dir: str) -> None:
    """Save the clean line-only transparent OCV/U1 curve figure."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 3.8))

    for x, y, label, i in all_plot_data:
        ax.plot(
            x,
            y,
            color=PROFESSIONAL_COLORS[i],
            linestyle=LINE_STYLES[i],
        )

    ax.axis("off")

    save_path = os.path.join(output_dir, "fig2e_ocv_u1_curves_clean.png")
    fig.savefig(
        save_path,
        bbox_inches="tight",
        transparent=True,
        pad_inches=0,
    )
    plt.close(fig)

    print(f"[OK] Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    set_plot_style()

    all_plot_data = load_ocv_curves(args.data_root)

    plot_ocv_full(
        all_plot_data=all_plot_data,
        output_dir=args.output_dir,
    )

    if args.save_clean:
        plot_ocv_clean(
            all_plot_data=all_plot_data,
            output_dir=args.output_dir,
        )

    print("[DONE] Figure 2e OCV/U1 curve generation complete.")


if __name__ == "__main__":
    main()