# -*- coding: utf-8 -*-
"""
plot_fig2e_delta_u_distribution_soc20.py

Figure 2e:
Delta-U distribution at SOC 20 for all material-capacity groups.

The script reads the first Excel file in each material-capacity folder,
finds the SOC 20 sheet, extracts U1-U41 for one representative cell ID,
computes Delta U_i = U_i - U_{i-1}, and plots Delta-U curves.

Default input:
    data/

Default output:
    results/figures/main/fig2e/fig2e_delta_u_distribution_soc20.png

Optional clean output:
    results/figures/main/fig2e/fig2e_delta_u_distribution_soc20_clean.png
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

import numpy as np
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

DENSE_DASH = (0, (2, 0.7))
LINE_STYLES = ["-", "-", "-", "-", DENSE_DASH, DENSE_DASH, DENSE_DASH, DENSE_DASH]


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
            "lines.linewidth": 1.5,
            "savefig.dpi": 600,
        }
    )


# =============================================================================
# Argument parser
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 2e Delta-U distribution at SOC 20."
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
def find_soc20_sheet(sheet_names: List[str]) -> str | None:
    """Find a sheet matching SOC 20, such as SOC20 or SOC 20."""
    for sheet in sheet_names:
        if re.search(r"SOC\s*20$", sheet, re.I):
            return sheet
    return None


def extract_delta_u_from_folder(
    data_root: str,
    folder: str,
    folder_index: int,
) -> Tuple[List[int], np.ndarray, str, int] | None:
    """Extract Delta-U curve from the SOC 20 sheet of one folder."""
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
        target_sheet = find_soc20_sheet(excel_file.sheet_names)

        if target_sheet is None:
            print(f"[WARN] Skip {folder}: SOC 20 sheet not found.")
            return None

        df = pd.read_excel(excel_path, sheet_name=target_sheet)

        if df.empty:
            print(f"[WARN] Skip {folder}: SOC 20 sheet is empty.")
            return None

        if "ID" not in df.columns:
            print(f"[WARN] Skip {folder}: ID column not found.")
            return None

        target_id = df["ID"].iloc[0]
        row = df[df["ID"] == target_id]

        if row.empty:
            print(f"[WARN] Skip {folder}: representative ID not found in sheet.")
            return None

        u_cols = [f"U{i}" for i in range(1, 42)]
        available_cols = [c for c in u_cols if c in row.columns]

        if len(available_cols) < 2:
            print(f"[WARN] Skip {folder}: insufficient U columns.")
            return None

        u_values = row[available_cols].values.flatten().astype(float)

        delta_u = np.diff(u_values)
        delta_index = [
            int(re.findall(r"\d+", c)[0])
            for c in available_cols[1:]
        ]

        return delta_index, delta_u, folder, folder_index

    except Exception as exc:
        print(f"[WARN] Error processing {folder}: {exc}")
        return None


def load_delta_u_data(data_root: str):
    """Load Delta-U distribution data from all folders."""
    all_data = []

    for idx, folder in enumerate(FOLDERS):
        item = extract_delta_u_from_folder(
            data_root=data_root,
            folder=folder,
            folder_index=idx,
        )
        if item is not None:
            all_data.append(item)

    if not all_data:
        raise RuntimeError(
            "No valid data extracted. Please check whether the Excel files contain SOC 20 and U1-U41 columns."
        )

    print(f"[DATA] Loaded {len(all_data)} Delta-U curves.")
    return all_data


# =============================================================================
# Plotting
# =============================================================================
def plot_delta_u_full(all_data, output_dir: str) -> None:
    """Save the full annotated Delta-U figure."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.axhline(
        0,
        color="black",
        linewidth=2,
        alpha=0.3,
    )

    for x, y, label, i in all_data:
        ax.plot(
            x,
            y,
            color=PROFESSIONAL_COLORS[i],
            linestyle=LINE_STYLES[i],
            label=label,
        )

    ax.set_xlabel(r"$\Delta$U index ($U_i - U_{i-1}$)", fontweight="bold")
    ax.set_ylabel(r"$\Delta$U value (V)", fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.4,
    )

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=7,
        frameon=False,
    )

    save_path = os.path.join(output_dir, "fig2e_delta_u_distribution_soc20.png")
    fig.savefig(
        save_path,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(f"[OK] Saved: {save_path}")


def plot_delta_u_clean(all_data, output_dir: str) -> None:
    """Save the clean line-only Delta-U figure."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))

    for x, y, label, i in all_data:
        ax.plot(
            x,
            y,
            color=PROFESSIONAL_COLORS[i],
            linestyle=LINE_STYLES[i],
        )

    ax.axis("off")

    save_path = os.path.join(output_dir, "fig2e_delta_u_distribution_soc20_clean.png")
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

    all_data = load_delta_u_data(args.data_root)

    plot_delta_u_full(
        all_data=all_data,
        output_dir=args.output_dir,
    )

    if args.save_clean:
        plot_delta_u_clean(
            all_data=all_data,
            output_dir=args.output_dir,
        )

    print("[DONE] Figure 2e Delta-U distribution generation complete.")


if __name__ == "__main__":
    main()