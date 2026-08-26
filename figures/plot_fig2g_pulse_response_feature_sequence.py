# -*- coding: utf-8 -*-
"""
plot_fig2g_pulse_response_feature_sequence.py

Figure 2g:
Representative pulse-voltage response and ordered U1-U41 feature sequence.

A representative pulse-voltage response is shown together with the
corresponding pulse protocol. Starting from the open-circuit voltage U1,
sequential voltage samples U2-U41 are extracted from pulse operations at
increasing C-rates from 0.5C to 2.5C. Each C-rate group consists of a
positive pulse, relaxation, negative pulse, and subsequent relaxation.

Default input:
    results/figures/main/fig2d/fig2d_soc_soh_data_cache.csv

Default output:
    results/figures/main/fig2g/fig2g_pulse_response_feature_sequence.png

Optional clean output:
    results/figures/main/fig2g/fig2g_pulse_response_feature_sequence_clean.png
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Default configuration
# =============================================================================
DEFAULT_OUTPUT_DIR = r"results/figures/main/fig2g"
DEFAULT_CACHE_FILE = r"results/figures/main/fig2d/fig2d_soc_soh_data_cache.csv"

DEFAULT_TARGET_KEYWORD = "LMO_25"

DPI = 600


# =============================================================================
# Style
# =============================================================================
def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": DPI,
        }
    )


# =============================================================================
# Arguments
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 2g pulse-response feature-sequence schematic."
    )

    parser.add_argument(
        "--cache_file",
        type=str,
        default=DEFAULT_CACHE_FILE,
        help="CSV cache containing U1-U41 columns.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output figures.",
    )

    parser.add_argument(
        "--target_keyword",
        type=str,
        default=DEFAULT_TARGET_KEYWORD,
        help="Keyword used to select one representative row from text columns.",
    )

    parser.add_argument(
        "--no_synthetic",
        action="store_true",
        help="Disable synthetic fallback if no real U1-U41 row is found.",
    )

    parser.add_argument(
        "--save_clean",
        action="store_true",
        help="Also save a clean version without labels and axes.",
    )

    return parser.parse_args()


# =============================================================================
# Data loading
# =============================================================================
def find_u_columns(df: pd.DataFrame) -> Optional[list[str]]:
    """Find U1-U41 columns from common naming conventions."""
    possible_sets = [
        [f"U{i}" for i in range(1, 42)],
        [f"U_{i}" for i in range(1, 42)],
        [f"u{i}" for i in range(1, 42)],
        [f"u_{i}" for i in range(1, 42)],
    ]

    for cols in possible_sets:
        if all(col in df.columns for col in cols):
            return cols

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 41:
        return numeric_cols[:41]

    return None


def load_real_u41(
    cache_file: str,
    target_keyword: Optional[str],
) -> Tuple[Optional[np.ndarray], Optional[list[str]]]:
    """Load one real U1-U41 sample from a CSV cache."""
    if not os.path.exists(cache_file):
        print(f"[WARN] Cache file not found: {cache_file}")
        return None, None

    df = pd.read_csv(cache_file)
    u_cols = find_u_columns(df)

    if u_cols is None:
        print("[WARN] Cannot identify U1-U41 columns.")
        return None, None

    df_selected = df.copy()

    if target_keyword:
        text_cols = df_selected.select_dtypes(include=["object"]).columns.tolist()
        mask = np.zeros(len(df_selected), dtype=bool)

        for col in text_cols:
            mask |= (
                df_selected[col]
                .astype(str)
                .str.contains(target_keyword, case=False, na=False)
                .to_numpy()
            )

        if mask.any():
            df_selected = df_selected.loc[mask].reset_index(drop=True)
        else:
            print(f"[WARN] Target keyword not found: {target_keyword}. Using first valid row.")

    U_all = df_selected[u_cols].to_numpy(dtype=float)
    valid = np.all(np.isfinite(U_all), axis=1)

    if not valid.any():
        print("[WARN] No finite U1-U41 row found.")
        return None, None

    return U_all[valid][0], u_cols


def make_synthetic_u41() -> np.ndarray:
    """Create a synthetic U1-U41 sequence for schematic fallback."""
    U = [3.72]
    ocv = 3.72

    for amp in [0.5, 1.0, 1.5, 2.0, 2.5]:
        U.extend(
            [
                ocv + 0.010 * amp,
                ocv + 0.045 * amp,
                ocv + 0.030 * amp,
                ocv + 0.010 * amp,
                ocv - 0.010 * amp,
                ocv - 0.050 * amp,
                ocv - 0.025 * amp,
                ocv - 0.005 * amp,
            ]
        )

    return np.array(U, dtype=float)


# =============================================================================
# Layout helpers
# =============================================================================
def build_positions():
    """Build x positions and protocol annotation metadata."""
    x = [0.0]
    group_info = []
    operation_info = []

    current_x = 1.4

    c_rates = ["0.5C", "1C", "1.5C", "2C", "2.5C"]
    feature_ranges = [
        r"$U_2$–$U_9$",
        r"$U_{10}$–$U_{17}$",
        r"$U_{18}$–$U_{25}$",
        r"$U_{26}$–$U_{33}$",
        r"$U_{34}$–$U_{41}$",
    ]

    for c_rate, feature_range in zip(c_rates, feature_ranges):
        xs = []

        for _ in range(8):
            x.append(current_x)
            xs.append(current_x)
            current_x += 0.72

        group_info.append((xs[0], xs[-1], c_rate, feature_range))

        operation_info.extend(
            [
                (xs[0], xs[1], "+ pulse"),
                (xs[2], xs[3], "+ rest"),
                (xs[4], xs[5], "- pulse"),
                (xs[6], xs[7], "- rest"),
            ]
        )

        current_x += 0.82

    return np.array(x), group_info, operation_info


def add_simple_bracket(ax, x0, x1, y, text, h, fontsize=7) -> None:
    """Add a compact bracket annotation."""
    ax.plot(
        [x0, x0, x1, x1],
        [y, y + h, y + h, y],
        color="black",
        lw=0.7,
        clip_on=False,
    )

    ax.text(
        (x0 + x1) / 2,
        y + h * 1.35,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )


# =============================================================================
# Plotting
# =============================================================================
def plot_pulse_response_feature_sequence(
    U: np.ndarray,
    output_dir: str,
    save_name: str,
    annotated: bool = True,
) -> None:
    """Plot the compact pulse-response feature-sequence schematic."""
    os.makedirs(output_dir, exist_ok=True)

    x, group_info, operation_info = build_positions()

    fig = plt.figure(figsize=(25 / 2.54, 8 / 2.54))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[4.0, 0.9],
        hspace=0.08,
    )

    ax = fig.add_subplot(gs[0, 0])
    ax_protocol = fig.add_subplot(gs[1, 0], sharex=ax)

    # Background operation shading.
    for x0, x1, op in operation_info:
        if op == "+ pulse":
            color, alpha = "#F4A261", 0.16
        elif op == "- pulse":
            color, alpha = "#457B9D", 0.16
        else:
            color, alpha = "#BFC0C0", 0.10

        ax.axvspan(
            x0 - 0.18,
            x1 + 0.18,
            color=color,
            alpha=alpha,
            lw=0,
        )

    ax.axvspan(
        x[0] - 0.25,
        x[0] + 0.25,
        color="#7A9E7E",
        alpha=0.18,
        lw=0,
    )

    # Main voltage response curve.
    ax.plot(
        x,
        U,
        lw=1,
        marker="o",
        ms=2.4,
        color="#2F3A4AB2",
        zorder=3,
    )

    y_min = float(np.nanmin(U))
    y_max = float(np.nanmax(U))
    y_range = y_max - y_min + 1e-8

    ax.set_ylim(y_min - 0.18 * y_range, y_max + 0.26 * y_range)
    ax.set_xlim(x.min() - 0.6, x.max() + 0.6)

    if annotated:
        ax.text(
            x[0],
            U[0] + 0.04 * y_range,
            r"$U_1$",
            ha="center",
            va="bottom",
            fontsize=7,
        )

        bracket_y = y_max + 0.09 * y_range

        for x0, x1, c_rate, feature_range in group_info:
            add_simple_bracket(
                ax,
                x0 - 0.1,
                x1 + 0.1,
                bracket_y,
                c_rate,
                0.018 * y_range,
            )

            ax.text(
                (x0 + x1) / 2,
                y_min - 0.105 * y_range,
                feature_range,
                ha="center",
                va="top",
                fontsize=7,
            )

        ax.text(
            x[0],
            bracket_y + 0.035 * y_range,
            "OCV",
            ha="center",
            va="bottom",
            fontsize=7,
        )

        ax.set_ylabel("Voltage response")
        ax.set_title(
            r"Representative pulse-voltage response and ordered feature sequence ($U_1$–$U_{41}$)",
            fontsize=9,
            pad=12,
        )

    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

        for spine in ax.spines.values():
            spine.set_visible(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Protocol panel.
    ax_protocol.axhline(0, color="black", lw=0.45)

    protocol_lw = 2.2
    pos_y = 1.30
    rest_y = 0.00
    neg_y = -1.30

    for x0, x1, op in operation_info:
        if op == "+ pulse":
            y, color = pos_y, "#F4A261"
        elif op == "- pulse":
            y, color = neg_y, "#457B9D"
        else:
            y, color = rest_y, "#BFC0C0"

        ax_protocol.plot(
            [x0, x1],
            [y, y],
            lw=protocol_lw,
            solid_capstyle="round",
            color=color,
        )

    ax_protocol.set_facecolor("none")

    if annotated:
        ax_protocol.text(
            x[0],
            0.35,
            "OCV",
            ha="center",
            va="bottom",
            fontsize=7,
        )

        ax_protocol.set_yticks([neg_y, rest_y, pos_y])
        ax_protocol.set_yticklabels(["Negative", "Rest", "Positive"])

        ax_protocol.set_ylabel("Protocol")
        ax_protocol.set_xlabel("Feature sequence")

        group_centers = [(x0 + x1) / 2 for x0, x1, _, _ in group_info]
        group_labels = [c_rate for _, _, c_rate, _ in group_info]

        ax_protocol.set_xticks([x[0]] + group_centers)
        ax_protocol.set_xticklabels(["OCV"] + group_labels)

    else:
        ax_protocol.set_xticks([])
        ax_protocol.set_yticks([])
        ax_protocol.set_xlabel("")
        ax_protocol.set_ylabel("")

        for spine in ax_protocol.spines.values():
            spine.set_visible(False)

    ax_protocol.set_ylim(-1.75, 1.75)
    ax_protocol.spines["top"].set_visible(False)
    ax_protocol.spines["right"].set_visible(False)

    fig.tight_layout()

    save_path = os.path.join(output_dir, save_name)
    fig.savefig(
        save_path,
        bbox_inches="tight",
        dpi=DPI,
        transparent=True,
    )
    plt.close(fig)

    print(f"[OK] Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    set_plot_style()

    U, _ = load_real_u41(
        cache_file=args.cache_file,
        target_keyword=args.target_keyword,
    )

    if U is None:
        if args.no_synthetic:
            raise RuntimeError("No real U1-U41 sample found.")
        print("[INFO] Using synthetic U1-U41 sequence.")
        U = make_synthetic_u41()
    else:
        print("[INFO] Loaded real U1-U41 sample.")

    plot_pulse_response_feature_sequence(
        U=U,
        output_dir=args.output_dir,
        save_name="fig2g_pulse_response_feature_sequence.png",
        annotated=True,
    )

    if args.save_clean:
        plot_pulse_response_feature_sequence(
            U=U,
            output_dir=args.output_dir,
            save_name="fig2g_pulse_response_feature_sequence_clean.png",
            annotated=False,
        )

    print("[DONE] Figure 2g pulse-response feature-sequence schematic generated.")


if __name__ == "__main__":
    main()