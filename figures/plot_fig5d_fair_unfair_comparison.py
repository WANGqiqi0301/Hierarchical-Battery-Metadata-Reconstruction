# -*- coding: utf-8 -*-
"""
plot_fig5d_fair_enhanced_benchmark_FIXED.py

Figure 5d:
Fair vs enhanced benchmark comparison.

IMPORTANT
---------
Benchmark values are read DIRECTLY from:
    results/benchmark/benchmark_comparison_summary.csv

and ONLY from these exact metric columns:
    test_material_acc
    test_soc_medae_raw
    test_soh_medae_raw

No fallback to material_acc, test_cls_acc, cls_acc, MedAPE, or MAPE is allowed.

Outputs:
    results/figures/main/fig5d/fig5d_fair_enhanced_benchmark.png
    results/figures/main/fig5d/fig5d_fair_enhanced_benchmark_pure.png
    results/figures/main/fig5d/fig5d_plot_data.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "benchmark"
    / "benchmark_comparison_summary.csv"
)

# Proposed is not present in benchmark_comparison_summary.csv,
# so its reference metrics are computed from the proposed-framework
# per-sample prediction file.
PROPOSED_PRED_CSV = (
    PROJECT_ROOT
    / "results"
    / "proposed_framework"
    / "further_analysis"
    / "tables"
    / "test_predictions_per_sample.csv"
)

SAVE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "main"
    / "fig5d"
)

SAVE_NAME = "fig5d_fair_enhanced_benchmark.png"
PURE_SAVE_NAME = "fig5d_fair_enhanced_benchmark_pure.png"
PLOT_DATA_NAME = "fig5d_plot_data.csv"

SAVE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Fixed plotting order
# =============================================================================
METHODS = [
    "TabNet",
    "XGBoost",
    "NODE",
    "FT-Transformer",
    "Proposed",
]

METHOD_MAP = {
    "tabnet": "TabNet",
    "xgboost": "XGBoost",
    "xgb": "XGBoost",
    "node": "NODE",
    "ft_transformer": "FT-Transformer",
    "ft-transformer": "FT-Transformer",
    "fttransformer": "FT-Transformer",
}

SETTING_MAP = {
    "fair": "Fair",
    "enhanced": "Enhanced",
}


# =============================================================================
# Helpers
# =============================================================================
def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
    )


def normalize_method_name(x: str) -> str:
    key = str(x).strip().lower()
    return METHOD_MAP.get(key, str(x))


def normalize_setting_name(x: str) -> str:
    key = str(x).strip().lower()
    return SETTING_MAP.get(key, str(x))


def get_value(
    df: pd.DataFrame,
    method: str,
    setting: str,
    metric: str,
) -> float:
    sub = df[
        (df["method"] == method)
        & (df["setting"] == setting)
    ]

    if sub.empty:
        raise RuntimeError(
            f"Missing plotting row: method={method}, setting={setting}"
        )

    return float(sub.iloc[0][metric])


# =============================================================================
# Read benchmark summary -- STRICTLY from test_* columns
# =============================================================================
def load_benchmark_summary() -> pd.DataFrame:
    print("=" * 88)
    print("[READ SUMMARY CSV]")
    print(INPUT_CSV)
    print("=" * 88)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Benchmark summary CSV not found:\n{INPUT_CSV}"
        )

    raw = pd.read_csv(INPUT_CSV)

    # These are the ONLY columns used for Figure 5d benchmark metrics.
    required_cols = [
        "model",
        "setting",
        "test_material_acc",
        "test_soc_medae_raw",
        "test_soh_medae_raw",
    ]

    missing_cols = [
        c for c in required_cols
        if c not in raw.columns
    ]

    if missing_cols:
        raise RuntimeError(
            "Missing required columns in benchmark summary:\n"
            f"{missing_cols}\n\n"
            f"Available columns:\n{raw.columns.tolist()}"
        )

    # Print the source values exactly as they exist in your summary CSV.
    source_cols = [
        "model",
        "setting",
    ]

    if "test_cls_acc" in raw.columns:
        source_cols.append("test_cls_acc")

    source_cols.extend(
        [
            "test_material_acc",
            "test_soc_medae_raw",
            "test_soh_medae_raw",
        ]
    )

    print("\n[SOURCE VALUES IN benchmark_comparison_summary.csv]")
    print(
        raw[source_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )

    # -------------------------------------------------------------------------
    # IMPORTANT: direct column access. No fallback of any kind.
    # -------------------------------------------------------------------------
    benchmark = pd.DataFrame(
        {
            "method": raw["model"].map(normalize_method_name),
            "setting": raw["setting"].map(normalize_setting_name),

            # EXACTLY test_material_acc
            "material_accuracy": pd.to_numeric(
                raw["test_material_acc"],
                errors="raise",
            ),

            # EXACTLY test_soc_medae_raw
            "soc_medae": pd.to_numeric(
                raw["test_soc_medae_raw"],
                errors="raise",
            ),

            # EXACTLY test_soh_medae_raw
            "soh_medae": pd.to_numeric(
                raw["test_soh_medae_raw"],
                errors="raise",
            ),
        }
    )

    benchmark = benchmark[
        benchmark["method"].isin(METHODS[:-1])
        & benchmark["setting"].isin(["Fair", "Enhanced"])
    ].copy()

    # Verify that every value copied into plotting data is identical to
    # the corresponding test_material_acc from the source CSV.
    for _, row in benchmark.iterrows():
        method_raw_names = {
            "TabNet": ["tabnet"],
            "XGBoost": ["xgboost", "xgb"],
            "NODE": ["node"],
            "FT-Transformer": ["ft_transformer", "ft-transformer", "fttransformer"],
        }[row["method"]]

        source = raw[
            raw["model"].astype(str).str.strip().str.lower().isin(method_raw_names)
            & (
                raw["setting"].astype(str).str.strip().str.lower()
                == row["setting"].lower()
            )
        ]

        if source.empty:
            raise RuntimeError(
                f"Cannot verify source row for {row['method']} / {row['setting']}"
            )

        source_material = float(source.iloc[0]["test_material_acc"])

        if not np.isclose(
            float(row["material_accuracy"]),
            source_material,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Internal material-accuracy mismatch detected."
            )

    print("\n[CHECK] Material accuracy source = test_material_acc ONLY")
    print(
        benchmark.to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )

    return benchmark


# =============================================================================
# Proposed-framework reference
# =============================================================================
def load_proposed_result() -> pd.DataFrame:
    print("\n" + "=" * 88)
    print("[READ PROPOSED PER-SAMPLE PREDICTIONS]")
    print(PROPOSED_PRED_CSV)
    print("=" * 88)

    if not PROPOSED_PRED_CSV.exists():
        raise FileNotFoundError(
            f"Proposed prediction CSV not found:\n{PROPOSED_PRED_CSV}"
        )

    pred = pd.read_csv(PROPOSED_PRED_CSV)

    required_cols = [
        "true_label",
        "pred_label",
        "soc_true",
        "soc_pred",
        "soh_true",
        "soh_pred",
    ]

    missing_cols = [
        c for c in required_cols
        if c not in pred.columns
    ]

    if missing_cols:
        raise RuntimeError(
            f"Missing proposed prediction columns: {missing_cols}"
        )

    true_material = (
        pred["true_label"]
        .astype(str)
        .str.split("_")
        .str[0]
        .to_numpy()
    )

    pred_material = (
        pred["pred_label"]
        .astype(str)
        .str.split("_")
        .str[0]
        .to_numpy()
    )

    soc_true = pd.to_numeric(
        pred["soc_true"],
        errors="coerce",
    ).to_numpy(dtype=float)

    soc_pred = pd.to_numeric(
        pred["soc_pred"],
        errors="coerce",
    ).to_numpy(dtype=float)

    soh_true = pd.to_numeric(
        pred["soh_true"],
        errors="coerce",
    ).to_numpy(dtype=float)

    soh_pred = pd.to_numeric(
        pred["soh_pred"],
        errors="coerce",
    ).to_numpy(dtype=float)

    soc_valid = np.isfinite(soc_true) & np.isfinite(soc_pred)
    soh_valid = np.isfinite(soh_true) & np.isfinite(soh_pred)

    proposed = pd.DataFrame(
        [
            {
                "method": "Proposed",
                "setting": "Proposed",
                "material_accuracy": float(
                    np.mean(true_material == pred_material)
                ),
                "soc_medae": float(
                    np.median(
                        np.abs(
                            soc_pred[soc_valid]
                            - soc_true[soc_valid]
                        )
                    )
                ),
                "soh_medae": float(
                    np.median(
                        np.abs(
                            soh_pred[soh_valid]
                            - soh_true[soh_valid]
                        )
                    )
                ),
            }
        ]
    )

    print("\n[PROPOSED VALUES]")
    print(
        proposed.to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )

    return proposed


# =============================================================================
# Build final plotting table
# =============================================================================
def build_plot_data() -> pd.DataFrame:
    benchmark = load_benchmark_summary()
    proposed = load_proposed_result()

    plot_df = pd.concat(
        [benchmark, proposed],
        ignore_index=True,
    )

    required_pairs = []
    for method in METHODS[:-1]:
        required_pairs.append((method, "Fair"))
        required_pairs.append((method, "Enhanced"))
    required_pairs.append(("Proposed", "Proposed"))

    missing_pairs = []
    for method, setting in required_pairs:
        sub = plot_df[
            (plot_df["method"] == method)
            & (plot_df["setting"] == setting)
        ]
        if sub.empty:
            missing_pairs.append((method, setting))

    if missing_pairs:
        raise RuntimeError(
            "Missing method/setting pairs:\n"
            + "\n".join(
                f"  {m}, {s}"
                for m, s in missing_pairs
            )
        )

    output_csv = SAVE_DIR / PLOT_DATA_NAME
    plot_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 88)
    print("[FINAL DATA SENT TO MATPLOTLIB]")
    print("=" * 88)
    print(
        plot_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )
    print(f"\n[SAVED PLOT DATA] {output_csv}")

    return plot_df


# =============================================================================
# Plot configuration
# =============================================================================
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    }
)

METRICS = [
    (
        "material_accuracy",
        "Material accuracy (%)",
        "higher",
    ),
    (
        "soc_medae",
        "SOC MedAE (%)",
        "lower",
    ),
    (
        "soh_medae",
        "SOH MedAE (%)",
        "lower",
    ),
]

COLORS = {
    "Fair": "#BDBBBB",
    "Enhanced": "#4E6E81",
    "Proposed": "#D95F02",
}

EDGE_COLOR = "#222222"


# =============================================================================
# Draw figure
# =============================================================================
def draw_figure(
    plot_df: pd.DataFrame,
    pure: bool,
) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7, 2.35),
    )

    x = np.arange(len(METHODS))
    width = 0.35

    for ax_idx, (metric, ylabel, direction) in enumerate(METRICS):
        ax = axes[ax_idx]

        fair_vals = []
        enhanced_vals = []

        for method in METHODS:
            if method == "Proposed":
                fair_vals.append(np.nan)
                enhanced_vals.append(np.nan)
            else:
                fair_vals.append(
                    get_value(
                        plot_df,
                        method,
                        "Fair",
                        metric,
                    )
                )
                enhanced_vals.append(
                    get_value(
                        plot_df,
                        method,
                        "Enhanced",
                        metric,
                    )
                )

        fair_vals = np.asarray(fair_vals, dtype=float)
        enhanced_vals = np.asarray(enhanced_vals, dtype=float)

        proposed_val = get_value(
            plot_df,
            "Proposed",
            "Proposed",
            metric,
        )

        # test_material_acc is stored as 0-1 fraction.
        # Convert to percentage ONLY at the plotting stage.
        if metric == "material_accuracy":
            fair_plot = fair_vals * 100.0
            enhanced_plot = enhanced_vals * 100.0
            proposed_plot = proposed_val * 100.0
        else:
            fair_plot = fair_vals
            enhanced_plot = enhanced_vals
            proposed_plot = proposed_val

        ax.bar(
            x - width / 2,
            fair_plot,
            width=width,
            color=COLORS["Fair"],
            edgecolor=EDGE_COLOR,
            linewidth=0,
            label="Existing models",
            zorder=2,
        )

        ax.bar(
            x + width / 2,
            enhanced_plot,
            width=width,
            color=COLORS["Enhanced"],
            edgecolor=EDGE_COLOR,
            linewidth=0,
            label="Existing models + controlled upstream inputs",
            zorder=2,
        )

        ax.bar(
            x[-1],
            proposed_plot,
            width=0.42,
            color=COLORS["Proposed"],
            edgecolor=EDGE_COLOR,
            linewidth=0,
            label="Proposed",
            zorder=3,
        )

        ax.axhline(
            proposed_plot,
            color=COLORS["Proposed"],
            lw=1.1,
            ls="--",
            alpha=0.8,
            zorder=1,
        )

        values = np.concatenate(
            [
                fair_plot[np.isfinite(fair_plot)],
                enhanced_plot[np.isfinite(enhanced_plot)],
                np.asarray([proposed_plot], dtype=float),
            ]
        )

        if metric == "material_accuracy":
            # Keep the material panel focused on the actual accuracy range.
            ymin = max(0.0, float(np.min(values)) - 5.0)
            ymax = min(100.0, float(np.max(values)) + 3.0)
            if ymax <= ymin:
                ymax = min(100.0, ymin + 10.0)
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(0, float(np.max(values)) * 1.25)

        if not pure:
            add_panel_label(
                ax,
                chr(ord("a") + ax_idx),
            )

            ax.set_ylabel(ylabel)

            direction_text = (
                "Higher is better"
                if direction == "higher"
                else "Lower is better"
            )

            ax.text(
                0.98,
                0.08,
                direction_text,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="#555555",
            )

            ax.set_xticks(x)
            ax.set_xticklabels(
                METHODS,
                rotation=35,
                ha="right",
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.grid(
                axis="y",
                linestyle="--",
                linewidth=0.45,
                alpha=0.35,
                zorder=0,
            )

        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(False)

            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.set_axis_off()

    if not pure:
        handles, labels = axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))

        fig.legend(
            unique.values(),
            unique.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=3,
            frameon=False,
            fontsize=8,
            handlelength=1.2,
            columnspacing=1.1,
        )

        plt.tight_layout(w_pad=1.25)

        save_path = SAVE_DIR / SAVE_NAME
        fig.savefig(
            save_path,
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"[OK] Saved: {save_path}")

    else:
        plt.subplots_adjust(
            left=0,
            right=1,
            bottom=0,
            top=1,
            wspace=0.08,
        )

        save_path = SAVE_DIR / PURE_SAVE_NAME
        fig.savefig(
            save_path,
            dpi=600,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)

        print(f"[OK] Saved pure: {save_path}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    plot_df = build_plot_data()

    draw_figure(
        plot_df=plot_df,
        pure=False,
    )

    draw_figure(
        plot_df=plot_df,
        pure=True,
    )

    print("\n[DONE] Figure 5d generated.")


if __name__ == "__main__":
    main()
