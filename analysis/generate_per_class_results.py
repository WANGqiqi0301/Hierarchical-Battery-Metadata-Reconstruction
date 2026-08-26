# analysis/generate_per_class_results.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


# =============================================================================
# Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results"
    / "proposed_framework"
    / "further_analysis"
    / "tables"
    / "test_predictions_per_sample.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "per_class"
    / "per_class_results.csv"
)


# =============================================================================
# Class order
# =============================================================================

CLASS_ORDER = [
    "LFP_35Ah",
    "LFP_68Ah",
    "LMO_10Ah",
    "LMO_24Ah",
    "LMO_25Ah",
    "LMO_26Ah",
    "NMC_15Ah",
    "NMC_21Ah",
]

MATERIAL_ORDER = [
    "LFP",
    "LMO",
    "NMC",
]


# =============================================================================
# Metrics
# =============================================================================

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return np.nan

    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return np.nan

    return float(np.mean(np.abs(y_pred - y_true)))


def medae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return np.nan

    return float(np.median(np.abs(y_pred - y_true)))


def median_ape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return np.nan

    ape = (
        np.abs(y_pred - y_true)
        / np.maximum(np.abs(y_true), eps)
        * 100.0
    )

    return float(np.median(ape))


def accuracy(true_label, pred_label):
    if len(true_label) == 0:
        return np.nan

    true_arr = np.asarray(true_label).astype(str)
    pred_arr = np.asarray(pred_label).astype(str)

    return float(np.mean(true_arr == pred_arr) * 100.0)


# =============================================================================
# Summary
# =============================================================================

def summarize_one_group(df_group, group_name):
    return {
        "Group": group_name,
        "No. of evaluated records": int(len(df_group)),

        # classification
        "Class accuracy (%)": accuracy(
            df_group["true_label"],
            df_group["pred_label"],
        ),
        "Material accuracy (%)": accuracy(
            df_group["true_material"],
            df_group["pred_material"],
        ),

        # SOC
        "SOC RMSE (%)": rmse(
            df_group["soc_true"],
            df_group["soc_pred"],
        ),
        "SOC MAE (%)": mae(
            df_group["soc_true"],
            df_group["soc_pred"],
        ),
        "SOC MedAE (%)": medae(
            df_group["soc_true"],
            df_group["soc_pred"],
        ),
        "SOC MedAPE (%)": median_ape(
            df_group["soc_true"],
            df_group["soc_pred"],
        ),

        # SOH
        "SOH RMSE (%)": rmse(
            df_group["soh_true"],
            df_group["soh_pred"],
        ),
        "SOH MAE (%)": mae(
            df_group["soh_true"],
            df_group["soh_pred"],
        ),
        "SOH MedAE (%)": medae(
            df_group["soh_true"],
            df_group["soh_pred"],
        ),
        "SOH MedAPE (%)": median_ape(
            df_group["soh_true"],
            df_group["soh_pred"],
        ),
    }


# =============================================================================
# Main analysis
# =============================================================================

def generate_per_class_results(input_csv, output_csv):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    df = pd.read_csv(input_csv)

    required_columns = [
        "true_label",
        "pred_label",
        "soc_true",
        "soc_pred",
        "soh_true",
        "soh_pred",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    df = df[required_columns].copy()

    numeric_columns = [
        "soc_true",
        "soc_pred",
        "soh_true",
        "soh_pred",
    ]

    for c in numeric_columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    print(f"[DATA] Valid records: {len(df)}")

    # -------------------------------------------------------------------------
    # Derive material labels
    # -------------------------------------------------------------------------

    df["true_material"] = (
        df["true_label"]
        .astype(str)
        .str.split("_")
        .str[0]
    )

    df["pred_material"] = (
        df["pred_label"]
        .astype(str)
        .str.split("_")
        .str[0]
    )

    rows = []

    # =========================================================================
    # All records
    # =========================================================================

    rows.append(
        summarize_one_group(
            df,
            "All records",
        )
    )

    # =========================================================================
    # Complete 8-class groups
    # =========================================================================

    for cls in CLASS_ORDER:
        sub = df[
            df["true_label"].astype(str) == cls
        ]

        rows.append(
            summarize_one_group(
                sub,
                cls,
            )
        )

    # =========================================================================
    # Material groups
    # =========================================================================

    for mat in MATERIAL_ORDER:
        sub = df[
            df["true_material"] == mat
        ]

        rows.append(
            summarize_one_group(
                sub,
                mat,
            )
        )

    result = pd.DataFrame(rows)

    result = result.round(
        {
            "Class accuracy (%)": 1,
            "Material accuracy (%)": 1,

            "SOC RMSE (%)": 2,
            "SOC MAE (%)": 2,
            "SOC MedAE (%)": 2,
            "SOC MedAPE (%)": 2,

            "SOH RMSE (%)": 2,
            "SOH MAE (%)": 2,
            "SOH MedAE (%)": 2,
            "SOH MedAPE (%)": 2,
        }
    )

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    output_json = output_csv.with_suffix(".json")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            result.to_dict(orient="records"),
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )

    # -------------------------------------------------------------------------
    # Overall metrics: explicitly print the metrics used in the manuscript
    # -------------------------------------------------------------------------

    overall = result.iloc[0]

    print("\n" + "=" * 120)
    print("[OVERALL TEST METRICS]")
    print("=" * 120)
    print(f"Class accuracy:    {overall['Class accuracy (%)']:.1f}%")
    print(f"Material accuracy: {overall['Material accuracy (%)']:.1f}%")
    print(f"SOC MedAE:         {overall['SOC MedAE (%)']:.2f}%")
    print(f"SOH MedAE:         {overall['SOH MedAE (%)']:.2f}%")

    print("\n" + "=" * 120)
    print("[PER-CLASS RESULTS]")
    print("=" * 120)
    print(result.to_string(index=False))

    print(f"\n[SAVED CSV]  {output_csv}")
    print(f"[SAVED JSON] {output_json}")

    return result


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    generate_per_class_results(
        input_csv=args.input,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
