# -*- coding: utf-8 -*-
"""
benchmark/run_all_benchmarks.py

Run all benchmark models and collect their final metrics.

Training:
    python benchmark/run_all_benchmarks.py --models all --setting both
    python benchmark/run_all_benchmarks.py --models xgboost tabnet --setting fair --quick

Retrospective evaluation without retraining:
    python benchmark/run_all_benchmarks.py --models all --setting both --summary-only
    python benchmark/run_all_benchmarks.py --models node ft_transformer --setting fair --summary-only
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from benchmark.common import ensure_dir

from benchmark.xgboost_benchmark import (
    run_xgboost_fair,
    run_xgboost_enhanced,
    run_xgboost_fair_summary_only,
    run_xgboost_enhanced_summary_only,
)
from benchmark.tabnet_benchmark import (
    run_tabnet_fair,
    run_tabnet_enhanced,
    run_tabnet_fair_summary_only,
    run_tabnet_enhanced_summary_only,
)
from benchmark.ft_transformer_benchmark import (
    run_ft_transformer_fair,
    run_ft_transformer_enhanced,
    run_ft_transformer_fair_summary_only,
    run_ft_transformer_enhanced_summary_only,
)
from benchmark.node_benchmark import (
    run_node_fair,
    run_node_enhanced,
    run_node_fair_summary_only,
    run_node_enhanced_summary_only,
)


DEFAULT_DATA_ROOT = r"data"
BASE_DIR = os.path.join("results", "benchmark")

ALL_MODELS = ["xgboost", "tabnet", "ft_transformer", "node"]


RUNNERS = {
    "xgboost": {
        "fair": {
            "train": run_xgboost_fair,
            "summary_only": run_xgboost_fair_summary_only,
        },
        "enhanced": {
            "train": run_xgboost_enhanced,
            "summary_only": run_xgboost_enhanced_summary_only,
        },
    },
    "tabnet": {
        "fair": {
            "train": run_tabnet_fair,
            "summary_only": run_tabnet_fair_summary_only,
        },
        "enhanced": {
            "train": run_tabnet_enhanced,
            "summary_only": run_tabnet_enhanced_summary_only,
        },
    },
    "ft_transformer": {
        "fair": {
            "train": run_ft_transformer_fair,
            "summary_only": run_ft_transformer_fair_summary_only,
        },
        "enhanced": {
            "train": run_ft_transformer_enhanced,
            "summary_only": run_ft_transformer_enhanced_summary_only,
        },
    },
    "node": {
        "fair": {
            "train": run_node_fair,
            "summary_only": run_node_fair_summary_only,
        },
        "enhanced": {
            "train": run_node_enhanced,
            "summary_only": run_node_enhanced_summary_only,
        },
    },
}


def normalize_models(models: List[str]) -> List[str]:
    models = [model.lower() for model in models]

    if "all" in models:
        return ALL_MODELS.copy()

    unknown = [model for model in models if model not in ALL_MODELS]
    if unknown:
        raise ValueError(
            f"Unknown model(s): {unknown}. "
            f"Choose from: {', '.join(ALL_MODELS)} or all."
        )

    return list(dict.fromkeys(models))


def normalize_summary(summary: Dict) -> Dict:
    """
    Support both:
    1. the old nested summary format containing summary["metrics"];
    2. the new flat metric dictionary returned by the modified benchmarks.
    """
    if not isinstance(summary, dict):
        raise TypeError(
            f"Benchmark runner must return a dict, got {type(summary).__name__}."
        )

    row = {}

    old_metrics = summary.get("metrics")
    if isinstance(old_metrics, dict):
        row.update(old_metrics)

    for key, value in summary.items():
        if key == "metrics":
            continue

        if isinstance(value, (dict, list, tuple)):
            row[key] = json.dumps(value, ensure_ascii=False, default=str)
        else:
            row[key] = value

    return row


def _ordered_summary(df: pd.DataFrame) -> pd.DataFrame:
    preferred_cols = [
        "model",
        "setting",
        "test_cls_acc",
        "test_material_acc",
        "test_soc_rmse_raw",
        "test_soc_mae_raw",
        "test_soc_medae_raw",
        "test_soc_mape_raw",
        "test_soc_medape_raw",
        "test_soh_rmse_raw",
        "test_soh_mae_raw",
        "test_soh_medae_raw",
        "test_soh_mape_raw",
        "test_soh_medape_raw",
        "input_feature_dim",
        "n_train",
        "n_test",
        "device",
        "num_epochs",
        "summary_only",
    ]

    ordered_cols = [col for col in preferred_cols if col in df.columns]
    ordered_cols += [col for col in df.columns if col not in ordered_cols]
    return df[ordered_cols]


def _save_and_print_summary(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        print("[WARNING] No benchmark result was generated.")
        return pd.DataFrame()

    df = _ordered_summary(pd.DataFrame(rows))

    ensure_dir(BASE_DIR)
    out_csv = Path(BASE_DIR) / "benchmark_comparison_summary.csv"
    out_json = Path(BASE_DIR) / "benchmark_comparison_summary.json"

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            df.to_dict(orient="records"),
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print("\n[ALL BENCHMARKS SUMMARY]")

    show_cols = [
        "model",
        "setting",
        "test_cls_acc",
        "test_material_acc",
        "test_soc_medae_raw",
        "test_soh_medae_raw",
        "test_soc_medape_raw",
        "test_soh_medape_raw",
        "n_test",
    ]
    show_cols = [col for col in show_cols if col in df.columns]

    if show_cols:
        print(df[show_cols].to_string(index=False))
    else:
        print(df.to_string(index=False))

    print(f"\n[SAVED] {out_csv}")
    print(f"[SAVED] {out_json}")

    return df


def run_all(
    data_root: str,
    models: List[str],
    setting: str,
    quick: bool = False,
    use_cache: bool = True,
    summary_only: bool = False,
) -> pd.DataFrame:
    models = normalize_models(models)
    settings = ["fair", "enhanced"] if setting == "both" else [setting]

    rows = []

    for model_name in models:
        for current_setting in settings:
            runner_group = RUNNERS[model_name][current_setting]

            print("\n" + "=" * 80)
            print(
                f"[RUN] model={model_name}, "
                f"setting={current_setting}, "
                f"summary_only={summary_only}"
            )
            print("=" * 80)

            if summary_only:
                summary = runner_group["summary_only"](
                    data_root=data_root,
                    use_cache=use_cache,
                )
            else:
                summary = runner_group["train"](
                    data_root=data_root,
                    quick=quick,
                    use_cache=use_cache,
                )

            rows.append(normalize_summary(summary))

    return _save_and_print_summary(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Choose from: all, xgboost, tabnet, ft_transformer, node",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="both",
        choices=["fair", "enhanced", "both"],
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Load existing trained models and regenerate predictions and "
            "metrics without retraining."
        ),
    )
    args = parser.parse_args()

    run_all(
        data_root=args.data_root,
        models=args.models,
        setting=args.setting,
        quick=args.quick,
        use_cache=not args.no_cache,
        summary_only=args.summary_only,
    )


if __name__ == "__main__":
    main()
