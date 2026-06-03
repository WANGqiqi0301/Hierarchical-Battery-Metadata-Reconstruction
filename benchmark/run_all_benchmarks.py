# -*- coding: utf-8 -*-
"""
benchmark/run_all_benchmarks.py

Run all benchmark models and collect summaries.

Run:
    python benchmark/run_all_benchmarks.py --models xgboost --setting both --quick
    python benchmark/run_all_benchmarks.py --models xgboost tabnet --setting fair
    python benchmark/run_all_benchmarks.py --models all --setting both
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from benchmark.common import ensure_dir

from benchmark.xgboost_benchmark import run_xgboost_fair, run_xgboost_enhanced
from benchmark.tabnet_benchmark import run_tabnet_fair, run_tabnet_enhanced
from benchmark.ft_transformer_benchmark import run_ft_transformer_fair, run_ft_transformer_enhanced
from benchmark.node_benchmark import run_node_fair, run_node_enhanced


DEFAULT_DATA_ROOT = r"F:\OneDrive_Personal\OneDrive\battery\second life battery\data"

BASE_DIR = os.path.join("results", "benchmark")


def flatten_summary(summary: Dict) -> Dict:
    row = {
        "model": summary.get("model"),
        "setting": summary.get("setting"),
        "input_feature_dim": summary.get("input_feature_dim"),
        "n_train": summary.get("n_train"),
        "n_test": summary.get("n_test"),
    }

    metrics = summary.get("metrics", {})
    for k, v in metrics.items():
        row[k] = v

    return row


def normalize_models(models: List[str]) -> List[str]:
    if len(models) == 1 and models[0].lower() == "all":
        return ["xgboost", "tabnet", "ft_transformer", "node"]
    return [m.lower() for m in models]


def run_all(
    data_root: str,
    models: List[str],
    setting: str,
    quick: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    models = normalize_models(models)
    rows = []

    for model_name in models:
        if model_name == "xgboost":
            if setting in ["fair", "both"]:
                rows.append(flatten_summary(run_xgboost_fair(data_root=data_root, quick=quick, use_cache=use_cache)))
            if setting in ["enhanced", "both"]:
                rows.append(flatten_summary(run_xgboost_enhanced(data_root=data_root, quick=quick, use_cache=use_cache)))

        elif model_name == "tabnet":
            if setting in ["fair", "both"]:
                rows.append(flatten_summary(run_tabnet_fair(data_root=data_root, quick=quick, use_cache=use_cache)))
            if setting in ["enhanced", "both"]:
                rows.append(flatten_summary(run_tabnet_enhanced(data_root=data_root, quick=quick, use_cache=use_cache)))

        elif model_name == "ft_transformer":
            if setting in ["fair", "both"]:
                rows.append(flatten_summary(run_ft_transformer_fair(data_root=data_root, quick=quick, use_cache=use_cache)))
            if setting in ["enhanced", "both"]:
                rows.append(flatten_summary(run_ft_transformer_enhanced(data_root=data_root, quick=quick, use_cache=use_cache)))

        elif model_name == "node":
            if setting in ["fair", "both"]:
                rows.append(flatten_summary(run_node_fair(data_root=data_root, quick=quick, use_cache=use_cache)))
            if setting in ["enhanced", "both"]:
                rows.append(flatten_summary(run_node_enhanced(data_root=data_root, quick=quick, use_cache=use_cache)))

        else:
            raise ValueError(f"Unknown model: {model_name}")

    df = pd.DataFrame(rows)

    ensure_dir(BASE_DIR)
    out_csv = os.path.join(BASE_DIR, "benchmark_comparison_summary.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\n[SAVED] {out_csv}")
    print(df.to_string(index=False))

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Choose from: all, xgboost, tabnet, ft_transformer, node",
    )
    parser.add_argument("--setting", type=str, default="both", choices=["fair", "enhanced", "both"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    run_all(
        data_root=args.data_root,
        models=args.models,
        setting=args.setting,
        quick=args.quick,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()