# -*- coding: utf-8 -*-
"""
benchmark/xgboost_benchmark.py

XGBoost benchmark.

Settings:
    fair:
        material, SOC, SOH all use base features:
            41U + pulse_width

    enhanced:
        material classifier uses base features
        SOC regressor uses:
            base features + controlled material hint
        SOH regressor uses:
            base features + controlled material hint + pseudo SOC hint

Important:
    This version saves fitted XGBoost sklearn-wrapper models using pickle,
    instead of model.save_model(...), to avoid compatibility errors such as:

        TypeError: `_estimator_type` undefined.

Run:
    Directly click Run in PyCharm / VS Code, or:

        python benchmark/xgboost_benchmark.py

    For command-line quick test:

        python benchmark/xgboost_benchmark.py --setting both --quick
"""

from __future__ import annotations

import os
import sys
import argparse
import pickle
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from xgboost import XGBClassifier, XGBRegressor

from benchmark.common import (
    prepare_benchmark_data,
    save_predictions_and_summary,
    ensure_dir,
    save_json,
)
from benchmark.enhanced_inputs import build_enhanced_inputs


# =============================================================================
# Config
# =============================================================================
DEFAULT_DATA_ROOT = r"F:\OneDrive_Personal\OneDrive\battery\second life battery\data"

BASE_DIR = os.path.join("results", "benchmark")
MODEL_NAME = "xgboost"

PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

RANDOM_SEED = 42

# Enhanced / controlled-upstream benchmark level
# Material hint accuracy target: 92.3%
TARGET_MATERIAL_ACC = 0.923

# Pseudo SOC hint RMSE target, in SOC percentage points
TARGET_SOC_RMSE_RAW = 7.75


# =============================================================================
# XGBoost config / helpers
# =============================================================================
def build_xgb_params(quick: bool = False) -> Dict:
    """
    Build XGBoost parameters.

    quick=True:
        Use a small model for fast workflow testing.

    quick=False:
        Use the full benchmark configuration.
    """
    if quick:
        return {
            "n_estimators": 20,
            "max_depth": 4,
            "learning_rate": 0.1,
            "tree_method": "hist",
            "random_state": RANDOM_SEED,
        }

    return {
        "n_estimators": 500,
        "max_depth": 10,
        "learning_rate": 0.05,
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
    }


def save_xgb_sklearn_model(model, path: str) -> None:
    """
    Save XGBoost sklearn wrapper safely.

    Some xgboost / sklearn version combinations may fail when calling:

        model.save_model(...)

    because `_estimator_type` is undefined inside the sklearn wrapper metadata.
    Pickle is more robust for saving the fitted sklearn wrapper in this case.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[SAVED] {path}")


# =============================================================================
# Fair benchmark
# =============================================================================
def run_xgboost_fair(
    data_root: str = DEFAULT_DATA_ROOT,
    quick: bool = False,
    use_cache: bool = True,
) -> Dict:
    """
    Fair XGBoost benchmark.

    Material / SOC / SOH all use the same base tabular input:
        41U + pulse_width
    """
    out_dir = os.path.join(BASE_DIR, MODEL_NAME, "fair")
    ensure_dir(out_dir)

    data = prepare_benchmark_data(
        data_root=data_root,
        pulse_list=PULSE_LIST,
        base_dir=BASE_DIR,
        seed=RANDOM_SEED,
        use_cache=use_cache,
    )

    params = build_xgb_params(quick=quick)

    print("[XGBoost fair] Training material classifier.")
    model_clf = XGBClassifier(**params)
    model_clf.fit(data.Xtr, data.ytr_cls)

    print("[XGBoost fair] Training SOC regressor.")
    model_soc = XGBRegressor(**params)
    model_soc.fit(
        data.Xtr,
        data.mtr["SOC"].to_numpy(dtype=np.float32),
    )

    print("[XGBoost fair] Training SOH regressor.")
    model_soh = XGBRegressor(**params)
    model_soh.fit(
        data.Xtr,
        data.mtr["SOH"].to_numpy(dtype=np.float32),
    )

    print("[XGBoost fair] Predicting test set.")
    pred_cls_idx = model_clf.predict(data.Xte)
    pred_soc = model_soc.predict(data.Xte)
    pred_soh = model_soh.predict(data.Xte)

    # Save fitted sklearn-wrapper models with pickle.
    save_xgb_sklearn_model(
        model_clf,
        os.path.join(out_dir, "model_classifier.pkl"),
    )
    save_xgb_sklearn_model(
        model_soc,
        os.path.join(out_dir, "model_soc.pkl"),
    )
    save_xgb_sklearn_model(
        model_soh,
        os.path.join(out_dir, "model_soh.pkl"),
    )

    summary = save_predictions_and_summary(
        out_dir=out_dir,
        model_name=MODEL_NAME,
        setting="fair",
        data=data,
        pred_cls_idx=pred_cls_idx,
        pred_soc=pred_soc,
        pred_soh=pred_soh,
        extra_report={
            "xgb_params": params,
            "save_format": "pickle_sklearn_wrapper",
        },
    )

    return summary


# =============================================================================
# Enhanced benchmark
# =============================================================================
def run_xgboost_enhanced(
    data_root: str = DEFAULT_DATA_ROOT,
    quick: bool = False,
    use_cache: bool = True,
) -> Dict:
    """
    Enhanced / controlled-upstream XGBoost benchmark.

    Material classifier:
        base features only

    SOC regressor:
        base features + controlled material hint

    SOH regressor:
        base features + controlled material hint + pseudo SOC hint

    The controlled material hint and pseudo SOC hint are generated at error
    levels comparable to the proposed framework.
    """
    out_dir = os.path.join(BASE_DIR, MODEL_NAME, "enhanced")
    ensure_dir(out_dir)

    data = prepare_benchmark_data(
        data_root=data_root,
        pulse_list=PULSE_LIST,
        base_dir=BASE_DIR,
        seed=RANDOM_SEED,
        use_cache=use_cache,
    )

    params = build_xgb_params(quick=quick)

    print("[XGBoost enhanced] Training baseline material classifier.")
    model_clf = XGBClassifier(**params)
    model_clf.fit(data.Xtr, data.ytr_cls)

    print("[XGBoost enhanced] Predicting material on test set.")
    pred_cls_idx = model_clf.predict(data.Xte)

    soc_tr_true = data.mtr["SOC"].to_numpy(dtype=np.float32)
    soc_te_true = data.mte["SOC"].to_numpy(dtype=np.float32)

    print("[XGBoost enhanced] Building controlled upstream inputs.")
    Xtr_soc, Xte_soc, Xtr_soh, Xte_soh, hint_report = build_enhanced_inputs(
        Xtr=data.Xtr,
        Xte=data.Xte,
        ytr_cls=data.ytr_cls,
        yte_cls=data.yte_cls,
        soc_tr_true=soc_tr_true,
        soc_te_true=soc_te_true,
        num_classes=data.num_classes,
        target_material_acc=TARGET_MATERIAL_ACC,
        target_soc_rmse=TARGET_SOC_RMSE_RAW,
        seed=RANDOM_SEED,
    )

    print("[XGBoost enhanced] Training SOC regressor with material hint.")
    model_soc = XGBRegressor(**params)
    model_soc.fit(Xtr_soc, soc_tr_true)

    print("[XGBoost enhanced] Training SOH regressor with material hint + pseudo SOC hint.")
    model_soh = XGBRegressor(**params)
    model_soh.fit(
        Xtr_soh,
        data.mtr["SOH"].to_numpy(dtype=np.float32),
    )

    print("[XGBoost enhanced] Predicting test set.")
    pred_soc = model_soc.predict(Xte_soc)
    pred_soh = model_soh.predict(Xte_soh)

    # Save fitted sklearn-wrapper models with pickle.
    save_xgb_sklearn_model(
        model_clf,
        os.path.join(out_dir, "model_classifier_baseline.pkl"),
    )
    save_xgb_sklearn_model(
        model_soc,
        os.path.join(out_dir, "model_soc_enhanced.pkl"),
    )
    save_xgb_sklearn_model(
        model_soh,
        os.path.join(out_dir, "model_soh_enhanced.pkl"),
    )

    save_json(
        hint_report,
        os.path.join(out_dir, "enhanced_hint_report.json"),
    )

    summary = save_predictions_and_summary(
        out_dir=out_dir,
        model_name=MODEL_NAME,
        setting="enhanced",
        data=data,
        pred_cls_idx=pred_cls_idx,
        pred_soc=pred_soc,
        pred_soh=pred_soh,
        extra_report={
            "xgb_params": params,
            "save_format": "pickle_sklearn_wrapper",
            "target_material_acc": TARGET_MATERIAL_ACC,
            "target_soc_rmse_raw": TARGET_SOC_RMSE_RAW,
            "enhanced_hint_report": hint_report,
        },
    )

    return summary


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="both",
        choices=["fair", "enhanced", "both"],
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use small XGBoost models for fast workflow testing.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild benchmark data cache.",
    )

    args = parser.parse_args()

    use_cache = not args.no_cache

    if args.setting in ["fair", "both"]:
        run_xgboost_fair(
            data_root=args.data_root,
            quick=args.quick,
            use_cache=use_cache,
        )

    if args.setting in ["enhanced", "both"]:
        run_xgboost_enhanced(
            data_root=args.data_root,
            quick=args.quick,
            use_cache=use_cache,
        )


if __name__ == "__main__":
    main()