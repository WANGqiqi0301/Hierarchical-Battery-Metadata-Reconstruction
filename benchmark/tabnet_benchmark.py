# -*- coding: utf-8 -*-
"""
benchmark/tabnet_benchmark.py

TabNet benchmark for battery passport reconstruction.

This script evaluates TabNet under two settings:

1. Fair benchmark
   - Material classification uses base tabular features.
   - SOC regression uses base tabular features.
   - SOH regression uses base tabular features.

2. Enhanced / controlled-upstream benchmark
   - Material classification uses base tabular features.
   - SOC regression uses base tabular features + controlled material hint.
   - SOH regression uses base tabular features + controlled material hint + pseudo SOC hint.

The enhanced setting corresponds to the previous "unfair" scripts, but is renamed
to "enhanced" or "controlled-upstream" because it provides benchmark models with
upstream information at error levels comparable to the proposed framework.

Input features:
    U1-U41 + pulse_width = 42 tabular features

Dependencies:
    pip install pytorch-tabnet

Project dependencies:
    benchmark/common.py
    benchmark/enhanced_inputs.py
    utils/data_loader.py

Run examples:
    python benchmark/tabnet_benchmark.py
    python benchmark/tabnet_benchmark.py --setting fair
    python benchmark/tabnet_benchmark.py --setting enhanced
    python benchmark/tabnet_benchmark.py --setting both --quick
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# =============================================================================
# Project root
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from benchmark.common import (
    ensure_dir,
    mape,
    median_ape,
    prepare_benchmark_data,
    save_json,
    save_predictions_and_summary,
)
from benchmark.enhanced_inputs import build_enhanced_inputs


# =============================================================================
# Configuration
# =============================================================================
DEFAULT_DATA_ROOT = r"F:\OneDrive_Personal\OneDrive\battery\second life battery\data"

BASE_DIR = os.path.join("results", "benchmark")
MODEL_NAME = "tabnet"

PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

RANDOM_SEED = 42

# Enhanced / controlled-upstream benchmark level
# Material hint accuracy target: 92.3%
TARGET_MATERIAL_ACC = 0.923

# Pseudo SOC hint RMSE target, in SOC percentage points
TARGET_SOC_RMSE_RAW = 7.75


# =============================================================================
# Custom TabNet metrics
# =============================================================================
class MAPEMetric(Metric):
    """Mean absolute percentage error for TabNet training logs."""

    def __init__(self) -> None:
        self._name = "mape"
        self._maximize = False

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return mape(y_true, y_score)


class MedAPEMetric(Metric):
    """Median absolute percentage error for TabNet training logs."""

    def __init__(self) -> None:
        self._name = "med_ape"
        self._maximize = False

    def __call__(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return median_ape(y_true, y_score)


# =============================================================================
# TabNet helpers
# =============================================================================
def build_tabnet_params() -> Dict[str, Any]:
    """
    Return TabNet hyperparameters.

    These values are intentionally kept close to the original comparison scripts.
    """
    return {
        "n_d": 16,
        "n_a": 16,
        "n_steps": 4,
        "gamma": 1.3,
        "seed": RANDOM_SEED,
        "verbose": 1,
    }


def save_tabnet_model(model: Any, path: str) -> None:
    """
    Save a fitted TabNet model using pickle.

    Rationale:
        Some model-specific save methods may create version-dependent files.
        Pickle stores the fitted Python object directly and keeps behavior
        consistent with the XGBoost benchmark in this project.

    Note:
        For long-term archival across Python/library versions, also preserve
        the environment file or package versions.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[SAVED] {path}")


def fit_tabnet_classifier(
    Xtr: np.ndarray,
    ytr_cls: np.ndarray,
    Xte: np.ndarray,
    yte_cls: np.ndarray,
    quick: bool = False,
) -> TabNetClassifier:
    """
    Fit a TabNet classifier for material-capacity classification.
    """
    model = TabNetClassifier(**build_tabnet_params())

    model.fit(
        X_train=Xtr,
        y_train=ytr_cls,
        eval_set=[(Xte, yte_cls)],
        eval_name=["test"],
        eval_metric=["accuracy"],
        max_epochs=5 if quick else 150,
        patience=3 if quick else 20,
        batch_size=256,
        virtual_batch_size=128,
    )

    return model


def fit_tabnet_regressor(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    quick: bool = False,
) -> TabNetRegressor:
    """
    Fit a TabNet regressor for SOC or SOH estimation.
    """
    model = TabNetRegressor(**build_tabnet_params())

    model.fit(
        X_train=Xtr,
        y_train=ytr.reshape(-1, 1).astype(np.float32),
        eval_set=[(Xte, yte.reshape(-1, 1).astype(np.float32))],
        eval_name=["test"],
        eval_metric=["rmse", MAPEMetric, MedAPEMetric],
        max_epochs=5 if quick else 150,
        patience=3 if quick else 20,
        batch_size=256,
        virtual_batch_size=128,
    )

    return model


# =============================================================================
# Fair benchmark
# =============================================================================
def run_tabnet_fair(
    data_root: str = DEFAULT_DATA_ROOT,
    quick: bool = False,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Run the fair TabNet benchmark.

    Fair setting:
        material, SOC and SOH models all use only:
            U1-U41 + pulse_width
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

    print("[TabNet fair] Training material classifier.")
    model_clf = fit_tabnet_classifier(
        Xtr=data.Xtr,
        ytr_cls=data.ytr_cls,
        Xte=data.Xte,
        yte_cls=data.yte_cls,
        quick=quick,
    )

    print("[TabNet fair] Training SOC regressor.")
    model_soc = fit_tabnet_regressor(
        Xtr=data.Xtr,
        ytr=data.mtr["SOC"].to_numpy(dtype=np.float32),
        Xte=data.Xte,
        yte=data.mte["SOC"].to_numpy(dtype=np.float32),
        quick=quick,
    )

    print("[TabNet fair] Training SOH regressor.")
    model_soh = fit_tabnet_regressor(
        Xtr=data.Xtr,
        ytr=data.mtr["SOH"].to_numpy(dtype=np.float32),
        Xte=data.Xte,
        yte=data.mte["SOH"].to_numpy(dtype=np.float32),
        quick=quick,
    )

    print("[TabNet fair] Predicting test set.")
    pred_cls_idx = model_clf.predict(data.Xte)
    pred_soc = model_soc.predict(data.Xte).reshape(-1)
    pred_soh = model_soh.predict(data.Xte).reshape(-1)

    save_tabnet_model(
        model_clf,
        os.path.join(out_dir, "model_classifier.pkl"),
    )
    save_tabnet_model(
        model_soc,
        os.path.join(out_dir, "model_soc.pkl"),
    )
    save_tabnet_model(
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
            "tabnet_params": build_tabnet_params(),
            "save_format": "pickle",
        },
    )

    return summary


# =============================================================================
# Enhanced benchmark
# =============================================================================
def run_tabnet_enhanced(
    data_root: str = DEFAULT_DATA_ROOT,
    quick: bool = False,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Run the enhanced / controlled-upstream TabNet benchmark.

    Enhanced setting:
        material classifier:
            base features

        SOC regressor:
            base features + controlled material hint

        SOH regressor:
            base features + controlled material hint + pseudo SOC hint
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

    print("[TabNet enhanced] Training baseline material classifier.")
    model_clf = fit_tabnet_classifier(
        Xtr=data.Xtr,
        ytr_cls=data.ytr_cls,
        Xte=data.Xte,
        yte_cls=data.yte_cls,
        quick=quick,
    )

    print("[TabNet enhanced] Predicting material on test set.")
    pred_cls_idx = model_clf.predict(data.Xte)

    soc_tr_true = data.mtr["SOC"].to_numpy(dtype=np.float32)
    soc_te_true = data.mte["SOC"].to_numpy(dtype=np.float32)

    print("[TabNet enhanced] Building controlled upstream inputs.")
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

    print("[TabNet enhanced] Training SOC regressor with material hint.")
    model_soc = fit_tabnet_regressor(
        Xtr=Xtr_soc,
        ytr=soc_tr_true,
        Xte=Xte_soc,
        yte=soc_te_true,
        quick=quick,
    )

    print("[TabNet enhanced] Training SOH regressor with material hint + pseudo SOC hint.")
    model_soh = fit_tabnet_regressor(
        Xtr=Xtr_soh,
        ytr=data.mtr["SOH"].to_numpy(dtype=np.float32),
        Xte=Xte_soh,
        yte=data.mte["SOH"].to_numpy(dtype=np.float32),
        quick=quick,
    )

    print("[TabNet enhanced] Predicting test set.")
    pred_soc = model_soc.predict(Xte_soc).reshape(-1)
    pred_soh = model_soh.predict(Xte_soh).reshape(-1)

    save_tabnet_model(
        model_clf,
        os.path.join(out_dir, "model_classifier_baseline.pkl"),
    )
    save_tabnet_model(
        model_soc,
        os.path.join(out_dir, "model_soc_enhanced.pkl"),
    )
    save_tabnet_model(
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
            "tabnet_params": build_tabnet_params(),
            "save_format": "pickle",
            "target_material_acc": TARGET_MATERIAL_ACC,
            "target_soc_rmse_raw": TARGET_SOC_RMSE_RAW,
            "enhanced_hint_report": hint_report,
        },
    )

    return summary


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TabNet fair/enhanced benchmark comparison."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Root directory of the battery dataset.",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="both",
        choices=["fair", "enhanced", "both"],
        help="Benchmark setting to run.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer epochs for fast workflow testing.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Rebuild benchmark dataset cache instead of loading existing cache.",
    )

    args = parser.parse_args()
    use_cache = not args.no_cache

    if args.setting in ["fair", "both"]:
        run_tabnet_fair(
            data_root=args.data_root,
            quick=args.quick,
            use_cache=use_cache,
        )

    if args.setting in ["enhanced", "both"]:
        run_tabnet_enhanced(
            data_root=args.data_root,
            quick=args.quick,
            use_cache=use_cache,
        )


if __name__ == "__main__":
    main()