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
import json
import argparse
import pickle
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
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
DEFAULT_DATA_ROOT = r"data"

BASE_DIR = os.path.join("results", "benchmark")
MODEL_NAME = "xgboost"

PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

RANDOM_SEED = 42

# Enhanced / controlled-upstream benchmark level.
# Use the proposed-method TEST-level error from further analysis.

TARGET_MATERIAL_ACC = 0.91875
TARGET_SOC_RMSE_RAW = 7.864401414996508


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


def load_xgb_sklearn_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"XGBoost pickle checkpoint not found: {path}")

    with open(path, "rb") as f:
        model = pickle.load(f)

    print(f"[LOADED] {path}")
    return model


def _medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.median(np.abs(y_pred - y_true)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_pred - y_true)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs((y_pred - y_true) / np.maximum(np.abs(y_true), 1e-8))) * 100.0)


def _medape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.median(np.abs((y_pred - y_true) / np.maximum(np.abs(y_true), 1e-8))) * 100.0)


def _class_names(data) -> np.ndarray:
    for attr in ("label_encoder", "le"):
        encoder = getattr(data, attr, None)
        if encoder is not None and hasattr(encoder, "inverse_transform"):
            return np.asarray(
                encoder.inverse_transform(np.arange(int(data.num_classes))),
                dtype=object,
            )

    for attr in ("class_names", "classes"):
        values = getattr(data, attr, None)
        if values is not None and len(values) == int(data.num_classes):
            return np.asarray(values, dtype=object)

    label_cols = (
        "class_label",
        "label",
        "target",
        "material_capacity",
        "material_capacity_label",
        "battery_type",
        "class_name",
    )
    ytr = np.asarray(data.ytr_cls, dtype=int)

    for col in label_cols:
        if col not in data.mtr.columns:
            continue

        names = []
        valid = True

        for cls_idx in range(int(data.num_classes)):
            values = data.mtr.loc[ytr == cls_idx, col].astype(str)

            if values.empty:
                valid = False
                break

            names.append(values.mode().iloc[0])

        if valid:
            return np.asarray(names, dtype=object)

    return np.asarray([str(i) for i in range(int(data.num_classes))], dtype=object)


def _material_names(data, class_names: np.ndarray) -> np.ndarray:
    material_cols = (
        "material",
        "Material",
        "cathode_material",
        "chemistry",
        "Chemistry",
    )
    ytr = np.asarray(data.ytr_cls, dtype=int)

    for col in material_cols:
        if col not in data.mtr.columns:
            continue

        materials = []
        valid = True

        for cls_idx in range(int(data.num_classes)):
            values = data.mtr.loc[ytr == cls_idx, col].astype(str)

            if values.empty:
                valid = False
                break

            materials.append(values.mode().iloc[0])

        if valid:
            return np.asarray(materials, dtype=object)

    return np.asarray([str(name).split("_")[0] for name in class_names], dtype=object)


def _build_extended_outputs(
    out_dir: str | Path,
    setting: str,
    data,
    pred_cls_idx: np.ndarray,
    pred_soc: np.ndarray,
    pred_soh: np.ndarray,
    extra_report: Dict | None = None,
) -> Dict:
    out_dir = Path(out_dir)
    ensure_dir(str(out_dir))

    true_cls = np.asarray(data.yte_cls, dtype=int).reshape(-1)
    pred_cls = np.asarray(pred_cls_idx, dtype=int).reshape(-1)

    soc_true = data.mte["SOC"].to_numpy(dtype=np.float64)
    soc_pred = np.asarray(pred_soc, dtype=np.float64).reshape(-1)

    # Training target is fractional SOH. Reporting uses percentage points.
    soh_true = data.mte["SOH"].to_numpy(dtype=np.float64) * 100.0
    soh_pred = np.asarray(pred_soh, dtype=np.float64).reshape(-1) * 100.0

    n = len(true_cls)

    if not all(len(x) == n for x in (pred_cls, soc_true, soc_pred, soh_true, soh_pred)):
        raise RuntimeError(
            "Prediction length mismatch: "
            f"true_cls={n}, pred_cls={len(pred_cls)}, "
            f"soc_true={len(soc_true)}, soc_pred={len(soc_pred)}, "
            f"soh_true={len(soh_true)}, soh_pred={len(soh_pred)}"
        )

    class_names = _class_names(data)
    material_names = _material_names(data, class_names)

    true_labels = class_names[true_cls]
    pred_labels = class_names[pred_cls]
    true_material = material_names[true_cls]
    pred_material = material_names[pred_cls]

    soc_ae = np.abs(soc_pred - soc_true)
    soh_ae = np.abs(soh_pred - soh_true)
    soc_ape = soc_ae / np.maximum(np.abs(soc_true), 1e-8) * 100.0
    soh_ape = soh_ae / np.maximum(np.abs(soh_true), 1e-8) * 100.0

    pred_df = pd.DataFrame(
        {
            "true_cls_idx": true_cls,
            "pred_cls_idx": pred_cls,
            "true_label": true_labels.astype(str),
            "pred_label": pred_labels.astype(str),
            "true_material": true_material.astype(str),
            "pred_material": pred_material.astype(str),
            "class_correct": true_cls == pred_cls,
            "material_correct": true_material.astype(str) == pred_material.astype(str),
            "soc_true": soc_true,
            "soc_pred": soc_pred,
            "soc_ae": soc_ae,
            "soc_ape": soc_ape,
            "soh_true": soh_true,
            "soh_pred": soh_pred,
            "soh_ae": soh_ae,
            "soh_ape": soh_ape,
        }
    )

    for col in ("ID", "pulse_ms"):
        if col in data.mte.columns:
            pred_df[col] = data.mte[col].reset_index(drop=True).to_numpy()

    pred_path = out_dir / "test_predictions_per_sample.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    metrics = {
        "model": MODEL_NAME,
        "setting": setting,
        "test_cls_acc": float(accuracy_score(true_cls, pred_cls)),
        "test_material_acc": float(
            accuracy_score(true_material.astype(str), pred_material.astype(str))
        ),
        "test_soc_rmse_raw": _rmse(soc_true, soc_pred),
        "test_soc_mae_raw": _mae(soc_true, soc_pred),
        "test_soc_medae_raw": _medae(soc_true, soc_pred),
        "test_soc_mape_raw": _mape(soc_true, soc_pred),
        "test_soc_medape_raw": _medape(soc_true, soc_pred),
        "test_soh_rmse_raw": _rmse(soh_true, soh_pred),
        "test_soh_mae_raw": _mae(soh_true, soh_pred),
        "test_soh_medae_raw": _medae(soh_true, soh_pred),
        "test_soh_mape_raw": _mape(soh_true, soh_pred),
        "test_soh_medape_raw": _medape(soh_true, soh_pred),
        "n_test": int(n),
    }

    if extra_report:
        for key, value in extra_report.items():
            if isinstance(value, (dict, list, tuple)):
                metrics[key] = json.dumps(value, ensure_ascii=False)
            else:
                metrics[key] = value

    metrics_path = out_dir / "final_metrics.csv"
    metrics_json = out_dir / "final_metrics.json"

    pd.DataFrame([metrics]).to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )
    save_json(metrics, str(metrics_json))

    print("\n[EXTENDED TEST METRICS]")
    print(f"setting: {setting}")
    print(f"test_cls_acc: {metrics['test_cls_acc']:.6f}")
    print(f"test_material_acc: {metrics['test_material_acc']:.6f}")
    print(f"test_soc_mae_raw: {metrics['test_soc_mae_raw']:.6f}")
    print(f"test_soc_medae_raw: {metrics['test_soc_medae_raw']:.6f}")
    print(f"test_soc_medape_raw: {metrics['test_soc_medape_raw']:.6f}")
    print(f"test_soh_mae_raw: {metrics['test_soh_mae_raw']:.6f}")
    print(f"test_soh_medae_raw: {metrics['test_soh_medae_raw']:.6f}")
    print(f"test_soh_medape_raw: {metrics['test_soh_medape_raw']:.6f}")
    print(f"[SAVED] {pred_path}")
    print(f"[SAVED] {metrics_path}")
    print(f"[SAVED] {metrics_json}")

    return metrics


def _merge_summary(summary: Dict, extended: Dict) -> Dict:
    merged = dict(summary) if isinstance(summary, dict) else {}
    merged.update(extended)
    return merged


def _save_and_print_master_summary(rows) -> pd.DataFrame:
    rows = list(rows)

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)

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
        "n_test",
        "summary_only",
    ]
    cols = [col for col in preferred_cols if col in summary.columns]
    cols += [col for col in summary.columns if col not in cols]
    summary = summary[cols]

    summary_dir = Path(BASE_DIR) / MODEL_NAME
    ensure_dir(str(summary_dir))

    summary_csv = summary_dir / "xgboost_benchmark_summary.csv"
    summary_json = summary_dir / "xgboost_benchmark_summary.json"

    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    save_json(summary.to_dict(orient="records"), str(summary_json))

    print("\n[XGBOOST BENCHMARK SUMMARY]")
    show_cols = [
        "setting",
        "test_cls_acc",
        "test_material_acc",
        "test_soc_medae_raw",
        "test_soh_medae_raw",
        "test_soc_medape_raw",
        "test_soh_medape_raw",
        "n_test",
    ]
    show_cols = [col for col in show_cols if col in summary.columns]
    print(summary[show_cols].to_string(index=False))
    print(f"\n[SAVED] {summary_csv}")
    print(f"[SAVED] {summary_json}")

    return summary


def run_xgboost_fair_summary_only(
    data_root: str = DEFAULT_DATA_ROOT,
    use_cache: bool = True,
) -> Dict:
    out_dir = os.path.join(BASE_DIR, MODEL_NAME, "fair")

    data = prepare_benchmark_data(
        data_root=data_root,
        pulse_list=PULSE_LIST,
        base_dir=BASE_DIR,
        seed=RANDOM_SEED,
        use_cache=use_cache,
    )

    model_clf = load_xgb_sklearn_model(
        os.path.join(out_dir, "model_classifier.pkl")
    )
    model_soc = load_xgb_sklearn_model(
        os.path.join(out_dir, "model_soc.pkl")
    )
    model_soh = load_xgb_sklearn_model(
        os.path.join(out_dir, "model_soh.pkl")
    )

    print("[XGBoost fair summary-only] Predicting test set.")
    pred_cls_idx = model_clf.predict(data.Xte)
    pred_soc = model_soc.predict(data.Xte)
    pred_soh = model_soh.predict(data.Xte)

    extra = {
        "summary_only": True,
        "classifier_checkpoint": os.path.join(out_dir, "model_classifier.pkl"),
        "soc_checkpoint": os.path.join(out_dir, "model_soc.pkl"),
        "soh_checkpoint": os.path.join(out_dir, "model_soh.pkl"),
    }

    summary = save_predictions_and_summary(
        out_dir=out_dir,
        model_name=MODEL_NAME,
        setting="fair",
        data=data,
        pred_cls_idx=pred_cls_idx,
        pred_soc=pred_soc,
        pred_soh=pred_soh,
        extra_report=extra,
    )
    extended = _build_extended_outputs(
        out_dir=out_dir,
        setting="fair",
        data=data,
        pred_cls_idx=pred_cls_idx,
        pred_soc=pred_soc,
        pred_soh=pred_soh,
        extra_report=extra,
    )

    return _merge_summary(summary, extended)


def run_xgboost_enhanced_summary_only(
    data_root: str = DEFAULT_DATA_ROOT,
    use_cache: bool = True,
) -> Dict:
    out_dir = os.path.join(BASE_DIR, MODEL_NAME, "enhanced")

    data = prepare_benchmark_data(
        data_root=data_root,
        pulse_list=PULSE_LIST,
        base_dir=BASE_DIR,
        seed=RANDOM_SEED,
        use_cache=use_cache,
    )

    soc_tr_true = data.mtr["SOC"].to_numpy(dtype=np.float32)
    soc_te_true = data.mte["SOC"].to_numpy(dtype=np.float32)

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

    model_clf = load_xgb_sklearn_model(
        os.path.join(out_dir, "model_classifier_baseline.pkl")
    )
    model_soc = load_xgb_sklearn_model(
        os.path.join(out_dir, "model_soc_enhanced.pkl")
    )
    model_soh = load_xgb_sklearn_model(
        os.path.join(out_dir, "model_soh_enhanced.pkl")
    )

    print("[XGBoost enhanced summary-only] Predicting test set.")
    pred_cls_idx = model_clf.predict(data.Xte)
    pred_soc = model_soc.predict(Xte_soc)
    pred_soh = model_soh.predict(Xte_soh)

    extra = {
        "summary_only": True,
        "classifier_checkpoint": os.path.join(out_dir, "model_classifier_baseline.pkl"),
        "soc_checkpoint": os.path.join(out_dir, "model_soc_enhanced.pkl"),
        "soh_checkpoint": os.path.join(out_dir, "model_soh_enhanced.pkl"),
        "enhanced_hint_report": hint_report,
    }

    summary = save_predictions_and_summary(
        out_dir=out_dir,
        model_name=MODEL_NAME,
        setting="enhanced",
        data=data,
        pred_cls_idx=pred_cls_idx,
        pred_soc=pred_soc,
        pred_soh=pred_soh,
        extra_report=extra,
    )
    extended = _build_extended_outputs(
        out_dir=out_dir,
        setting="enhanced",
        data=data,
        pred_cls_idx=pred_cls_idx,
        pred_soc=pred_soc,
        pred_soh=pred_soh,
        extra_report=extra,
    )

    return _merge_summary(summary, extended)


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

    extra = {
        "xgb_params": params,
        "save_format": "pickle_sklearn_wrapper",
        "summary_only": False,
    }

    summary = save_predictions_and_summary(
        out_dir=out_dir,
        model_name=MODEL_NAME,
        setting="fair",
        data=data,
        pred_cls_idx=pred_cls_idx,
        pred_soc=pred_soc,
        pred_soh=pred_soh,
        extra_report=extra,
    )
    extended = _build_extended_outputs(
        out_dir=out_dir,
        setting="fair",
        data=data,
        pred_cls_idx=pred_cls_idx,
        pred_soc=pred_soc,
        pred_soh=pred_soh,
        extra_report=extra,
    )

    return _merge_summary(summary, extended)


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

    extra = {
        "xgb_params": params,
        "save_format": "pickle_sklearn_wrapper",
        "target_material_acc": TARGET_MATERIAL_ACC,
        "target_soc_rmse_raw": TARGET_SOC_RMSE_RAW,
        "enhanced_hint_report": hint_report,
        "summary_only": False,
    }

    summary = save_predictions_and_summary(
        out_dir=out_dir,
        model_name=MODEL_NAME,
        setting="enhanced",
        data=data,
        pred_cls_idx=pred_cls_idx,
        pred_soc=pred_soc,
        pred_soh=pred_soh,
        extra_report=extra,
    )
    extended = _build_extended_outputs(
        out_dir=out_dir,
        setting="enhanced",
        data=data,
        pred_cls_idx=pred_cls_idx,
        pred_soc=pred_soc,
        pred_soh=pred_soh,
        extra_report=extra,
    )

    return _merge_summary(summary, extended)


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
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Load existing XGBoost pickle models and regenerate predictions/metrics without training.",
    )

    args = parser.parse_args()

    use_cache = not args.no_cache
    rows = []

    if args.setting in ["fair", "both"]:
        if args.summary_only:
            rows.append(
                run_xgboost_fair_summary_only(
                    data_root=args.data_root,
                    use_cache=use_cache,
                )
            )
        else:
            rows.append(
                run_xgboost_fair(
                    data_root=args.data_root,
                    quick=args.quick,
                    use_cache=use_cache,
                )
            )

    if args.setting in ["enhanced", "both"]:
        if args.summary_only:
            rows.append(
                run_xgboost_enhanced_summary_only(
                    data_root=args.data_root,
                    use_cache=use_cache,
                )
            )
        else:
            rows.append(
                run_xgboost_enhanced(
                    data_root=args.data_root,
                    quick=args.quick,
                    use_cache=use_cache,
                )
            )

    _save_and_print_master_summary(rows)


if __name__ == "__main__":
    main()