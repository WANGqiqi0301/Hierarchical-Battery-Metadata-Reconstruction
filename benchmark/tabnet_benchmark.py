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
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

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
DEFAULT_DATA_ROOT = r"data"

BASE_DIR = os.path.join("results", "benchmark")
MODEL_NAME = "tabnet"

PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

RANDOM_SEED = 42

# Enhanced / controlled-upstream benchmark level.
# Use the proposed-method TEST-level error from further analysis.

TARGET_MATERIAL_ACC = 0.91875
TARGET_SOC_RMSE_RAW = 7.864401414996508

# Internal validation split for TabNet early stopping.
# Important: this validation split is drawn from the training IDs only.
VAL_ID_FRAC = 0.2
VAL_SEED_OFFSET = 1000


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


def _strip_tabnet_zip_suffix(path: str) -> str:
    """
    Return the path expected by pytorch-tabnet save_model().

    pytorch-tabnet's save_model(path) automatically writes path + ".zip".
    Therefore, if the caller passes "xxx.zip", we strip the suffix before saving
    or loading to avoid creating "xxx.zip.zip".
    """
    return path[:-4] if path.endswith(".zip") else path


def save_tabnet_model(model: Any, path: str) -> str:
    """
    Save a fitted TabNet model in pytorch-tabnet native .zip format.

    This matches the original TabNet scripts, where model.save_model(path)
    writes a zip file at path + ".zip".

    Parameters
    ----------
    model:
        Fitted TabNetClassifier or TabNetRegressor.

    path:
        Save path with or without the .zip suffix. The actual saved file is .zip.

    Returns
    -------
    str
        Actual zip file path.
    """
    path_no_ext = _strip_tabnet_zip_suffix(path)
    ensure_dir(os.path.dirname(path_no_ext))

    actual_path = model.save_model(path_no_ext)

    # pytorch-tabnet usually returns the actual saved path. Keep a fallback for
    # versions that return None.
    if actual_path is None:
        actual_path = path_no_ext + ".zip"

    print(f"[SAVED] {actual_path}")
    return actual_path


def load_tabnet_model(model_cls: Any, path: str, tabnet_params: Dict[str, Any] | None = None) -> Any:
    """
    Load a TabNet model from pytorch-tabnet native .zip format.

    This helper keeps the read format aligned with save_tabnet_model(). It does
    not change the training or CLI workflow; it is provided for scripts that need
    to reload the saved benchmark models later.

    Parameters
    ----------
    model_cls:
        TabNetClassifier or TabNetRegressor.

    path:
        Zip checkpoint path, or the same path without .zip.

    tabnet_params:
        Optional parameters used to instantiate the model before loading.

    Returns
    -------
    Any
        Loaded TabNet model.
    """
    path_no_ext = _strip_tabnet_zip_suffix(path)
    zip_path = path_no_ext + ".zip"

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"TabNet checkpoint not found: {zip_path}")

    params = build_tabnet_params() if tabnet_params is None else dict(tabnet_params)
    model = model_cls(**params)
    model.load_model(zip_path)
    print(f"[LOADED] {zip_path}")
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
    extra_report: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
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


def _merge_summary(summary: Dict[str, Any], extended: Dict[str, Any]) -> Dict[str, Any]:
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

    summary_csv = summary_dir / "tabnet_benchmark_summary.csv"
    summary_json = summary_dir / "tabnet_benchmark_summary.json"

    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    save_json(summary.to_dict(orient="records"), str(summary_json))

    print("\n[TABNET BENCHMARK SUMMARY]")
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


def run_tabnet_fair_summary_only(
    data_root: str = DEFAULT_DATA_ROOT,
    use_cache: bool = True,
) -> Dict[str, Any]:
    out_dir = os.path.join(BASE_DIR, MODEL_NAME, "fair")

    data = prepare_benchmark_data(
        data_root=data_root,
        pulse_list=PULSE_LIST,
        base_dir=BASE_DIR,
        seed=RANDOM_SEED,
        use_cache=use_cache,
    )

    model_clf = load_tabnet_model(
        TabNetClassifier,
        os.path.join(out_dir, "model_classifier"),
    )
    model_soc = load_tabnet_model(
        TabNetRegressor,
        os.path.join(out_dir, "model_soc"),
    )
    model_soh = load_tabnet_model(
        TabNetRegressor,
        os.path.join(out_dir, "model_soh"),
    )

    print("[TabNet fair summary-only] Predicting test set.")
    pred_cls_idx = model_clf.predict(data.Xte)
    pred_soc = model_soc.predict(data.Xte).reshape(-1)
    pred_soh = model_soh.predict(data.Xte).reshape(-1)

    extra = {
        "summary_only": True,
        "classifier_checkpoint": os.path.join(out_dir, "model_classifier.zip"),
        "soc_checkpoint": os.path.join(out_dir, "model_soc.zip"),
        "soh_checkpoint": os.path.join(out_dir, "model_soh.zip"),
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


def run_tabnet_enhanced_summary_only(
    data_root: str = DEFAULT_DATA_ROOT,
    use_cache: bool = True,
) -> Dict[str, Any]:
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

    model_clf = load_tabnet_model(
        TabNetClassifier,
        os.path.join(out_dir, "model_classifier_baseline"),
    )
    model_soc = load_tabnet_model(
        TabNetRegressor,
        os.path.join(out_dir, "model_soc_enhanced"),
    )
    model_soh = load_tabnet_model(
        TabNetRegressor,
        os.path.join(out_dir, "model_soh_enhanced"),
    )

    print("[TabNet enhanced summary-only] Predicting test set.")
    pred_cls_idx = model_clf.predict(data.Xte)
    pred_soc = model_soc.predict(Xte_soc).reshape(-1)
    pred_soh = model_soh.predict(Xte_soh).reshape(-1)

    extra = {
        "summary_only": True,
        "classifier_checkpoint": os.path.join(out_dir, "model_classifier_baseline.zip"),
        "soc_checkpoint": os.path.join(out_dir, "model_soc_enhanced.zip"),
        "soh_checkpoint": os.path.join(out_dir, "model_soh_enhanced.zip"),
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


def make_train_val_masks_by_id(
    mtr,
    val_id_frac: float = VAL_ID_FRAC,
    seed: int = RANDOM_SEED + VAL_SEED_OFFSET,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Create an internal validation split from training IDs only.

    This prevents TabNet from using the held-out TEST_RANDOM set for
    early stopping / best-epoch selection.
    """
    if "ID" not in mtr.columns:
        raise RuntimeError("mtr must contain an 'ID' column for ID-level validation split.")

    ids = np.array(sorted(mtr["ID"].astype(str).unique()), dtype=object)
    if len(ids) <= 1:
        raise RuntimeError("Not enough training IDs to create validation split.")

    rng = np.random.RandomState(int(seed))
    rng.shuffle(ids)

    n_val = int(max(1, round(len(ids) * float(val_id_frac))))
    n_val = min(n_val, len(ids) - 1)

    val_ids = set(map(str, ids[:n_val]))

    val_mask = mtr["ID"].astype(str).isin(val_ids).to_numpy()
    tr_mask = ~val_mask

    if int(tr_mask.sum()) == 0 or int(val_mask.sum()) == 0:
        raise RuntimeError("Empty TabNet train or validation subset after ID split.")

    info = {
        "val_id_frac": float(val_id_frac),
        "val_seed": int(seed),
        "n_tabnet_train": int(tr_mask.sum()),
        "n_tabnet_val": int(val_mask.sum()),
        "n_tabnet_train_ids": int(mtr.loc[tr_mask, "ID"].astype(str).nunique()),
        "n_tabnet_val_ids": int(mtr.loc[val_mask, "ID"].astype(str).nunique()),
    }

    print(
        "[TabNet split] "
        f"train samples={info['n_tabnet_train']} "
        f"({info['n_tabnet_train_ids']} IDs), "
        f"val samples={info['n_tabnet_val']} "
        f"({info['n_tabnet_val_ids']} IDs)"
    )

    return tr_mask, val_mask, info


def fit_tabnet_classifier(
    Xtr: np.ndarray,
    ytr_cls: np.ndarray,
    Xval: np.ndarray,
    yval_cls: np.ndarray,
    quick: bool = False,
) -> TabNetClassifier:
    """
    Fit a TabNet classifier for material-capacity classification.

    Xval/yval_cls are an internal validation set drawn from training IDs only.
    The held-out test set must not be passed here.
    """
    model = TabNetClassifier(**build_tabnet_params())

    model.fit(
        X_train=Xtr,
        y_train=ytr_cls,
        eval_set=[(Xval, yval_cls)],
        eval_name=["val"],
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
    Xval: np.ndarray,
    yval: np.ndarray,
    quick: bool = False,
) -> TabNetRegressor:
    """
    Fit a TabNet regressor for SOC or SOH estimation.

    Xval/yval are an internal validation set drawn from training IDs only.
    The held-out test set must not be passed here.
    """
    model = TabNetRegressor(**build_tabnet_params())

    model.fit(
        X_train=Xtr,
        y_train=ytr.reshape(-1, 1).astype(np.float32),
        eval_set=[(Xval, yval.reshape(-1, 1).astype(np.float32))],
        eval_name=["val"],
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

    tr_mask, val_mask, split_info = make_train_val_masks_by_id(
        data.mtr,
        val_id_frac=VAL_ID_FRAC,
        seed=RANDOM_SEED + VAL_SEED_OFFSET,
    )

    print("[TabNet fair] Training material classifier.")
    model_clf = fit_tabnet_classifier(
        Xtr=data.Xtr[tr_mask],
        ytr_cls=data.ytr_cls[tr_mask],
        Xval=data.Xtr[val_mask],
        yval_cls=data.ytr_cls[val_mask],
        quick=quick,
    )

    print("[TabNet fair] Training SOC regressor.")
    soc_tr = data.mtr["SOC"].to_numpy(dtype=np.float32)
    model_soc = fit_tabnet_regressor(
        Xtr=data.Xtr[tr_mask],
        ytr=soc_tr[tr_mask],
        Xval=data.Xtr[val_mask],
        yval=soc_tr[val_mask],
        quick=quick,
    )

    print("[TabNet fair] Training SOH regressor.")
    soh_tr = data.mtr["SOH"].to_numpy(dtype=np.float32)
    model_soh = fit_tabnet_regressor(
        Xtr=data.Xtr[tr_mask],
        ytr=soh_tr[tr_mask],
        Xval=data.Xtr[val_mask],
        yval=soh_tr[val_mask],
        quick=quick,
    )

    print("[TabNet fair] Predicting test set.")
    pred_cls_idx = model_clf.predict(data.Xte)
    pred_soc = model_soc.predict(data.Xte).reshape(-1)
    pred_soh = model_soh.predict(data.Xte).reshape(-1)

    save_tabnet_model(
        model_clf,
        os.path.join(out_dir, "model_classifier"),
    )
    save_tabnet_model(
        model_soc,
        os.path.join(out_dir, "model_soc"),
    )
    save_tabnet_model(
        model_soh,
        os.path.join(out_dir, "model_soh"),
    )

    extra = {
        "tabnet_params": build_tabnet_params(),
        "save_format": "tabnet_zip",
        "tabnet_internal_val_split": split_info,
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

    tr_mask, val_mask, split_info = make_train_val_masks_by_id(
        data.mtr,
        val_id_frac=VAL_ID_FRAC,
        seed=RANDOM_SEED + VAL_SEED_OFFSET,
    )

    print("[TabNet enhanced] Training baseline material classifier.")
    model_clf = fit_tabnet_classifier(
        Xtr=data.Xtr[tr_mask],
        ytr_cls=data.ytr_cls[tr_mask],
        Xval=data.Xtr[val_mask],
        yval_cls=data.ytr_cls[val_mask],
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
        Xtr=Xtr_soc[tr_mask],
        ytr=soc_tr_true[tr_mask],
        Xval=Xtr_soc[val_mask],
        yval=soc_tr_true[val_mask],
        quick=quick,
    )

    print("[TabNet enhanced] Training SOH regressor with material hint + pseudo SOC hint.")
    soh_tr_true = data.mtr["SOH"].to_numpy(dtype=np.float32)
    model_soh = fit_tabnet_regressor(
        Xtr=Xtr_soh[tr_mask],
        ytr=soh_tr_true[tr_mask],
        Xval=Xtr_soh[val_mask],
        yval=soh_tr_true[val_mask],
        quick=quick,
    )

    print("[TabNet enhanced] Predicting test set.")
    pred_soc = model_soc.predict(Xte_soc).reshape(-1)
    pred_soh = model_soh.predict(Xte_soh).reshape(-1)

    save_tabnet_model(
        model_clf,
        os.path.join(out_dir, "model_classifier_baseline"),
    )
    save_tabnet_model(
        model_soc,
        os.path.join(out_dir, "model_soc_enhanced"),
    )
    save_tabnet_model(
        model_soh,
        os.path.join(out_dir, "model_soh_enhanced"),
    )

    save_json(
        hint_report,
        os.path.join(out_dir, "enhanced_hint_report.json"),
    )

    extra = {
        "tabnet_params": build_tabnet_params(),
        "save_format": "tabnet_zip",
        "target_material_acc": TARGET_MATERIAL_ACC,
        "target_soc_rmse_raw": TARGET_SOC_RMSE_RAW,
        "enhanced_hint_report": hint_report,
        "tabnet_internal_val_split": split_info,
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
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Load existing trained TabNet zip models and regenerate predictions/metrics without training.",
    )

    args = parser.parse_args()
    use_cache = not args.no_cache

    rows = []

    if args.setting in ["fair", "both"]:
        if args.summary_only:
            rows.append(
                run_tabnet_fair_summary_only(
                    data_root=args.data_root,
                    use_cache=use_cache,
                )
            )
        else:
            rows.append(
                run_tabnet_fair(
                    data_root=args.data_root,
                    quick=args.quick,
                    use_cache=use_cache,
                )
            )

    if args.setting in ["enhanced", "both"]:
        if args.summary_only:
            rows.append(
                run_tabnet_enhanced_summary_only(
                    data_root=args.data_root,
                    use_cache=use_cache,
                )
            )
        else:
            rows.append(
                run_tabnet_enhanced(
                    data_root=args.data_root,
                    quick=args.quick,
                    use_cache=use_cache,
                )
            )

    _save_and_print_master_summary(rows)


if __name__ == "__main__":
    main()

