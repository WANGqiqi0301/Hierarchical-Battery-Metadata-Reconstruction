# -*- coding: utf-8 -*-
"""
benchmark/common.py

Shared utilities for benchmark comparison experiments.

This module centralizes:
1) loading 41U + pulse-width tabular features
2) cache handling
3) ID-based train/test split
4) train-only imputation and scaling
5) label encoding
6) common metrics
7) result saving

Benchmark input:
    U1-U41 + pulse_width = 42 tabular features

Important:
    This version uses the reorganized project data loader:
        from utils.data_loader import LoadConfig, load_pulsebat_classification
"""

from __future__ import annotations

import os
import json
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error

from utils.data_loader import LoadConfig, load_pulsebat_classification


@dataclass
class BenchmarkData:
    Xtr: np.ndarray
    Xte: np.ndarray

    ytr: np.ndarray
    yte: np.ndarray

    ytr_cls: np.ndarray
    yte_cls: np.ndarray

    mtr: pd.DataFrame
    mte: pd.DataFrame

    label_encoder: LabelEncoder
    num_classes: int

    imputer: Optional[SimpleImputer]
    scaler: Optional[StandardScaler]

    feature_dim: int
    test_ids: List[str]


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        ensure_dir(p)


def _jsonify(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_jsonify)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(mean_squared_error(a, b)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.mean(np.abs(a - b)))


def mape(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    denom = np.maximum(np.abs(a), eps)
    return float(np.mean(np.abs((b - a) / denom)) * 100.0)


def median_ape(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    denom = np.maximum(np.abs(a), eps)
    return float(np.median(np.abs((b - a) / denom)) * 100.0)


def evaluate_predictions(
    yte_cls: np.ndarray,
    pred_cls_idx: np.ndarray,
    soc_true: np.ndarray,
    soc_pred: np.ndarray,
    soh_true: np.ndarray,
    soh_pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "material_acc": float(accuracy_score(yte_cls, pred_cls_idx)),

        "soc_rmse": rmse(soc_true, soc_pred),
        "soc_mae": mae(soc_true, soc_pred),
        "soc_mape": mape(soc_true, soc_pred),
        "soc_median_ape": median_ape(soc_true, soc_pred),

        "soh_rmse": rmse(soh_true, soh_pred),
        "soh_mae": mae(soh_true, soh_pred),
        "soh_mape": mape(soh_true, soh_pred),
        "soh_median_ape": median_ape(soh_true, soh_pred),

        "n_test": int(len(soc_true)),
    }


def load_one_pulse_setting(
    data_root: str,
    soc,
    pulse_ms: int,
    u_start: int = 1,
    u_end: int = 41,
    drop_first_class: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    cfg = LoadConfig(
        data_root=data_root,
        soc=soc,
        pulse_width_ms=int(pulse_ms),
        u_start=int(u_start),
        u_end=int(u_end),
        drop_first_21_only_class=bool(drop_first_class),
        include_soc_in_X=False,
        verbose=False,
    )

    X_df, y_ser, meta = load_pulsebat_classification(cfg)
    X = X_df.to_numpy(dtype=float)
    y = y_ser.to_numpy()

    if len(X) == 0:
        return X, y, meta

    pt_col = np.full((X.shape[0], 1), fill_value=float(pulse_ms), dtype=float)
    X_42 = np.hstack([X, pt_col])

    meta = meta.copy()
    meta["pulse_ms"] = int(pulse_ms)

    return X_42, y, meta


def build_raw_benchmark_dataset(
    data_root: str,
    pulse_list: List[int],
    cache_file: str,
    u_start: int = 1,
    u_end: int = 41,
    drop_first_class: bool = True,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    if use_cache and os.path.exists(cache_file):
        print(f"[CACHE] Loading: {cache_file}")
        with open(cache_file, "rb") as f:
            d = pickle.load(f)
        return d["Xtr"], d["ytr"], d["mtr"], d["Xte"], d["yte"], d["mte"]

    print("[DATA] No cache. Building benchmark dataset with 41U + pulse_width.")
    soc_list = list(range(5, 90, 5))

    Xtr_list, ytr_list, mtr_list = [], [], []
    for soc in soc_list:
        for pt in pulse_list:
            X, y, meta = load_one_pulse_setting(
                data_root=data_root,
                soc=int(soc),
                pulse_ms=int(pt),
                u_start=u_start,
                u_end=u_end,
                drop_first_class=drop_first_class,
            )
            if len(y) > 0:
                Xtr_list.append(X)
                ytr_list.append(y)
                mtr_list.append(meta)
                print(f"  Train: SOC={soc}% | pulse={pt} ms", end="\r")

    Xte_list, yte_list, mte_list = [], [], []
    for pt in pulse_list:
        X, y, meta = load_one_pulse_setting(
            data_root=data_root,
            soc="TEST_RANDOM",
            pulse_ms=int(pt),
            u_start=u_start,
            u_end=u_end,
            drop_first_class=drop_first_class,
        )
        if len(y) > 0:
            Xte_list.append(X)
            yte_list.append(y)
            mte_list.append(meta)
            print(f"  Test: TEST_RANDOM | pulse={pt} ms", end="\r")

    if not Xtr_list or not Xte_list:
        raise RuntimeError("Empty benchmark dataset. Please check data_root and pulse_list.")

    Xtr = np.vstack(Xtr_list)
    ytr = np.concatenate(ytr_list)
    mtr = pd.concat(mtr_list, axis=0, ignore_index=True)

    Xte = np.vstack(Xte_list)
    yte = np.concatenate(yte_list)
    mte = pd.concat(mte_list, axis=0, ignore_index=True)

    ensure_dir(os.path.dirname(cache_file))
    with open(cache_file, "wb") as f:
        pickle.dump(
            {
                "Xtr": Xtr,
                "ytr": ytr,
                "mtr": mtr,
                "Xte": Xte,
                "yte": yte,
                "mte": mte,
            },
            f,
        )

    print(f"\n[CACHE] Saved: {cache_file}")

    return Xtr, ytr, mtr, Xte, yte, mte


def split_by_id(
    Xtr_raw: np.ndarray,
    ytr_raw: np.ndarray,
    mtr_raw: pd.DataFrame,
    Xte_raw: np.ndarray,
    yte_raw: np.ndarray,
    mte_raw: pd.DataFrame,
    seed: int = 42,
    test_id_frac: float = 0.2,
    test_id_count: int = 0,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    if "ID" not in mtr_raw.columns or "ID" not in mte_raw.columns:
        raise RuntimeError("Meta must contain ID column for ID split.")

    ids = np.unique(
        np.concatenate(
            [
                mtr_raw["ID"].astype(str).to_numpy(),
                mte_raw["ID"].astype(str).to_numpy(),
            ]
        )
    )

    rng = np.random.RandomState(int(seed))
    rng.shuffle(ids)

    if int(test_id_count) > 0:
        n_test = min(max(1, int(test_id_count)), len(ids) - 1)
    else:
        n_test = int(max(1, round(len(ids) * float(test_id_frac))))
        n_test = min(n_test, len(ids) - 1)

    test_ids = set(ids[:n_test].astype(str).tolist())

    tr_mask = ~mtr_raw["ID"].astype(str).isin(test_ids).to_numpy()
    te_mask = mte_raw["ID"].astype(str).isin(test_ids).to_numpy()

    Xtr = Xtr_raw[tr_mask]
    ytr = ytr_raw[tr_mask]
    mtr = mtr_raw.loc[tr_mask].reset_index(drop=True)

    Xte = Xte_raw[te_mask]
    yte = yte_raw[te_mask]
    mte = mte_raw.loc[te_mask].reset_index(drop=True)

    overlap = set(mtr["ID"].astype(str).unique()) & set(mte["ID"].astype(str).unique())
    if overlap:
        raise RuntimeError(f"ID leakage detected after split: {list(overlap)[:5]}")

    return Xtr, ytr, mtr, Xte, yte, mte, sorted(test_ids)


def preprocess_features(
    Xtr: np.ndarray,
    Xte: np.ndarray,
    impute: bool = True,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[SimpleImputer], Optional[StandardScaler]]:
    imputer = None
    scaler = None

    if impute:
        imputer = SimpleImputer(strategy="mean")
        Xtr = imputer.fit_transform(Xtr)
        Xte = imputer.transform(Xte)

    if scale:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

    return Xtr.astype(np.float32), Xte.astype(np.float32), imputer, scaler


def encode_labels(
    ytr: np.ndarray,
    yte: np.ndarray,
    Xte: np.ndarray,
    mte: pd.DataFrame,
) -> Tuple[LabelEncoder, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    le = LabelEncoder()
    ytr_cls = le.fit_transform(ytr)

    known_mask = np.isin(yte, le.classes_)
    Xte = Xte[known_mask]
    yte = yte[known_mask]
    mte = mte.loc[known_mask].reset_index(drop=True)

    yte_cls = le.transform(yte)

    return le, ytr_cls, yte_cls, yte, Xte, mte


def prepare_benchmark_data(
    data_root: str,
    pulse_list: List[int],
    base_dir: str = os.path.join("results", "benchmark"),
    cache_name: str = "data_cache_42_feats.pkl",
    seed: int = 42,
    test_id_frac: float = 0.2,
    test_id_count: int = 0,
    u_start: int = 1,
    u_end: int = 41,
    drop_first_class: bool = True,
    impute: bool = True,
    scale: bool = True,
    use_cache: bool = True,
) -> BenchmarkData:
    cache_file = os.path.join(base_dir, cache_name)
    ensure_dir(base_dir)

    Xtr_raw, ytr_raw, mtr_raw, Xte_raw, yte_raw, mte_raw = build_raw_benchmark_dataset(
        data_root=data_root,
        pulse_list=pulse_list,
        cache_file=cache_file,
        u_start=u_start,
        u_end=u_end,
        drop_first_class=drop_first_class,
        use_cache=use_cache,
    )

    Xtr, ytr, mtr, Xte, yte, mte, test_ids = split_by_id(
        Xtr_raw=Xtr_raw,
        ytr_raw=ytr_raw,
        mtr_raw=mtr_raw,
        Xte_raw=Xte_raw,
        yte_raw=yte_raw,
        mte_raw=mte_raw,
        seed=seed,
        test_id_frac=test_id_frac,
        test_id_count=test_id_count,
    )

    Xtr, Xte, imputer, scaler = preprocess_features(
        Xtr=Xtr,
        Xte=Xte,
        impute=impute,
        scale=scale,
    )

    le, ytr_cls, yte_cls, yte, Xte, mte = encode_labels(
        ytr=ytr,
        yte=yte,
        Xte=Xte,
        mte=mte,
    )

    data = BenchmarkData(
        Xtr=Xtr,
        Xte=Xte,
        ytr=ytr,
        yte=yte,
        ytr_cls=ytr_cls,
        yte_cls=yte_cls,
        mtr=mtr,
        mte=mte,
        label_encoder=le,
        num_classes=len(le.classes_),
        imputer=imputer,
        scaler=scaler,
        feature_dim=int(Xtr.shape[1]),
        test_ids=test_ids,
    )

    print(f"[DATA] Train samples = {len(data.Xtr)}")
    print(f"[DATA] Test  samples = {len(data.Xte)}")
    print(f"[DATA] Feature dim   = {data.feature_dim}")
    print(f"[DATA] Classes       = {list(data.label_encoder.classes_)}")

    return data


def save_predictions_and_summary(
    out_dir: str,
    model_name: str,
    setting: str,
    data: BenchmarkData,
    pred_cls_idx: np.ndarray,
    pred_soc: np.ndarray,
    pred_soh: np.ndarray,
    extra_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ensure_dir(out_dir)

    pred_cls_idx = np.asarray(pred_cls_idx).reshape(-1)
    pred_soc = np.asarray(pred_soc).reshape(-1)
    pred_soh = np.asarray(pred_soh).reshape(-1)

    if len(pred_cls_idx) != len(data.Xte):
        raise RuntimeError(f"Material prediction length mismatch: {len(pred_cls_idx)} vs {len(data.Xte)}")
    if len(pred_soc) != len(data.Xte):
        raise RuntimeError(f"SOC prediction length mismatch: {len(pred_soc)} vs {len(data.Xte)}")
    if len(pred_soh) != len(data.Xte):
        raise RuntimeError(f"SOH prediction length mismatch: {len(pred_soh)} vs {len(data.Xte)}")

    pred_cls_label = data.label_encoder.inverse_transform(pred_cls_idx.astype(int))

    soc_true = data.mte["SOC"].to_numpy(dtype=np.float64)
    soh_true = data.mte["SOH"].to_numpy(dtype=np.float64)

    metrics = evaluate_predictions(
        yte_cls=data.yte_cls,
        pred_cls_idx=pred_cls_idx,
        soc_true=soc_true,
        soc_pred=pred_soc,
        soh_true=soh_true,
        soh_pred=pred_soh,
    )

    pred_df = pd.DataFrame(
        {
            "ID": data.mte["ID"].astype(str).to_numpy() if "ID" in data.mte.columns else np.arange(len(data.mte)),
            "True_Material": data.yte,
            "Pred_Material": pred_cls_label,
            "True_SOC": soc_true,
            "Pred_SOC": pred_soc,
            "True_SOH": soh_true,
            "Pred_SOH": pred_soh,
        }
    )

    pred_path = os.path.join(out_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    summary = {
        "model": model_name,
        "setting": setting,
        "input_feature_dim": int(data.feature_dim),
        "metrics": metrics,
        "n_train": int(len(data.Xtr)),
        "n_test": int(len(data.Xte)),
        "classes": [str(x) for x in data.label_encoder.classes_.tolist()],
    }

    if extra_report is not None:
        summary["extra"] = extra_report

    summary_path = os.path.join(out_dir, "summary.json")
    save_json(summary, summary_path)

    report_lines = [
        f"================ {model_name.upper()} {setting.upper()} RESULTS ================",
        f"Input feature dim       : {data.feature_dim}",
        f"Material accuracy       : {metrics['material_acc']:.6f}",
        "------------------------------------------------------------",
        f"SOC RMSE                : {metrics['soc_rmse']:.6f}",
        f"SOC MAE                 : {metrics['soc_mae']:.6f}",
        f"SOC MAPE                : {metrics['soc_mape']:.6f}",
        f"SOC MedAPE              : {metrics['soc_median_ape']:.6f}",
        "------------------------------------------------------------",
        f"SOH RMSE                : {metrics['soh_rmse']:.6f}",
        f"SOH MAE                 : {metrics['soh_mae']:.6f}",
        f"SOH MAPE                : {metrics['soh_mape']:.6f}",
        f"SOH MedAPE              : {metrics['soh_median_ape']:.6f}",
        f"n_test                  : {metrics['n_test']}",
        "============================================================",
    ]

    report_text = "\n".join(report_lines)
    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"[SAVED] {pred_path}")
    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {report_path}")

    return summary