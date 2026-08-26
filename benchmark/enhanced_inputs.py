# -*- coding: utf-8 -*-
"""
benchmark/enhanced_inputs.py

Controlled-upstream / enhanced benchmark inputs.

This replaces the previous "unfair" naming.

Logic:
1) Build a material hint with a controlled target accuracy.
2) Build a pseudo SOC hint with a controlled target RMSE.
3) SOC regressor receives:
       base features + material hint
4) SOH regressor receives:
       base features + material hint + pseudo SOC hint
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from benchmark.common import rmse, mae, mape, median_ape


def one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int).reshape(-1)
    out = np.zeros((len(indices), int(num_classes)), dtype=np.float32)
    out[np.arange(len(indices)), indices] = 1.0
    return out


def make_controlled_material_hint(
    y_true_cls: np.ndarray,
    num_classes: int,
    target_acc: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.RandomState(int(seed))
    y_true_cls = np.asarray(y_true_cls, dtype=int).reshape(-1)

    n = len(y_true_cls)
    n_correct = int(round(float(target_acc) * n))
    n_correct = min(max(0, n_correct), n)

    perm = rng.permutation(n)
    correct_idx = perm[:n_correct]
    wrong_idx = perm[n_correct:]

    hint_cls = np.empty_like(y_true_cls)
    hint_cls[correct_idx] = y_true_cls[correct_idx]

    for idx in wrong_idx:
        candidates = list(range(int(num_classes)))
        candidates.remove(int(y_true_cls[idx]))
        hint_cls[idx] = rng.choice(candidates)

    hint_onehot = one_hot(hint_cls, num_classes=num_classes)
    realized_acc = float(np.mean(hint_cls == y_true_cls))

    return hint_cls, hint_onehot, realized_acc


def synthesize_soc_with_target_rmse(
    soc_true: np.ndarray,
    target_rmse: float,
    low: float = 0.0,
    high: float = 100.0,
    seed: int = 42,
    search_steps: int = 40,
) -> np.ndarray:
    soc_true = np.asarray(soc_true, dtype=np.float32).reshape(-1)

    def generate_with_sigma(sigma: float) -> np.ndarray:
        rng = np.random.RandomState(int(seed))
        noise = rng.normal(loc=0.0, scale=float(sigma), size=len(soc_true))
        pred = np.clip(soc_true + noise, float(low), float(high))
        return pred.astype(np.float32)

    sigma_lo = 0.0
    sigma_hi = max(float(target_rmse), 1.0)

    for _ in range(20):
        pred_hi = generate_with_sigma(sigma_hi)
        if rmse(soc_true, pred_hi) >= float(target_rmse):
            break
        sigma_hi *= 1.5

    best_pred = generate_with_sigma(sigma_hi)
    best_gap = abs(rmse(soc_true, best_pred) - float(target_rmse))

    for _ in range(int(search_steps)):
        sigma_mid = 0.5 * (sigma_lo + sigma_hi)
        pred_mid = generate_with_sigma(sigma_mid)
        rmse_mid = rmse(soc_true, pred_mid)

        gap = abs(rmse_mid - float(target_rmse))
        if gap < best_gap:
            best_gap = gap
            best_pred = pred_mid.copy()

        if rmse_mid < float(target_rmse):
            sigma_lo = sigma_mid
        else:
            sigma_hi = sigma_mid

    return best_pred.astype(np.float32)


def summarize_soc_hint_quality(
    soc_true: np.ndarray,
    soc_pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "rmse": float(rmse(soc_true, soc_pred)),
        "mae": float(mae(soc_true, soc_pred)),
        "mape": float(mape(soc_true, soc_pred)),
        "median_ape": float(median_ape(soc_true, soc_pred)),
    }


def build_enhanced_inputs(
    Xtr: np.ndarray,
    Xte: np.ndarray,
    ytr_cls: np.ndarray,
    yte_cls: np.ndarray,
    soc_tr_true: np.ndarray,
    soc_te_true: np.ndarray,
    num_classes: int,
    target_material_acc: float,
    target_soc_rmse: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    mat_hint_cls_tr, mat_hint_oh_tr, mat_acc_tr = make_controlled_material_hint(
        y_true_cls=ytr_cls,
        num_classes=num_classes,
        target_acc=target_material_acc,
        seed=seed + 11,
    )

    mat_hint_cls_te, mat_hint_oh_te, mat_acc_te = make_controlled_material_hint(
        y_true_cls=yte_cls,
        num_classes=num_classes,
        target_acc=target_material_acc,
        seed=seed + 12,
    )

    soc_hint_tr = synthesize_soc_with_target_rmse(
        soc_true=soc_tr_true,
        target_rmse=target_soc_rmse,
        low=0.0,
        high=100.0,
        seed=seed + 21,
    )

    soc_hint_te = synthesize_soc_with_target_rmse(
        soc_true=soc_te_true,
        target_rmse=target_soc_rmse,
        low=0.0,
        high=100.0,
        seed=seed + 22,
    )

    Xtr_soc = np.hstack([Xtr, mat_hint_oh_tr]).astype(np.float32)
    Xte_soc = np.hstack([Xte, mat_hint_oh_te]).astype(np.float32)

    Xtr_soh = np.hstack([Xtr, mat_hint_oh_tr, soc_hint_tr.reshape(-1, 1)]).astype(np.float32)
    Xte_soh = np.hstack([Xte, mat_hint_oh_te, soc_hint_te.reshape(-1, 1)]).astype(np.float32)

    report = {
        "material_hint": {
            "target_accuracy": float(target_material_acc),
            "train_accuracy": float(mat_acc_tr),
            "test_accuracy": float(mat_acc_te),
        },
        "soc_hint_train": {
            "target_rmse": float(target_soc_rmse),
            **summarize_soc_hint_quality(soc_tr_true, soc_hint_tr),
        },
        "soc_hint_test": {
            "target_rmse": float(target_soc_rmse),
            **summarize_soc_hint_quality(soc_te_true, soc_hint_te),
        },
        "feature_dims": {
            "base": int(Xtr.shape[1]),
            "soc_input": int(Xtr_soc.shape[1]),
            "soh_input": int(Xtr_soh.shape[1]),
        },
    }

    return Xtr_soc, Xte_soc, Xtr_soh, Xte_soh, report