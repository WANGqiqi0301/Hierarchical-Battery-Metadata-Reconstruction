# utils/metrics.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root mean squared error.
    """
    if len(y_true) == 0:
        return 0.0

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    diff = y_pred - y_true
    return float(np.sqrt(np.mean(diff * diff)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean absolute error.
    """
    if len(y_true) == 0:
        return 0.0

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    return float(np.mean(np.abs(y_pred - y_true)))


def ape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Absolute percentage error for each sample.

    Returns
    -------
    np.ndarray
        Sample-wise absolute percentage error in percent.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    denom = np.maximum(np.abs(y_true), eps)
    return np.abs((y_pred - y_true) / denom) * 100.0


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Mean absolute percentage error in percent.
    """
    if len(y_true) == 0:
        return 0.0

    return float(np.mean(ape(y_true, y_pred, eps=eps)))


def medape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Median absolute percentage error in percent.

    This is the main SOC/SOH error metric used for manuscript reporting.
    """
    if len(y_true) == 0:
        return 0.0

    return float(np.median(ape(y_true, y_pred, eps=eps)))