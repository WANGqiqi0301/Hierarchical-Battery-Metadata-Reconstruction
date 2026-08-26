# proposed_framework/training/evaluator.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from utils.metrics import rmse, mae, mape, medape


def soh_z_to_raw_tensor(
    soh_z: torch.Tensor,
    soh_norm: Optional[Tuple[float, float]],
    zscore_normalize: bool,
) -> torch.Tensor:
    """
    Convert SOH from model target space back to raw SOH space.

    Parameters
    ----------
    soh_z:
        SOH values in model target space.

    soh_norm:
        Training-set SOH normalization statistics: (mean, std).
        Required only when zscore_normalize=True.

    zscore_normalize:
        Whether SOH was z-score normalized during training.

    Returns
    -------
    torch.Tensor
        SOH values in raw SOH space.
    """
    soh = soh_z

    if zscore_normalize:
        if soh_norm is None:
            raise RuntimeError("soh_norm is required when zscore_normalize=True.")

        mean, std = float(soh_norm[0]), float(soh_norm[1])
        soh = soh * std + mean

    return soh


def soc_z_to_raw_tensor(
    soc_z: torch.Tensor,
    soc_norm: Optional[Tuple[float, float]],
    normalize_soc: bool,
    zscore_normalize: bool,
) -> torch.Tensor:
    """
    Convert SOC from model target space back to raw SOC percentage.

    Parameters
    ----------
    soc_z:
        SOC values in model target space.

    soc_norm:
        Training-set SOC normalization statistics: (mean, std).
        Required only when zscore_normalize=True.

    normalize_soc:
        Whether SOC was divided by 100 during training.

    zscore_normalize:
        Whether SOC was z-score normalized during training.

    Returns
    -------
    torch.Tensor
        SOC values in raw percentage space when normalize_soc=True.
    """
    soc = soc_z

    if zscore_normalize:
        if soc_norm is None:
            raise RuntimeError("soc_norm is required when zscore_normalize=True.")

        mean, std = float(soc_norm[0]), float(soc_norm[1])
        soc = soc * std + mean

    if normalize_soc:
        soc = soc * 100.0

    return soc


def inverse_targets(
    soc_z: np.ndarray,
    soh_z: np.ndarray,
    soc_norm: Optional[Tuple[float, float]],
    soh_norm: Optional[Tuple[float, float]],
    normalize_soc: bool,
    zscore_normalize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse-transform SOC and SOH targets from model target space to raw units.

    Training target space:
    - SOC: SOC / 100 if normalize_soc=True, then optional z-score.
    - SOH: optional z-score.

    Output:
    - SOC: percentage if normalize_soc=True.
    - SOH: original SOH unit.
    """
    soc = np.asarray(soc_z, dtype=np.float64)
    soh = np.asarray(soh_z, dtype=np.float64)

    if zscore_normalize:
        if soc_norm is None or soh_norm is None:
            raise RuntimeError(
                "soc_norm and soh_norm are required when zscore_normalize=True."
            )

        soc_mean, soc_std = float(soc_norm[0]), float(soc_norm[1])
        soh_mean, soh_std = float(soh_norm[0]), float(soh_norm[1])

        soc = soc * soc_std + soc_mean
        soh = soh * soh_std + soh_mean

    if normalize_soc:
        soc = soc * 100.0

    return soc, soh


@torch.no_grad()
def eval_one_epoch(
    model,
    loader,
    device: str,
    w_cls: float,
    w_soc: float,
    w_soh: float,
    criterion_cls,
    criterion_reg,
    soc_nll_weight: float = 1.0,
    soc_norm: Optional[Tuple[float, float]] = None,
    soh_norm: Optional[Tuple[float, float]] = None,
    normalize_soc: bool = True,
    zscore_normalize: bool = False,
    n_mc: int = 32,
):
    """
    Evaluate one epoch.

    This function reports both target-space metrics and raw-space metrics.

    Notes
    -----
    Raw-space MedAPE is the main SOC/SOH error metric used for manuscript
    reporting. MAPE is retained as a diagnostic metric because it is more
    sensitive to outliers.
    """
    model.eval()

    total_loss = 0.0
    n = 0

    y_true_cls = []
    y_pred_cls = []

    soc_true_all = []
    soc_pred_all = []

    soh_true_all = []
    soh_pred_all = []

    for x3, pt, y_cls, soc, soh in loader:
        x3 = x3.to(device)
        pt = pt.to(device)
        y_cls = y_cls.to(device)

        soc = soc.to(device).view(-1)
        soh = soh.to(device).view(-1)

        logits_mat, soc_pred, _, cond_soc, soh_pred, cond_soh = model(
            x_img=x3,
            x_pt=pt,
            soc_tf=None,
            n_mc=int(n_mc),
        )

        soc_pred = soc_pred.view(-1)
        soh_pred = soh_pred.view(-1)

        loss_cls = criterion_cls(logits_mat, y_cls)

        soc_logp_eval = model.soc_flow.log_prob(soc, cond_soc).view(-1)
        loss_soc = (-soc_logp_eval).mean() * float(soc_nll_weight)

        soh_logp_eval = model.soh_flow.log_prob(soh, cond_soh).view(-1)
        loss_soh = (-soh_logp_eval).mean()

        loss = (
            float(w_cls) * loss_cls
            + float(w_soc) * loss_soc
            + float(w_soh) * loss_soh
        )

        batch_size = int(y_cls.size(0))
        total_loss += float(loss.item()) * batch_size
        n += batch_size

        y_true_cls.append(y_cls.detach().cpu().numpy())
        y_pred_cls.append(logits_mat.detach().cpu().argmax(1).numpy())

        soc_true_all.append(soc.detach().cpu().view(-1).numpy())
        soc_pred_all.append(soc_pred.detach().cpu().view(-1).numpy())

        soh_true_all.append(soh.detach().cpu().view(-1).numpy())
        soh_pred_all.append(soh_pred.detach().cpu().view(-1).numpy())

    if n == 0:
        return {
            "loss": 0.0,
            "cls_acc": 0.0,
            "soc_rmse": 0.0,
            "soc_mae": 0.0,
            "soc_mape": 0.0,
            "soc_medape": 0.0,
            "soh_rmse": 0.0,
            "soh_mae": 0.0,
            "soh_mape": 0.0,
            "soh_medape": 0.0,
            "soc_rmse_raw": 0.0,
            "soc_mae_raw": 0.0,
            "soc_mape_raw": 0.0,
            "soc_medape_raw": 0.0,
            "soh_rmse_raw": 0.0,
            "soh_mae_raw": 0.0,
            "soh_mape_raw": 0.0,
            "soh_medape_raw": 0.0,
        }

    y_true_cls = np.concatenate(y_true_cls)
    y_pred_cls = np.concatenate(y_pred_cls)

    soc_true = np.concatenate(soc_true_all)
    soc_pred = np.concatenate(soc_pred_all)

    soh_true = np.concatenate(soh_true_all)
    soh_pred = np.concatenate(soh_pred_all)

    if soc_true.shape[0] != soc_pred.shape[0]:
        raise RuntimeError(
            f"SOC length mismatch: true={soc_true.shape}, pred={soc_pred.shape}"
        )

    if soh_true.shape[0] != soh_pred.shape[0]:
        raise RuntimeError(
            f"SOH length mismatch: true={soh_true.shape}, pred={soh_pred.shape}"
        )

    soc_true_raw, soh_true_raw = inverse_targets(
        soc_true,
        soh_true,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
    )

    soc_pred_raw, soh_pred_raw = inverse_targets(
        soc_pred,
        soh_pred,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
    )

    return {
        "loss": total_loss / n,
        "cls_acc": float(accuracy_score(y_true_cls, y_pred_cls)),
        "soc_rmse": rmse(soc_true, soc_pred),
        "soc_mae": mae(soc_true, soc_pred),
        "soc_mape": mape(soc_true, soc_pred),
        "soc_medape": medape(soc_true, soc_pred),
        "soh_rmse": rmse(soh_true, soh_pred),
        "soh_mae": mae(soh_true, soh_pred),
        "soh_mape": mape(soh_true, soh_pred),
        "soh_medape": medape(soh_true, soh_pred),
        "soc_rmse_raw": rmse(soc_true_raw, soc_pred_raw),
        "soc_mae_raw": mae(soc_true_raw, soc_pred_raw),
        "soc_mape_raw": mape(soc_true_raw, soc_pred_raw),
        "soc_medape_raw": medape(soc_true_raw, soc_pred_raw),
        "soh_rmse_raw": rmse(soh_true_raw, soh_pred_raw),
        "soh_mae_raw": mae(soh_true_raw, soh_pred_raw),
        "soh_mape_raw": mape(soh_true_raw, soh_pred_raw),
        "soh_medape_raw": medape(soh_true_raw, soh_pred_raw),
    }