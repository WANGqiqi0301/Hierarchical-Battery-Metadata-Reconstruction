# proposed_framework/training/trainer.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from utils.metrics import rmse, mae, mape, medape
from proposed_framework.training.losses import bin_index_from_edges
from proposed_framework.training.evaluator import (
    soc_z_to_raw_tensor,
    soh_z_to_raw_tensor,
)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device: str,
    w_cls: float,
    w_soc: float,
    w_soh: float,
    grad_clip: float,
    criterion_cls,
    criterion_reg,
    soc_nll_weight: float = 1.0,
    # SOC bin weighting
    soc_bin_edges: Optional[List[Tuple[float, float]]] = None,
    soc_bin_weights: Optional[np.ndarray] = None,
    soc_norm: Optional[Tuple[float, float]] = None,
    normalize_soc: bool = True,
    zscore_normalize: bool = False,
    # SOH bin weighting
    soh_bin_edges: Optional[List[Tuple[float, float]]] = None,
    soh_bin_weights: Optional[np.ndarray] = None,
    soh_norm: Optional[Tuple[float, float]] = None,
    # MC samples for point estimate
    n_mc: int = 16,
):
    """
    Train one epoch.

    The SOC and SOH heads are conditional normalizing flows. Therefore,
    SOC/SOH losses are negative log-likelihoods rather than deterministic
    regression losses.

    Notes
    -----
    Training metrics are computed in the model target space. Raw-space metrics
    are computed in evaluator.py.
    """
    model.train()

    total_loss = 0.0
    n = 0

    y_true_cls = []
    y_pred_cls = []

    soc_true_all = []
    soc_pred_all = []

    soh_true_all = []
    soh_pred_all = []

    soc_bin_weight_tensor = None

    if soc_bin_edges is not None and soc_bin_weights is not None:
        soc_bin_weight_tensor = torch.tensor(
            soc_bin_weights,
            dtype=torch.float32,
            device=device,
        )

    soh_bin_weight_tensor = None

    if soh_bin_edges is not None and soh_bin_weights is not None:
        soh_bin_weight_tensor = torch.tensor(
            soh_bin_weights,
            dtype=torch.float32,
            device=device,
        )

    for x3, pt, y_cls, soc, soh in loader:
        x3 = x3.to(device)
        pt = pt.to(device)
        y_cls = y_cls.to(device)

        soc = soc.to(device).view(-1)
        soh = soh.to(device).view(-1)

        optimizer.zero_grad(set_to_none=True)

        logits_mat, soc_pred, soc_logp, cond_soc, soh_pred, cond_soh = model(
            x_img=x3,
            x_pt=pt,
            soc_tf=soc,
            n_mc=int(n_mc),
        )

        soc_pred = soc_pred.view(-1)
        soh_pred = soh_pred.view(-1)

        loss_cls = criterion_cls(logits_mat, y_cls)

        if soc_logp is None:
            raise RuntimeError(
                "soc_logp is None. In training, call model(..., soc_tf=soc)."
            )

        nll_soc_per_sample = (-soc_logp).view(-1) * float(soc_nll_weight)

        if soc_bin_weight_tensor is not None and soc_bin_edges is not None:
            soc_raw = soc_z_to_raw_tensor(
                soc_z=soc,
                soc_norm=soc_norm,
                normalize_soc=normalize_soc,
                zscore_normalize=zscore_normalize,
            ).view(-1)

            soc_bin_idx = bin_index_from_edges(soc_raw, soc_bin_edges)
            soc_sample_weights = soc_bin_weight_tensor[soc_bin_idx]

            loss_soc = (nll_soc_per_sample * soc_sample_weights).mean()
        else:
            loss_soc = nll_soc_per_sample.mean()

        soh_logp = model.soh_flow.log_prob(soh, cond_soh).view(-1)
        nll_soh_per_sample = -soh_logp

        if soh_bin_weight_tensor is not None and soh_bin_edges is not None:
            soh_raw = soh_z_to_raw_tensor(
                soh_z=soh,
                soh_norm=soh_norm,
                zscore_normalize=zscore_normalize,
            ).view(-1)

            soh_bin_idx = bin_index_from_edges(soh_raw, soh_bin_edges)
            soh_sample_weights = soh_bin_weight_tensor[soh_bin_idx]

            loss_soh = (nll_soh_per_sample * soh_sample_weights).mean()
        else:
            loss_soh = nll_soh_per_sample.mean()

        loss = (
            float(w_cls) * loss_cls
            + float(w_soc) * loss_soc
            + float(w_soh) * loss_soh
        )

        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(grad_clip),
            )

        optimizer.step()

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
    }