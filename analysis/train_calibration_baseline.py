# -*- coding: utf-8 -*-
"""
train_calibration_baseline.py

Train a Gaussian calibration baseline for SOC/SOH estimation.

This script is the cleaned replacement for the old
h13_3LayerHier_Regression_SOCvariance.py workflow.

It uses the new organized data utilities and saves all outputs under:

    results/calibration_baseline/

Outputs:
    results/calibration_baseline/checkpoints/best.pt
    results/calibration_baseline/checkpoints/last.pt
    results/calibration_baseline/logs/train_log.csv
    results/calibration_baseline/metrics/final_metrics.csv
    results/calibration_baseline/metrics/test_predictions.csv
    results/calibration_baseline/metrics/calibration_metrics.csv
    results/calibration_baseline/metrics/calibration_curve_soc.csv
    results/calibration_baseline/metrics/calibration_curve_soh.csv
    results/calibration_baseline/metrics/calibration_soc_calibrated.png
    results/calibration_baseline/metrics/calibration_soh_calibrated.png
    results/calibration_baseline/run_config.json
    results/calibration_baseline/label_mapping.json

Run:
    python analysis/train_calibration_baseline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Project path
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# =============================================================================
# New organized project imports
# =============================================================================
from utils.cache import ensure_dir, load_or_build_cache, drop_nan_inf_rows
from utils.metrics import medape, mape, rmse, mae
from utils.seed import set_random_seed

from proposed_framework.data.build_dataset import (
    build_train_mix_soc_mix_pt,
    build_test_random_mix_pt,
    pick_test_ids,
    apply_id_split,
)


# =============================================================================
# Configuration
# =============================================================================
DATA_ROOT = PROJECT_ROOT / "data"

EXP_DIR = PROJECT_ROOT / "results" / "calibration_baseline"

DEFAULT_PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

SOC_COL = "SOC"
SOH_COL = "SOH"

U_START = 1
U_END = 41
DROP_FIRST_CLASS = True

USE_PT_AS_FEATURE = True

BATCH_SIZE = 128
NUM_WORKERS = 0
SEED = 42

TEST_ID_FRAC = 0.2
TEST_ID_COUNT = 0

WIDTH = 32
BLOCKS = 4
DROP2D = 0.0
HEAD_DROPOUT = 0.2

SOC_HIDDEN = 64
SOH_HIDDEN = 64

LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0

MAX_EPOCHS = 400
EARLY_STOPPING = False
PATIENCE = 20
RESUME = True

W_CLS = 1.0
W_SOC = 1.0
W_SOH = 1.0

NORMALIZE_SOC = True
ZSCORE_NORMALIZE = True

# The old h13 baseline used raw U1-U41 values without train-only U z-score.
# Keep this False to reproduce the old Gaussian baseline behavior.
NORMALIZE_U_WITH_TRAIN_STATS = False


# =============================================================================
# Utility functions
# =============================================================================
def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def torch_load_compatible(path: str | Path, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def heteroscedastic_nll(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood with predicted log-variance.

    Loss:
        0.5 * (exp(-logvar) * (y - mu)^2 + logvar)
    """
    logvar = torch.clamp(logvar, min=-10.0, max=5.0)
    inv_var = torch.exp(-logvar)
    return 0.5 * (inv_var * (y - mu) ** 2 + logvar).mean()


def inverse_targets(
    soc_z: np.ndarray,
    soh_z: np.ndarray,
    soc_norm: Optional[Tuple[float, float]],
    soh_norm: Optional[Tuple[float, float]],
    normalize_soc: bool,
    zscore_normalize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert model target space back to raw SOC/SOH units.

    SOC:
        target space = SOC / 100, then optional z-score.
        raw output = SOC percentage.

    SOH:
        target space = optional z-score.
        raw output = original SOH unit.
    """
    soc = np.asarray(soc_z, dtype=np.float64).copy()
    soh = np.asarray(soh_z, dtype=np.float64).copy()

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


# =============================================================================
# Feature builder and dataset
# =============================================================================
def build_3ch_5x8_from_u41(u: np.ndarray) -> np.ndarray:
    """
    Convert U1-U41 into a structured 3-channel 5x8 representation.

    Channel 1:
        Raw pulse response values U2-U41.

    Channel 2:
        Voltage-difference features.

    Channel 3:
        U1 rested-voltage baseline repeated.
    """
    u = np.asarray(u, dtype=np.float32)

    if u.shape[0] != 41:
        raise ValueError(f"Expected 41 U values, got {u.shape[0]}.")

    u1 = float(u[0])
    u2_41 = u[1:]

    ch1 = u2_41.reshape(5, 8)

    diff = np.empty(40, dtype=np.float32)
    diff[0] = u[1] - u[0]
    diff[1:] = u[2:] - u[1:-1]
    ch2 = diff.reshape(5, 8)

    ch3 = np.full((5, 8), u1, dtype=np.float32)

    return np.stack([ch1, ch2, ch3], axis=0)


class CalibrationBaselineDataset(Dataset):
    """
    Dataset for the Gaussian calibration baseline.

    Each sample returns:
        x3: structured U input, shape (3, 5, 8)
        pt: normalized log-pulse-width feature, shape (1,)
        y_cls: material-capacity class label
        soc: SOC target in training target space
        soh: SOH target in training target space
    """

    def __init__(
        self,
        X_u: np.ndarray,
        y_cls: np.ndarray,
        meta: pd.DataFrame,
        soc_col: str,
        soh_col: str,
        pt_col: str = "pulse_ms",
        use_pt_as_feature: bool = True,
        pt_norm: Optional[Tuple[float, float]] = None,
        normalize_soc: bool = True,
        zscore_normalize: bool = True,
        soc_norm: Optional[Tuple[float, float]] = None,
        soh_norm: Optional[Tuple[float, float]] = None,
    ):
        self.X_u = np.asarray(X_u, dtype=np.float32)
        self.y_cls = np.asarray(y_cls, dtype=np.int64)
        self.meta = meta.reset_index(drop=True)

        self.soc_col = soc_col
        self.soh_col = soh_col

        if soc_col not in self.meta.columns or soh_col not in self.meta.columns:
            raise RuntimeError(
                f"Meta must contain soc_col='{soc_col}' and soh_col='{soh_col}'."
            )

        soc = self.meta[soc_col].astype(float).to_numpy(dtype=np.float32)
        soh = self.meta[soh_col].astype(float).to_numpy(dtype=np.float32)

        self.normalize_soc = bool(normalize_soc)
        self.zscore_normalize = bool(zscore_normalize)

        if self.normalize_soc:
            soc = soc / 100.0

        if self.zscore_normalize:
            if soc_norm is None or soh_norm is None:
                raise RuntimeError(
                    "zscore_normalize=True requires soc_norm and soh_norm."
                )

            soc_mean, soc_std = float(soc_norm[0]), float(soc_norm[1])
            soh_mean, soh_std = float(soh_norm[0]), float(soh_norm[1])

            soc = (soc - soc_mean) / (soc_std + 1e-8)
            soh = (soh - soh_mean) / (soh_std + 1e-8)

        self.soc = soc.astype(np.float32)
        self.soh = soh.astype(np.float32)

        self.use_pt = bool(use_pt_as_feature)

        if self.use_pt and pt_col in self.meta.columns:
            self.pt_ms = self.meta[pt_col].astype(float).to_numpy(dtype=np.float32)
            pt_log = np.log1p(self.pt_ms)

            if pt_norm is None:
                self.pt_mean = float(pt_log.mean())
                self.pt_std = float(pt_log.std() + 1e-8)
            else:
                self.pt_mean = float(pt_norm[0])
                self.pt_std = float(pt_norm[1])
        else:
            self.pt_ms = None
            self.pt_mean = 0.0
            self.pt_std = 1.0

    def __len__(self) -> int:
        return int(self.X_u.shape[0])

    def __getitem__(self, idx: int):
        x3 = torch.from_numpy(build_3ch_5x8_from_u41(self.X_u[idx]))

        if self.use_pt and self.pt_ms is not None:
            p = (np.log1p(float(self.pt_ms[idx])) - self.pt_mean) / self.pt_std
            pt = torch.tensor([p], dtype=torch.float32)
        else:
            pt = torch.tensor([0.0], dtype=torch.float32)

        y_cls = torch.tensor(int(self.y_cls[idx]), dtype=torch.long)
        soc = torch.tensor(float(self.soc[idx]), dtype=torch.float32)
        soh = torch.tensor(float(self.soh[idx]), dtype=torch.float32)

        return x3, pt, y_cls, soc, soh


# =============================================================================
# Model
# =============================================================================
class ResBlock(nn.Module):
    def __init__(self, c: int, drop2d: float = 0.0):
        super().__init__()

        groups = 8 if c % 8 == 0 else 4

        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, c)

        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, c)

        self.act = nn.ReLU(inplace=True)

        if drop2d and drop2d > 0:
            self.drop = nn.Dropout2d(float(drop2d))
        else:
            self.drop = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(self.conv1(x)))
        h = self.drop(h)
        h = self.gn2(self.conv2(h))
        return self.act(x + h)


class MicroResNetEncoder2D3Ch(nn.Module):
    """
    Small 2D encoder for 3-channel pulse-response maps.
    """

    def __init__(
        self,
        width: int = 32,
        blocks: int = 4,
        drop2d: float = 0.0,
    ):
        super().__init__()

        groups = 8 if width % 8 == 0 else 4

        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.GroupNorm(groups, width),
            nn.ReLU(inplace=True),
        )

        self.body = nn.Sequential(
            *[
                ResBlock(width, drop2d=drop2d)
                for _ in range(int(blocks))
            ]
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        z = self.stem(x_img)
        z = self.body(z)
        z = self.pool(z).flatten(1)
        return z


class GaussianCalibrationBaseline(nn.Module):
    """
    Hierarchical Gaussian probabilistic baseline.

    Heads:
        material:
            logits -> p = softmax(logits)

        SOC:
            heteroscedastic Gaussian regression from [z, p, pt]
            outputs soc_mu and soc_logvar.

        SOH:
            heteroscedastic Gaussian regression from
            [z, p, soc_mu, soc_sigma, pt]
            outputs soh_mu and soh_logvar.

    Both SOC and SOH therefore define Gaussian predictive distributions.
    """

    def __init__(
        self,
        num_classes: int,
        width: int = 32,
        blocks: int = 4,
        drop2d: float = 0.0,
        use_pt_as_feature: bool = True,
        soc_hidden: int = 64,
        soh_hidden: int = 64,
        head_dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = MicroResNetEncoder2D3Ch(
            width=width,
            blocks=blocks,
            drop2d=drop2d,
        )

        self.use_pt = bool(use_pt_as_feature)

        pt_dim = 1 if self.use_pt else 0
        p_dim = int(num_classes)

        self.head_mat = nn.Sequential(
            nn.Linear(width, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(64, num_classes),
        )

        soc_in = width + p_dim + pt_dim

        self.head_soc_mu = nn.Sequential(
            nn.Linear(soc_in, soc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(soc_hidden, 1),
        )

        self.head_soc_logvar = nn.Sequential(
            nn.Linear(soc_in, soc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(soc_hidden, 1),
        )

        # SOH is conditioned on the probabilistic SOC summary:
        # predicted SOC mean and predicted SOC standard deviation.
        soh_in = width + p_dim + 2 + pt_dim

        self.head_soh_mu = nn.Sequential(
            nn.Linear(soh_in, soh_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(soh_hidden, 1),
        )

        self.head_soh_logvar = nn.Sequential(
            nn.Linear(soh_in, soh_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(soh_hidden, 1),
        )

    def forward(
        self,
        x_img: torch.Tensor,
        x_pt: torch.Tensor,
    ):
        z = self.encoder(x_img)

        logits_mat = self.head_mat(z)
        p = torch.softmax(logits_mat, dim=1)

        if self.use_pt:
            feat_soc = torch.cat([z, p, x_pt], dim=1)
        else:
            feat_soc = torch.cat([z, p], dim=1)

        soc_mu = self.head_soc_mu(feat_soc).squeeze(1)
        soc_logvar = self.head_soc_logvar(feat_soc).squeeze(1)
        soc_logvar = torch.clamp(soc_logvar, min=-10.0, max=5.0)
        soc_sigma = torch.exp(0.5 * soc_logvar)

        mu_use = soc_mu.unsqueeze(1)
        sigma_use = soc_sigma.unsqueeze(1)

        if self.use_pt:
            feat_soh = torch.cat(
                [z, p, mu_use, sigma_use, x_pt],
                dim=1,
            )
        else:
            feat_soh = torch.cat(
                [z, p, mu_use, sigma_use],
                dim=1,
            )

        soh_mu = self.head_soh_mu(feat_soh).squeeze(1)
        soh_logvar = self.head_soh_logvar(feat_soh).squeeze(1)
        soh_logvar = torch.clamp(soh_logvar, min=-10.0, max=5.0)
        soh_sigma = torch.exp(0.5 * soh_logvar)

        return (
            logits_mat,
            soc_mu,
            soc_logvar,
            soc_sigma,
            soh_mu,
            soh_logvar,
            soh_sigma,
        )


# =============================================================================
# Train and evaluation loops
# =============================================================================
def train_one_epoch(
    model: GaussianCalibrationBaseline,
    loader: DataLoader,
    optimizer,
    device: str,
    w_cls: float,
    w_soc: float,
    w_soh: float,
    grad_clip: float,
    criterion_cls,
    criterion_reg,
    soc_nll_weight: float = 1.0,
    soh_nll_weight: float = 1.0,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    n_samples = 0

    y_cls_true = []
    y_cls_pred = []

    soc_true_all = []
    soc_mu_all = []

    soh_true_all = []
    soh_mu_all = []

    for x3, pt, y_cls, soc, soh in loader:
        x3 = x3.to(device)
        pt = pt.to(device)
        y_cls = y_cls.to(device)
        soc = soc.to(device)
        soh = soh.to(device)

        optimizer.zero_grad(set_to_none=True)

        (
            logits,
            soc_mu,
            soc_logvar,
            _,
            soh_mu,
            soh_logvar,
            _,
        ) = model(x3, pt)

        loss_cls = criterion_cls(logits, y_cls)

        loss_soc = heteroscedastic_nll(
            soc_mu,
            soc_logvar,
            soc,
        ) * float(soc_nll_weight)

        loss_soh = heteroscedastic_nll(
            soh_mu,
            soh_logvar,
            soh,
        ) * float(soh_nll_weight)

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
        n_samples += batch_size

        y_cls_true.append(y_cls.detach().cpu().numpy())
        y_cls_pred.append(
            logits.detach().cpu().argmax(dim=1).numpy()
        )

        soc_true_all.append(soc.detach().cpu().numpy())
        soc_mu_all.append(soc_mu.detach().cpu().numpy())

        soh_true_all.append(soh.detach().cpu().numpy())
        soh_mu_all.append(soh_mu.detach().cpu().numpy())

    if n_samples == 0:
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

    y_cls_true = np.concatenate(y_cls_true)
    y_cls_pred = np.concatenate(y_cls_pred)

    soc_true = np.concatenate(soc_true_all)
    soc_mu = np.concatenate(soc_mu_all)

    soh_true = np.concatenate(soh_true_all)
    soh_mu = np.concatenate(soh_mu_all)

    return {
        "loss": total_loss / n_samples,
        "cls_acc": float(
            accuracy_score(y_cls_true, y_cls_pred)
        ),
        "soc_rmse": rmse(soc_true, soc_mu),
        "soc_mae": mae(soc_true, soc_mu),
        "soc_mape": mape(soc_true, soc_mu),
        "soc_medape": medape(soc_true, soc_mu),
        "soh_rmse": rmse(soh_true, soh_mu),
        "soh_mae": mae(soh_true, soh_mu),
        "soh_mape": mape(soh_true, soh_mu),
        "soh_medape": medape(soh_true, soh_mu),
    }


@torch.no_grad()
def eval_one_epoch(
    model: GaussianCalibrationBaseline,
    loader: DataLoader,
    device: str,
    w_cls: float,
    w_soc: float,
    w_soh: float,
    criterion_cls,
    criterion_reg,
    soc_nll_weight: float = 1.0,
    soh_nll_weight: float = 1.0,
    soc_norm: Optional[Tuple[float, float]] = None,
    soh_norm: Optional[Tuple[float, float]] = None,
    normalize_soc: bool = True,
    zscore_normalize: bool = True,
    return_predictions: bool = False,
) -> Dict:
    model.eval()

    total_loss = 0.0
    n_samples = 0

    y_cls_true = []
    y_cls_pred = []

    soc_true_all = []
    soc_mu_all = []
    soc_logvar_all = []
    soc_sigma_all = []

    soh_true_all = []
    soh_mu_all = []
    soh_logvar_all = []
    soh_sigma_all = []

    for x3, pt, y_cls, soc, soh in loader:
        x3 = x3.to(device)
        pt = pt.to(device)
        y_cls = y_cls.to(device)
        soc = soc.to(device)
        soh = soh.to(device)

        (
            logits,
            soc_mu,
            soc_logvar,
            soc_sigma,
            soh_mu,
            soh_logvar,
            soh_sigma,
        ) = model(x3, pt)

        loss_cls = criterion_cls(logits, y_cls)

        loss_soc = heteroscedastic_nll(
            soc_mu,
            soc_logvar,
            soc,
        ) * float(soc_nll_weight)

        loss_soh = heteroscedastic_nll(
            soh_mu,
            soh_logvar,
            soh,
        ) * float(soh_nll_weight)

        loss = (
            float(w_cls) * loss_cls
            + float(w_soc) * loss_soc
            + float(w_soh) * loss_soh
        )

        batch_size = int(y_cls.size(0))
        total_loss += float(loss.item()) * batch_size
        n_samples += batch_size

        y_cls_true.append(y_cls.detach().cpu().numpy())
        y_cls_pred.append(
            logits.detach().cpu().argmax(dim=1).numpy()
        )

        soc_true_all.append(soc.detach().cpu().numpy())
        soc_mu_all.append(soc_mu.detach().cpu().numpy())
        soc_logvar_all.append(
            soc_logvar.detach().cpu().numpy()
        )
        soc_sigma_all.append(
            soc_sigma.detach().cpu().numpy()
        )

        soh_true_all.append(soh.detach().cpu().numpy())
        soh_mu_all.append(soh_mu.detach().cpu().numpy())
        soh_logvar_all.append(
            soh_logvar.detach().cpu().numpy()
        )
        soh_sigma_all.append(
            soh_sigma.detach().cpu().numpy()
        )

    if n_samples == 0:
        raise RuntimeError("Evaluation loader is empty.")

    y_cls_true = np.concatenate(y_cls_true)
    y_cls_pred = np.concatenate(y_cls_pred)

    soc_true = np.concatenate(soc_true_all)
    soc_mu = np.concatenate(soc_mu_all)
    soc_logvar = np.concatenate(soc_logvar_all)
    soc_sigma = np.concatenate(soc_sigma_all)

    soh_true = np.concatenate(soh_true_all)
    soh_mu = np.concatenate(soh_mu_all)
    soh_logvar = np.concatenate(soh_logvar_all)
    soh_sigma = np.concatenate(soh_sigma_all)

    soc_true_raw, soh_true_raw = inverse_targets(
        soc_true,
        soh_true,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
    )

    soc_pred_raw, soh_pred_raw = inverse_targets(
        soc_mu,
        soh_mu,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
    )

    out = {
        "loss": total_loss / n_samples,
        "cls_acc": float(
            accuracy_score(y_cls_true, y_cls_pred)
        ),
        "soc_rmse": rmse(soc_true, soc_mu),
        "soc_mae": mae(soc_true, soc_mu),
        "soc_mape": mape(soc_true, soc_mu),
        "soc_medape": medape(soc_true, soc_mu),
        "soh_rmse": rmse(soh_true, soh_mu),
        "soh_mae": mae(soh_true, soh_mu),
        "soh_mape": mape(soh_true, soh_mu),
        "soh_medape": medape(soh_true, soh_mu),
        "soc_rmse_raw": rmse(soc_true_raw, soc_pred_raw),
        "soc_mae_raw": mae(soc_true_raw, soc_pred_raw),
        "soc_mape_raw": mape(soc_true_raw, soc_pred_raw),
        "soc_medape_raw": medape(
            soc_true_raw,
            soc_pred_raw,
        ),
        "soh_rmse_raw": rmse(soh_true_raw, soh_pred_raw),
        "soh_mae_raw": mae(soh_true_raw, soh_pred_raw),
        "soh_mape_raw": mape(soh_true_raw, soh_pred_raw),
        "soh_medape_raw": medape(
            soh_true_raw,
            soh_pred_raw,
        ),
        "n_test": int(n_samples),
    }

    if return_predictions:
        pred_df = pd.DataFrame(
            {
                "y_cls_true": y_cls_true,
                "y_cls_pred": y_cls_pred,
                "soc_true_z": soc_true,
                "soc_pred_z": soc_mu,
                "soc_logvar_z": soc_logvar,
                "soc_sigma_z": soc_sigma,
                "soc_true_raw": soc_true_raw,
                "soc_pred_raw": soc_pred_raw,
                "soh_true_z": soh_true,
                "soh_pred_z": soh_mu,
                "soh_logvar_z": soh_logvar,
                "soh_sigma_z": soh_sigma,
                "soh_true_raw": soh_true_raw,
                "soh_pred_raw": soh_pred_raw,
            }
        )

        out["predictions"] = pred_df

    return out


def search_best_temperature(
    standardized_residuals: np.ndarray,
    target_name: str,
) -> Tuple[np.ndarray, float, Dict[str, float], pd.DataFrame]:
    """
    Search T in [0.3, 3.0] using the same logic as plot_calibration.py.

    T minimizes the mean absolute calibration deviation over the
    central 10-90% confidence range.
    """
    residuals = np.asarray(
        standardized_residuals,
        dtype=np.float64,
    )
    residuals = residuals[np.isfinite(residuals)]

    if residuals.size == 0:
        raise RuntimeError(
            f"No finite standardized residuals for {target_name}."
        )

    expected_p = np.linspace(0.0, 1.0, 100)
    core_mask = (
        (expected_p >= 0.10)
        & (expected_p <= 0.90)
    )

    normal = torch.distributions.Normal(0.0, 1.0)

    best_t = 1.0
    best_core_mean = float("inf")
    best_u = None
    best_observed = None

    for t in np.linspace(0.3, 3.0, 300):
        u = normal.cdf(
            torch.tensor(
                residuals / float(t),
                dtype=torch.float64,
            )
        ).numpy()

        u = u[np.isfinite(u)]
        observed_p = np.array(
            [np.mean(u <= p) for p in expected_p],
            dtype=np.float64,
        )

        deviation_abs = np.abs(observed_p - expected_p)
        core_mean = float(
            np.mean(deviation_abs[core_mask])
        )

        if core_mean < best_core_mean:
            best_core_mean = core_mean
            best_t = float(t)
            best_u = u.copy()
            best_observed = observed_p.copy()

    if best_u is None or best_observed is None:
        raise RuntimeError(
            f"Temperature search failed for {target_name}."
        )

    deviation = best_observed - expected_p
    deviation_abs = np.abs(deviation)

    metrics = {
        "best_temperature": best_t,
        "mean_calibration_deviation_10_90_pct": (
            float(np.mean(deviation_abs[core_mask])) * 100.0
        ),
        "max_calibration_deviation_10_90_pct": (
            float(np.max(deviation_abs[core_mask])) * 100.0
        ),
        "mean_calibration_deviation_all_pct": (
            float(np.mean(deviation_abs)) * 100.0
        ),
        "max_calibration_deviation_all_pct": (
            float(np.max(deviation_abs)) * 100.0
        ),
    }

    curve_df = pd.DataFrame(
        {
            "expected_confidence": expected_p,
            "observed_confidence": best_observed,
            "calibration_error_obs_minus_exp": deviation,
            "absolute_calibration_deviation": deviation_abs,
            "is_core_10_90": core_mask.astype(int),
        }
    )

    print(
        f"[CALIBRATION] {target_name} | "
        f"Best T={best_t:.3f} | "
        f"Mean 10-90={metrics['mean_calibration_deviation_10_90_pct']:.3f}% | "
        f"Max 10-90={metrics['max_calibration_deviation_10_90_pct']:.3f}% | "
        f"Mean all={metrics['mean_calibration_deviation_all_pct']:.3f}% | "
        f"Max all={metrics['max_calibration_deviation_all_pct']:.3f}%"
    )

    return best_u, best_t, metrics, curve_df


def plot_calibration_curve(
    u_values: np.ndarray,
    target_name: str,
    best_t: float,
    save_path: str | Path,
) -> None:
    """
    Reproduce the calibrated-regression curve style used by plot_calibration.py.
    """
    import matplotlib.pyplot as plt

    u_values = np.asarray(u_values, dtype=np.float64)
    u_values = u_values[np.isfinite(u_values)]
    u_values = np.sort(u_values)

    expected_p = np.linspace(0.0, 1.0, 100)
    observed_p = np.array(
        [np.mean(u_values <= p) for p in expected_p]
    )

    core_mask = (
        (expected_p >= 0.10)
        & (expected_p <= 0.90)
    )
    core_ece = float(
        np.mean(
            np.abs(
                observed_p[core_mask]
                - expected_p[core_mask]
            )
        )
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Perfect Calibration",
        zorder=3,
    )

    ax.fill_between(
        [0, 1],
        [0.05, 1.05],
        [-0.05, 0.95],
        color="gray",
        alpha=0.15,
        label="±5% Tolerance",
        zorder=1,
    )
    ax.fill_between(
        [0, 1],
        [0.10, 1.10],
        [-0.10, 0.90],
        color="gray",
        alpha=0.05,
        label="±10% Tolerance",
        zorder=1,
    )

    ax.plot(
        expected_p,
        observed_p,
        label=(
            f"Calibrated Gaussian "
            f"(Core ECE: {core_ece:.4f})"
        ),
        color="#D55E00",
        lw=2.5,
        zorder=4,
    )

    ax.axvline(
        x=0.1,
        color="orange",
        linestyle=":",
        alpha=0.8,
        zorder=2,
    )
    ax.axvline(
        x=0.9,
        color="orange",
        linestyle=":",
        alpha=0.8,
        zorder=2,
    )
    ax.text(
        0.5,
        0.02,
        "Core Region (10%-90%)",
        color="orange",
        ha="center",
        va="bottom",
        fontsize=10,
        alpha=0.9,
        fontweight="bold",
    )

    ax.set_title(
        f"{target_name} (T={best_t:.2f}) Post-calibrated Regression\n"
        "(Outlier-robust)",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel(
        "Expected Confidence Level (Predicted CDF)",
        fontsize=12,
    )
    ax.set_ylabel(
        "Observed Fraction (Empirical CDF)",
        fontsize=12,
    )
    ax.legend(
        loc="upper left",
        fontsize=11,
        framealpha=0.9,
    )
    ax.grid(
        True,
        linestyle="--",
        alpha=0.4,
        zorder=0,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_gaussian_calibration_analysis(
    pred_df: pd.DataFrame,
    metrics_dir: str | Path,
) -> pd.DataFrame:
    """
    Compute post-hoc temperature-scaled calibration for Gaussian SOC and SOH.

    Outputs the two calibrated curves and the exact statistics needed for
    Supplementary Table 14.
    """
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    required = {
        "soc_true_z",
        "soc_pred_z",
        "soc_sigma_z",
        "soh_true_z",
        "soh_pred_z",
        "soh_sigma_z",
    }
    missing = required.difference(pred_df.columns)
    if missing:
        raise RuntimeError(
            "Missing Gaussian prediction columns required for calibration: "
            + ", ".join(sorted(missing))
        )

    soc_sigma = np.clip(
        pred_df["soc_sigma_z"].to_numpy(dtype=np.float64),
        1e-6,
        None,
    )
    soh_sigma = np.clip(
        pred_df["soh_sigma_z"].to_numpy(dtype=np.float64),
        1e-6,
        None,
    )

    soc_z = (
        pred_df["soc_true_z"].to_numpy(dtype=np.float64)
        - pred_df["soc_pred_z"].to_numpy(dtype=np.float64)
    ) / soc_sigma

    soh_z = (
        pred_df["soh_true_z"].to_numpy(dtype=np.float64)
        - pred_df["soh_pred_z"].to_numpy(dtype=np.float64)
    ) / soh_sigma

    rows = []

    for target_name, z_values in [
        ("SOC", soc_z),
        ("SOH", soh_z),
    ]:
        best_u, best_t, metrics, curve_df = search_best_temperature(
            standardized_residuals=z_values,
            target_name=f"Gaussian baseline {target_name}",
        )

        curve_df.to_csv(
            metrics_dir / f"calibration_curve_{target_name.lower()}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        plot_calibration_curve(
            u_values=best_u,
            target_name=target_name,
            best_t=best_t,
            save_path=(
                metrics_dir
                / f"calibration_{target_name.lower()}_calibrated.png"
            ),
        )

        rows.append(
            {
                "model": "Gaussian baseline",
                "target": target_name,
                "best_calibration_temperature_T": (
                    metrics["best_temperature"]
                ),
                "mean_calibration_deviation_10_90_pct": (
                    metrics[
                        "mean_calibration_deviation_10_90_pct"
                    ]
                ),
                "max_calibration_deviation_10_90_pct": (
                    metrics[
                        "max_calibration_deviation_10_90_pct"
                    ]
                ),
                "mean_calibration_deviation_all_pct": (
                    metrics[
                        "mean_calibration_deviation_all_pct"
                    ]
                ),
                "max_calibration_deviation_all_pct": (
                    metrics[
                        "max_calibration_deviation_all_pct"
                    ]
                ),
            }
        )

    calibration_df = pd.DataFrame(rows)
    calibration_df.to_csv(
        metrics_dir / "calibration_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[GAUSSIAN CALIBRATION METRICS]")
    print(calibration_df.to_string(index=False))

    return calibration_df


# =============================================================================
# Main experiment
# =============================================================================
def run_experiment(
    data_root: str | Path = DATA_ROOT,
    pulse_list: Optional[List[int]] = None,
    exp_dir: str | Path = EXP_DIR,
    u_start: int = U_START,
    u_end: int = U_END,
    drop_first_class: bool = DROP_FIRST_CLASS,
    soc_col: str = SOC_COL,
    soh_col: str = SOH_COL,
    use_pt_as_feature: bool = USE_PT_AS_FEATURE,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    grad_clip: float = GRAD_CLIP,
    max_epochs: int = MAX_EPOCHS,
    early_stopping: bool = EARLY_STOPPING,
    patience: int = PATIENCE,
    resume: bool = RESUME,
    num_workers: int = NUM_WORKERS,
    seed: int = SEED,
    width: int = WIDTH,
    blocks: int = BLOCKS,
    drop2d: float = DROP2D,
    head_dropout: float = HEAD_DROPOUT,
    soc_hidden: int = SOC_HIDDEN,
    soh_hidden: int = SOH_HIDDEN,
    w_cls: float = W_CLS,
    w_soc: float = W_SOC,
    w_soh: float = W_SOH,
    test_id_frac: float = TEST_ID_FRAC,
    test_id_count: int = TEST_ID_COUNT,
    normalize_soc: bool = NORMALIZE_SOC,
    zscore_normalize: bool = ZSCORE_NORMALIZE,
    normalize_u_with_train_stats: bool = NORMALIZE_U_WITH_TRAIN_STATS,
) -> Dict[str, float]:
    start_time = time.time()

    if pulse_list is None:
        pulse_list = DEFAULT_PULSE_LIST

    data_root = Path(data_root)
    exp_dir = Path(exp_dir)

    cache_dir = exp_dir / "cache"
    ckpt_dir = exp_dir / "checkpoints"
    logs_dir = exp_dir / "logs"
    splits_dir = exp_dir / "splits"
    metrics_dir = exp_dir / "metrics"

    for path in [
        exp_dir,
        cache_dir,
        ckpt_dir,
        logs_dir,
        splits_dir,
        metrics_dir,
    ]:
        ensure_dir(str(path))

    set_random_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_config = {
        "data_root": str(data_root),
        "pulse_list": list(map(int, pulse_list)),
        "exp_dir": str(exp_dir),
        "u_start": u_start,
        "u_end": u_end,
        "drop_first_class": drop_first_class,
        "soc_col": soc_col,
        "soh_col": soh_col,
        "use_pt_as_feature": use_pt_as_feature,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "max_epochs": max_epochs,
        "early_stopping": early_stopping,
        "patience": patience,
        "resume": resume,
        "num_workers": num_workers,
        "seed": seed,
        "width": width,
        "blocks": blocks,
        "drop2d": drop2d,
        "head_dropout": head_dropout,
        "soc_hidden": soc_hidden,
        "soh_hidden": soh_hidden,
        "w_cls": w_cls,
        "w_soc": w_soc,
        "w_soh": w_soh,
        "test_id_frac": test_id_frac,
        "test_id_count": test_id_count,
        "normalize_soc": normalize_soc,
        "zscore_normalize": zscore_normalize,
        "normalize_u_with_train_stats": normalize_u_with_train_stats,
    }

    save_json(exp_dir / "run_config.json", run_config)

    soc_list = list(range(5, 90, 5))

    train_kwargs = {
        "data_root": str(data_root),
        "soc_list": soc_list,
        "pulse_list": list(map(int, pulse_list)),
        "u_start": u_start,
        "u_end": u_end,
        "drop_first_class": drop_first_class,
    }

    Xtr_raw, ytr_raw, mtr_raw, tag_tr, hit_tr = load_or_build_cache(
        str(cache_dir),
        "raw_train",
        build_train_mix_soc_mix_pt,
        train_kwargs,
    )

    test_kwargs = {
        "data_root": str(data_root),
        "pulse_list": list(map(int, pulse_list)),
        "u_start": u_start,
        "u_end": u_end,
        "drop_first_class": drop_first_class,
    }

    Xte_raw, yte_raw, mte_raw, tag_te, hit_te = load_or_build_cache(
        str(cache_dir),
        "raw_test",
        build_test_random_mix_pt,
        test_kwargs,
    )

    print(f"[CACHE] Train tag: {tag_tr} | hit={hit_tr}")
    print(f"[CACHE] Test  tag: {tag_te} | hit={hit_te}")

    Xtr_raw, ytr_raw, mtr_raw = drop_nan_inf_rows(
        Xtr_raw,
        ytr_raw,
        mtr_raw,
        "RAW_TRAIN",
    )

    Xte_raw, yte_raw, mte_raw = drop_nan_inf_rows(
        Xte_raw,
        yte_raw,
        mte_raw,
        "RAW_TEST",
    )

    if Xtr_raw.shape[1] != 41 or Xte_raw.shape[1] != 41:
        raise ValueError(
            f"Expected X dimension = 41 for U1-U41. "
            f"Got train={Xtr_raw.shape}, test={Xte_raw.shape}."
        )

    for col in [soc_col, soh_col]:
        if col not in mtr_raw.columns or col not in mte_raw.columns:
            raise RuntimeError(f"Metadata must contain column '{col}'.")

    all_ids = pd.concat(
        [mtr_raw["ID"], mte_raw["ID"]],
        axis=0,
    ).astype(str).to_numpy()

    test_ids = pick_test_ids(
        all_ids=all_ids,
        test_id_frac=test_id_frac,
        test_id_count=test_id_count,
        seed=seed,
    )

    if test_id_count and test_id_count > 0:
        split_name = f"testIDs_seed{seed}_n{test_id_count}"
    else:
        split_name = f"testIDs_seed{seed}_frac{test_id_frac}"

    with open(splits_dir / f"{split_name}.txt", "w", encoding="utf-8") as f:
        for test_id in test_ids:
            f.write(str(test_id) + "\n")

    Xtr, ytr_str, mtr, Xte, yte_str, mte = apply_id_split(
        Xtr=Xtr_raw,
        ytr_str=ytr_raw,
        mtr=mtr_raw,
        Xte=Xte_raw,
        yte_str=yte_raw,
        mte=mte_raw,
        test_ids=test_ids,
    )

    if len(ytr_str) == 0 or len(yte_str) == 0:
        raise RuntimeError("Empty train or test data after applying ID split.")

    print(
        f"[DATA] Final TRAIN samples = {len(ytr_str)} | "
        f"unique IDs = {mtr['ID'].astype(str).nunique()}"
    )
    print(
        f"[DATA] Final TEST  samples = {len(yte_str)} | "
        f"unique IDs = {mte['ID'].astype(str).nunique()}"
    )

    if normalize_u_with_train_stats:
        u_mean = Xtr.mean(axis=0, keepdims=True)
        u_std = Xtr.std(axis=0, keepdims=True) + 1e-8

        Xtr = (Xtr - u_mean) / u_std
        Xte = (Xte - u_mean) / u_std

        np.savez_compressed(
            exp_dir / "u41_norm_train_only.npz",
            u_mean=u_mean.astype(np.float32),
            u_std=u_std.astype(np.float32),
        )

        print("[NORM] Applied U1-U41 z-score using TRAIN statistics only.")
    else:
        print("[NORM] U1-U41 train-only normalization is disabled for this baseline.")

    soc_train = mtr[soc_col].astype(float).to_numpy(dtype=np.float64)

    if normalize_soc:
        soc_train = soc_train / 100.0

    soc_norm = (
        float(soc_train.mean()),
        float(soc_train.std() + 1e-8),
    )

    soh_train = mtr[soh_col].astype(float).to_numpy(dtype=np.float64)

    soh_norm = (
        float(soh_train.mean()),
        float(soh_train.std() + 1e-8),
    )

    np.savez_compressed(
        exp_dir / "target_norm_train_only.npz",
        soc_mean=np.array([soc_norm[0]], dtype=np.float32),
        soc_std=np.array([soc_norm[1]], dtype=np.float32),
        soh_mean=np.array([soh_norm[0]], dtype=np.float32),
        soh_std=np.array([soh_norm[1]], dtype=np.float32),
    )

    label_encoder = LabelEncoder()
    ytr_cls = label_encoder.fit_transform(ytr_str)

    train_classes = set(label_encoder.classes_.tolist())

    mask_known = np.array(
        [label in train_classes for label in yte_str],
        dtype=bool,
    )

    if not mask_known.all():
        n_removed = int((~mask_known).sum())
        print(f"[WARN] Removing {n_removed} test samples with unseen labels.")

        Xte = Xte[mask_known]
        yte_str = yte_str[mask_known]
        mte = mte.loc[mask_known].reset_index(drop=True)

    yte_cls = label_encoder.transform(yte_str)

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    save_json(
        exp_dir / "label_mapping.json",
        {
            "classes": class_names,
            "split_name": split_name,
        },
    )

    if use_pt_as_feature and "pulse_ms" in mtr.columns:
        pt_train_ms = mtr["pulse_ms"].astype(float).to_numpy(dtype=np.float32)
        pt_log = np.log1p(pt_train_ms)

        pt_norm = (
            float(pt_log.mean()),
            float(pt_log.std() + 1e-8),
        )
    elif use_pt_as_feature and "pulse_width_ms" in mtr.columns:
        pt_train_ms = mtr["pulse_width_ms"].astype(float).to_numpy(dtype=np.float32)
        pt_log = np.log1p(pt_train_ms)

        pt_norm = (
            float(pt_log.mean()),
            float(pt_log.std() + 1e-8),
        )
    else:
        pt_norm = (0.0, 1.0)

    pt_col = "pulse_ms" if "pulse_ms" in mte.columns else "pulse_width_ms"

    ds_tr = CalibrationBaselineDataset(
        X_u=Xtr,
        y_cls=ytr_cls,
        meta=mtr,
        soc_col=soc_col,
        soh_col=soh_col,
        pt_col=pt_col,
        use_pt_as_feature=use_pt_as_feature,
        pt_norm=pt_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
        soc_norm=soc_norm if zscore_normalize else None,
        soh_norm=soh_norm if zscore_normalize else None,
    )

    ds_te = CalibrationBaselineDataset(
        X_u=Xte,
        y_cls=yte_cls,
        meta=mte,
        soc_col=soc_col,
        soh_col=soh_col,
        pt_col=pt_col,
        use_pt_as_feature=use_pt_as_feature,
        pt_norm=pt_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
        soc_norm=soc_norm if zscore_normalize else None,
        soh_norm=soh_norm if zscore_normalize else None,
    )

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    dl_te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    model = GaussianCalibrationBaseline(
        num_classes=num_classes,
        width=width,
        blocks=blocks,
        drop2d=drop2d,
        use_pt_as_feature=use_pt_as_feature,
        soc_hidden=soc_hidden,
        soh_hidden=soh_hidden,
        head_dropout=head_dropout,
    ).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss(beta=1.0)

    soc_nll_weight = 1.0
    soh_nll_weight = 1.0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    last_path = ckpt_dir / "last.pt"
    best_path = ckpt_dir / "best.pt"

    start_epoch = 0
    best_score = -1e9
    best_epoch = -1
    bad_count = 0

    if resume and last_path.exists():
        ckpt = torch_load_compatible(last_path, map_location=device)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        start_epoch = int(ckpt.get("epoch", 0) + 1)
        best_score = float(ckpt.get("best_score", -1e9))
        best_epoch = int(ckpt.get("best_epoch", -1))
        bad_count = int(ckpt.get("bad_count", 0))

        print(
            f"[RESUME] start_epoch={start_epoch}, "
            f"best_score={best_score:.6f}, best_epoch={best_epoch}"
        )

    log_path = logs_dir / "train_log.csv"

    for epoch in range(start_epoch, int(max_epochs)):
        epoch_start = time.time()

        tr = train_one_epoch(
            model=model,
            loader=dl_tr,
            optimizer=optimizer,
            device=device,
            w_cls=w_cls,
            w_soc=w_soc,
            w_soh=w_soh,
            grad_clip=grad_clip,
            criterion_cls=criterion_cls,
            criterion_reg=criterion_reg,
            soc_nll_weight=soc_nll_weight,
            soh_nll_weight=soh_nll_weight,
        )

        te = eval_one_epoch(
            model=model,
            loader=dl_te,
            device=device,
            w_cls=w_cls,
            w_soc=w_soc,
            w_soh=w_soh,
            criterion_cls=criterion_cls,
            criterion_reg=criterion_reg,
            soc_nll_weight=soc_nll_weight,
            soh_nll_weight=soh_nll_weight,
            soc_norm=soc_norm if zscore_normalize else None,
            soh_norm=soh_norm if zscore_normalize else None,
            normalize_soc=normalize_soc,
            zscore_normalize=zscore_normalize,
            return_predictions=False,
        )

        elapsed = time.time() - epoch_start

        # Select the best checkpoint using TRAIN metrics only.
        # The held-out TEST set is not used for model selection.
        score = (
            tr["cls_acc"]
            - 0.1 * (tr["soc_rmse"] + tr["soh_rmse"])
        )

        row = pd.DataFrame(
            [
                {
                    "epoch": int(epoch),
                    "train_loss": tr["loss"],
                    "train_cls_acc": tr["cls_acc"],
                    "train_soc_rmse": tr["soc_rmse"],
                    "train_soc_mae": tr["soc_mae"],
                    "train_soc_mape": tr["soc_mape"],
                    "train_soc_medape": tr["soc_medape"],
                    "train_soh_rmse": tr["soh_rmse"],
                    "train_soh_mae": tr["soh_mae"],
                    "train_soh_mape": tr["soh_mape"],
                    "train_soh_medape": tr["soh_medape"],
                    "test_loss": te["loss"],
                    "test_cls_acc": te["cls_acc"],
                    "test_soc_rmse": te["soc_rmse"],
                    "test_soc_mae": te["soc_mae"],
                    "test_soc_mape": te["soc_mape"],
                    "test_soc_medape": te["soc_medape"],
                    "test_soh_rmse": te["soh_rmse"],
                    "test_soh_mae": te["soh_mae"],
                    "test_soh_mape": te["soh_mape"],
                    "test_soh_medape": te["soh_medape"],
                    "test_soc_rmse_raw": te["soc_rmse_raw"],
                    "test_soc_mae_raw": te["soc_mae_raw"],
                    "test_soc_mape_raw": te["soc_mape_raw"],
                    "test_soc_medape_raw": te["soc_medape_raw"],
                    "test_soh_rmse_raw": te["soh_rmse_raw"],
                    "test_soh_mae_raw": te["soh_mae_raw"],
                    "test_soh_mape_raw": te["soh_mape_raw"],
                    "test_soh_medape_raw": te["soh_medape_raw"],
                    "train_selection_score": score,
                    "best_score_so_far": max(best_score, score),
                    "epoch_duration_sec": elapsed,
                }
            ]
        )

        if not log_path.exists():
            row.to_csv(log_path, index=False, encoding="utf-8-sig")
        else:
            row.to_csv(
                log_path,
                mode="a",
                header=False,
                index=False,
                encoding="utf-8-sig",
            )

        improved = score > best_score

        if improved:
            best_score = float(score)
            best_epoch = int(epoch)
            bad_count = 0

            torch.save(
                {
                    "epoch": int(epoch),
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "run_config": run_config,
                },
                best_path,
            )
        else:
            bad_count += 1

        torch.save(
            {
                "epoch": int(epoch),
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "bad_count": bad_count,
                "run_config": run_config,
            },
            last_path,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"TE cls={te['cls_acc']:.4f} | "
            f"SOC MedAPE(raw)={te['soc_medape_raw']:.3f}% | "
            f"SOH MedAPE(raw)={te['soh_medape_raw']:.3f}% | "
            f"train_select_score={score:.6f} | "
            f"time={elapsed:.2f}s"
        )

        if early_stopping and bad_count >= patience:
            print(
                f"[EARLY STOP] best_score={best_score:.6f} "
                f"at epoch={best_epoch}"
            )
            break

    if best_path.exists():
        ckpt = torch_load_compatible(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(
            f"[BEST] Loaded best checkpoint from epoch={ckpt.get('epoch')} | "
            f"score={ckpt.get('best_score')}"
        )

    te = eval_one_epoch(
        model=model,
        loader=dl_te,
        device=device,
        w_cls=w_cls,
        w_soc=w_soc,
        w_soh=w_soh,
        criterion_cls=criterion_cls,
        criterion_reg=criterion_reg,
        soc_nll_weight=soc_nll_weight,
        soh_nll_weight=soh_nll_weight,
        soc_norm=soc_norm if zscore_normalize else None,
        soh_norm=soh_norm if zscore_normalize else None,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
        return_predictions=True,
    )

    pred_df = te.pop("predictions")

    pred_df.to_csv(
        metrics_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    calibration_df = run_gaussian_calibration_analysis(
        pred_df=pred_df,
        metrics_dir=metrics_dir,
    )

    final_metrics = {
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "test_cls_acc": float(te["cls_acc"]),
        "test_soc_rmse": float(te["soc_rmse"]),
        "test_soc_mae": float(te["soc_mae"]),
        "test_soc_mape": float(te["soc_mape"]),
        "test_soc_medape": float(te["soc_medape"]),
        "test_soh_rmse": float(te["soh_rmse"]),
        "test_soh_mae": float(te["soh_mae"]),
        "test_soh_mape": float(te["soh_mape"]),
        "test_soh_medape": float(te["soh_medape"]),
        "test_soc_rmse_raw": float(te["soc_rmse_raw"]),
        "test_soc_mae_raw": float(te["soc_mae_raw"]),
        "test_soc_mape_raw": float(te["soc_mape_raw"]),
        "test_soc_medape_raw": float(te["soc_medape_raw"]),
        "test_soh_rmse_raw": float(te["soh_rmse_raw"]),
        "test_soh_mae_raw": float(te["soh_mae_raw"]),
        "test_soh_mape_raw": float(te["soh_mape_raw"]),
        "test_soh_medape_raw": float(te["soh_medape_raw"]),
        "n_train": int(len(ds_tr)),
        "n_test": int(len(ds_te)),
        "num_classes": int(num_classes),
        "device": device,
        "elapsed_sec": float(time.time() - start_time),
        "gaussian_soc_best_T": float(
            calibration_df.loc[
                calibration_df["target"] == "SOC",
                "best_calibration_temperature_T",
            ].iloc[0]
        ),
        "gaussian_soc_mean_calibration_deviation_10_90_pct": float(
            calibration_df.loc[
                calibration_df["target"] == "SOC",
                "mean_calibration_deviation_10_90_pct",
            ].iloc[0]
        ),
        "gaussian_soh_best_T": float(
            calibration_df.loc[
                calibration_df["target"] == "SOH",
                "best_calibration_temperature_T",
            ].iloc[0]
        ),
        "gaussian_soh_mean_calibration_deviation_10_90_pct": float(
            calibration_df.loc[
                calibration_df["target"] == "SOH",
                "mean_calibration_deviation_10_90_pct",
            ].iloc[0]
        ),
    }

    pd.DataFrame([final_metrics]).to_csv(
        metrics_dir / "final_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        metrics_dir / "final_metrics.json",
        final_metrics,
    )

    print("\n[FINAL METRICS]")
    for key, value in final_metrics.items():
        print(f"{key}: {value}")

    print(f"\n[OK] Saved outputs under: {exp_dir}")

    return final_metrics


def main() -> None:
    run_experiment(
        data_root=DATA_ROOT,
        pulse_list=DEFAULT_PULSE_LIST,
        exp_dir=EXP_DIR,
        u_start=U_START,
        u_end=U_END,
        drop_first_class=DROP_FIRST_CLASS,
        soc_col=SOC_COL,
        soh_col=SOH_COL,
        use_pt_as_feature=USE_PT_AS_FEATURE,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,
        max_epochs=MAX_EPOCHS,
        early_stopping=EARLY_STOPPING,
        patience=PATIENCE,
        resume=RESUME,
        num_workers=NUM_WORKERS,
        seed=SEED,
        width=WIDTH,
        blocks=BLOCKS,
        drop2d=DROP2D,
        head_dropout=HEAD_DROPOUT,
        soc_hidden=SOC_HIDDEN,
        soh_hidden=SOH_HIDDEN,
        w_cls=W_CLS,
        w_soc=W_SOC,
        w_soh=W_SOH,
        test_id_frac=TEST_ID_FRAC,
        test_id_count=TEST_ID_COUNT,
        normalize_soc=NORMALIZE_SOC,
        zscore_normalize=ZSCORE_NORMALIZE,
        normalize_u_with_train_stats=NORMALIZE_U_WITH_TRAIN_STATS,
    )


if __name__ == "__main__":
    main()