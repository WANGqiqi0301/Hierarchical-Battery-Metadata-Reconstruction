# -*- coding: utf-8 -*-
"""
train_calibration_hybrid.py

Train and evaluate a hierarchical hybrid probabilistic model:

    Material head: classification
    SOC head:      conditional normalizing flow
    SOH head:      heteroscedastic Gaussian

The model factorization is:

    p(M, SOC, SOH | x)
      = p(M | x)
        p_flow(SOC | M, x)
        p_gaussian(SOH | SOC, M, x)

The SOC flow, encoder, and material head are reused from the existing
Hier3HeadModel implementation. The SOH flow is replaced with Gaussian mean and
log-variance heads.

Training:
    - SOC flow is optimized using exact negative log-likelihood.
    - SOH Gaussian head is optimized using heteroscedastic Gaussian NLL.
    - During training, true SOC is used as the SOH context by default
      (teacher forcing), matching the existing hierarchical training scheme.
    - During test evaluation, predicted SOC is used as the SOH context.

Outputs:
    results/calibration_hybrid/
        checkpoints/best.pt
        checkpoints/last.pt
        logs/train_log.csv
        metrics/final_metrics.csv
        metrics/final_metrics.json
        metrics/test_predictions.csv
        metrics/probabilistic_metrics.csv
        metrics/per_sample_probabilistic_metrics.csv
        run_config.json
        label_mapping.json
        target_norm_train_only.npz

Run:
    python analysis/train_calibration_hybrid.py
"""

from __future__ import annotations

import json
import os
import sys
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
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Project imports
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
from proposed_framework.data.pulse_dataset import HierPulseDataset
from proposed_framework.models.hierarchical_model import Hier3HeadModel


# =============================================================================
# Configuration
# =============================================================================
DATA_ROOT = PROJECT_ROOT / "data"
EXP_DIR = PROJECT_ROOT / "results" / "calibration_hybrid"

DEFAULT_PULSE_LIST = [
    30, 50, 70, 100, 300,
    500, 700, 1000, 3000, 5000,
]

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

FLOW_LAYERS = 6
FLOW_BINS = 8
FLOW_TAIL_BOUND = 3.0

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
NORMALIZE_U_WITH_TRAIN_STATS = True

# Hierarchical conditioning
TRAIN_SOH_WITH_TRUE_SOC = True
EVAL_SOC_MC_SAMPLES = 256

# Probabilistic evaluation
FLOW_EVAL_SAMPLES = 1000
INTERVAL_LEVELS = [0.50, 0.80, 0.90, 0.95]
SIGMA_MIN = 1e-6


# =============================================================================
# Utilities
# =============================================================================
def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def torch_load_compatible(path: str | Path, map_location: str):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def extract_state_dict(ckpt) -> dict:
    if not isinstance(ckpt, dict):
        raise RuntimeError("Unsupported checkpoint format.")

    if "model" in ckpt:
        return ckpt["model"]
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]

    return ckpt


def heteroscedastic_nll(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    logvar = torch.clamp(
        logvar,
        min=-10.0,
        max=5.0,
    )

    inv_var = torch.exp(-logvar)

    return 0.5 * (
        inv_var * (y - mu) ** 2
        + logvar
        + np.log(2.0 * np.pi)
    ).mean()


def inverse_single_target(
    values_z: np.ndarray,
    target: str,
    soc_norm: Tuple[float, float],
    soh_norm: Tuple[float, float],
    normalize_soc: bool,
    zscore_normalize: bool,
) -> np.ndarray:
    values = np.asarray(
        values_z,
        dtype=np.float64,
    )

    if not zscore_normalize:
        raw = values.copy()
    elif target == "SOC":
        raw = (
            values * float(soc_norm[1])
            + float(soc_norm[0])
        )
    elif target == "SOH":
        raw = (
            values * float(soh_norm[1])
            + float(soh_norm[0])
        )
    else:
        raise ValueError(
            f"Unknown target: {target}"
        )

    if (
        target == "SOC"
        and normalize_soc
    ):
        raw = raw * 100.0

    return raw


def target_scale_factor(
    target: str,
    soc_norm: Tuple[float, float],
    soh_norm: Tuple[float, float],
    normalize_soc: bool,
    zscore_normalize: bool,
) -> float:
    if not zscore_normalize:
        factor = 1.0
    elif target == "SOC":
        factor = float(soc_norm[1])
    elif target == "SOH":
        factor = float(soh_norm[1])
    else:
        raise ValueError(
            f"Unknown target: {target}"
        )

    if (
        target == "SOC"
        and normalize_soc
    ):
        factor *= 100.0

    return factor


def normalize_flow_samples(
    samples: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """
    Convert common flow sample layouts to [B, S].
    """
    if not isinstance(samples, torch.Tensor):
        samples = torch.as_tensor(samples)

    if samples.ndim == 3:
        if samples.shape[-1] == 1:
            samples = samples.squeeze(-1)
        elif samples.shape[1] == 1:
            samples = samples.squeeze(1)
        else:
            raise RuntimeError(
                "Unexpected 3D flow sample shape: "
                f"{tuple(samples.shape)}"
            )

    if samples.ndim != 2:
        raise RuntimeError(
            "Expected 2D flow samples after squeezing, got "
            f"{tuple(samples.shape)}"
        )

    if samples.shape[0] == batch_size:
        return samples

    if samples.shape[1] == batch_size:
        return samples.transpose(0, 1)

    raise RuntimeError(
        "Cannot identify batch dimension in flow samples: "
        f"{tuple(samples.shape)}, batch_size={batch_size}"
    )


def sample_from_flow(
    flow_module,
    context: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """
    Return flow samples in shape [B, S].
    """
    if hasattr(flow_module, "sample"):
        try:
            samples = flow_module.sample(
                context,
                int(num_samples),
            )
        except Exception:
            try:
                samples = flow_module.sample(
                    int(num_samples),
                    context=context,
                )
            except Exception:
                samples = flow_module.sample(
                    num_samples=int(num_samples),
                    context=context,
                )

    elif hasattr(flow_module, "flow"):
        samples = flow_module.flow.sample(
            num_samples=int(num_samples),
            context=context,
        )

    else:
        raise AttributeError(
            "Flow module does not expose sample()."
        )

    return normalize_flow_samples(
        samples=samples,
        batch_size=int(context.shape[0]),
    )


def flow_log_prob(
    flow_module,
    target: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    """
    Return one SOC log-probability per sample.
    """
    target = target.view(-1, 1)

    if hasattr(flow_module, "log_prob"):
        try:
            out = flow_module.log_prob(
                target,
                context,
            )
        except Exception:
            try:
                out = flow_module.log_prob(
                    target,
                    context=context,
                )
            except Exception:
                out = flow_module.log_prob(
                    inputs=target,
                    context=context,
                )

        return out.view(-1)

    if hasattr(flow_module, "flow"):
        inner = flow_module.flow

        try:
            out = inner.log_prob(
                target,
                context=context,
            )
        except TypeError:
            out = inner.log_prob(
                inputs=target,
                context=context,
            )

        return out.view(-1)

    raise AttributeError(
        "Flow module does not expose log_prob()."
    )


def flow_sample_mean(
    flow_module,
    context: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    samples = sample_from_flow(
        flow_module=flow_module,
        context=context,
        num_samples=num_samples,
    )

    return samples.mean(
        dim=1,
        keepdim=True,
    )


# =============================================================================
# CRPS and interval metrics
# =============================================================================
def gaussian_crps(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.clip(
        np.asarray(sigma, dtype=np.float64),
        SIGMA_MIN,
        None,
    )

    z = (y - mu) / sigma
    z_t = torch.tensor(
        z,
        dtype=torch.float64,
    )

    normal = torch.distributions.Normal(
        0.0,
        1.0,
    )

    cdf = normal.cdf(z_t).numpy()
    pdf = torch.exp(
        normal.log_prob(z_t)
    ).numpy()

    value = sigma * (
        z * (2.0 * cdf - 1.0)
        + 2.0 * pdf
        - 1.0 / np.sqrt(np.pi)
    )

    return np.maximum(
        value,
        0.0,
    )


def sample_crps(
    y: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    y = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(-1)

    samples = np.asarray(
        samples,
        dtype=np.float64,
    )

    if samples.ndim != 2:
        raise ValueError(
            f"Expected samples [N, S], got {samples.shape}."
        )

    n_samples = int(samples.shape[1])

    term_obs = np.mean(
        np.abs(
            samples - y[:, None]
        ),
        axis=1,
    )

    sorted_samples = np.sort(
        samples,
        axis=1,
    )

    weights = (
        2.0 * np.arange(
            1,
            n_samples + 1,
        )
        - n_samples
        - 1.0
    )

    pairwise_expectation = (
        2.0
        * np.sum(
            sorted_samples
            * weights[None, :],
            axis=1,
        )
        / float(n_samples ** 2)
    )

    return np.maximum(
        term_obs
        - 0.5 * pairwise_expectation,
        0.0,
    )


def interval_metrics_from_samples(
    y: np.ndarray,
    samples: np.ndarray,
    levels: List[float],
) -> Dict[str, np.ndarray]:
    y = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(-1)

    samples = np.asarray(
        samples,
        dtype=np.float64,
    )

    out = {}

    for level in levels:
        alpha = 1.0 - float(level)

        lower = np.quantile(
            samples,
            alpha / 2.0,
            axis=1,
        )

        upper = np.quantile(
            samples,
            1.0 - alpha / 2.0,
            axis=1,
        )

        covered = (
            (y >= lower)
            & (y <= upper)
        ).astype(np.float64)

        width = upper - lower
        key = int(round(level * 100))

        out[f"lower_{key}"] = lower
        out[f"upper_{key}"] = upper
        out[f"covered_{key}"] = covered
        out[f"width_{key}"] = width

    return out


def interval_metrics_gaussian(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    levels: List[float],
) -> Dict[str, np.ndarray]:
    y = np.asarray(
        y,
        dtype=np.float64,
    ).reshape(-1)

    mu = np.asarray(
        mu,
        dtype=np.float64,
    ).reshape(-1)

    sigma = np.clip(
        np.asarray(
            sigma,
            dtype=np.float64,
        ).reshape(-1),
        SIGMA_MIN,
        None,
    )

    normal = torch.distributions.Normal(
        0.0,
        1.0,
    )

    out = {}

    for level in levels:
        alpha = 1.0 - float(level)

        z_value = float(
            normal.icdf(
                torch.tensor(
                    1.0 - alpha / 2.0,
                    dtype=torch.float64,
                )
            ).item()
        )

        lower = mu - z_value * sigma
        upper = mu + z_value * sigma

        covered = (
            (y >= lower)
            & (y <= upper)
        ).astype(np.float64)

        width = upper - lower
        key = int(round(level * 100))

        out[f"lower_{key}"] = lower
        out[f"upper_{key}"] = upper
        out[f"covered_{key}"] = covered
        out[f"width_{key}"] = width

    return out


# =============================================================================
# Hybrid model
# =============================================================================
class HybridFlowGaussianModel(nn.Module):
    """
    Hierarchical hybrid probabilistic model.

    Reuses from Hier3HeadModel:
        encoder
        material classification head
        SOC conditional flow

    Adds:
        SOH Gaussian mean head
        SOH Gaussian log-variance head
    """

    def __init__(
        self,
        num_classes: int,
        width: int = WIDTH,
        blocks: int = BLOCKS,
        drop2d: float = DROP2D,
        use_pt_as_feature: bool = USE_PT_AS_FEATURE,
        head_dropout: float = HEAD_DROPOUT,
        flow_layers: int = FLOW_LAYERS,
        flow_bins: int = FLOW_BINS,
        flow_tail_bound: float = FLOW_TAIL_BOUND,
        soh_hidden: int = SOH_HIDDEN,
    ):
        super().__init__()

        try:
            base = Hier3HeadModel(
                num_classes=int(num_classes),
                width=int(width),
                blocks=int(blocks),
                drop2d=float(drop2d),
                use_pt_as_feature=bool(
                    use_pt_as_feature
                ),
                head_dropout=float(
                    head_dropout
                ),
                flow_layers=int(
                    flow_layers
                ),
                flow_bins=int(
                    flow_bins
                ),
                flow_tail_bound=float(
                    flow_tail_bound
                ),
            )
        except TypeError:
            base = Hier3HeadModel(
                num_classes=int(num_classes),
                width=int(width),
                blocks=int(blocks),
                drop2d=float(drop2d),
                use_pt_as_feature=bool(
                    use_pt_as_feature
                ),
                head_dropout=float(
                    head_dropout
                ),
            )

        self.encoder = base.encoder
        self.head_mat = base.head_mat
        self.soc_flow = base.soc_flow

        self.use_pt = bool(
            use_pt_as_feature
        )
        self.num_classes = int(
            num_classes
        )
        self.width = int(width)

        pt_dim = 1 if self.use_pt else 0

        # SOH context:
        # encoder representation + material probabilities + SOC value + pt
        soh_in_dim = (
            int(width)
            + int(num_classes)
            + 1
            + pt_dim
        )

        self.head_soh_mu = nn.Sequential(
            nn.Linear(
                soh_in_dim,
                int(soh_hidden),
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(
                float(head_dropout)
            ),
            nn.Linear(
                int(soh_hidden),
                1,
            ),
        )

        self.head_soh_logvar = nn.Sequential(
            nn.Linear(
                soh_in_dim,
                int(soh_hidden),
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(
                float(head_dropout)
            ),
            nn.Linear(
                int(soh_hidden),
                1,
            ),
        )

    def build_material_logits(
        self,
        z: torch.Tensor,
        pt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Keep compatibility with implementations whose material head either
        includes or excludes pulse width.
        """
        first_linear = None

        for layer in self.head_mat.modules():
            if isinstance(layer, nn.Linear):
                first_linear = layer
                break

        if first_linear is None:
            raise RuntimeError(
                "Material head contains no Linear layer."
            )

        if (
            self.use_pt
            and int(first_linear.in_features)
            == int(z.shape[1] + pt.shape[1])
        ):
            return self.head_mat(
                torch.cat(
                    [z, pt],
                    dim=1,
                )
            )

        return self.head_mat(z)

    def forward(
        self,
        x_img: torch.Tensor,
        x_pt: torch.Tensor,
        soc_tf: Optional[torch.Tensor] = None,
        n_mc: int = EVAL_SOC_MC_SAMPLES,
    ):
        z = self.encoder(x_img)

        logits_mat = self.build_material_logits(
            z=z,
            pt=x_pt,
        )

        p_mat = torch.softmax(
            logits_mat,
            dim=1,
        )

        if self.use_pt:
            cond_soc = torch.cat(
                [z, p_mat, x_pt],
                dim=1,
            )
        else:
            cond_soc = torch.cat(
                [z, p_mat],
                dim=1,
            )

        soc_pred = flow_sample_mean(
            flow_module=self.soc_flow,
            context=cond_soc,
            num_samples=int(n_mc),
        ).view(-1)

        if soc_tf is not None:
            soc_context = (
                soc_tf.detach()
                .view(-1, 1)
            )
        else:
            soc_context = (
                soc_pred.detach()
                .view(-1, 1)
            )

        if self.use_pt:
            cond_soh = torch.cat(
                [
                    z,
                    p_mat,
                    soc_context,
                    x_pt,
                ],
                dim=1,
            )
        else:
            cond_soh = torch.cat(
                [
                    z,
                    p_mat,
                    soc_context,
                ],
                dim=1,
            )

        soh_mu = (
            self.head_soh_mu(
                cond_soh
            )
            .squeeze(1)
        )

        soh_logvar = (
            self.head_soh_logvar(
                cond_soh
            )
            .squeeze(1)
        )

        soh_logvar = torch.clamp(
            soh_logvar,
            min=-10.0,
            max=5.0,
        )

        soh_sigma = torch.exp(
            0.5 * soh_logvar
        )

        return (
            logits_mat,
            soc_pred,
            cond_soc,
            soh_mu,
            soh_logvar,
            soh_sigma,
            cond_soh,
        )


# =============================================================================
# Train and evaluation
# =============================================================================
def train_one_epoch(
    model: HybridFlowGaussianModel,
    loader: DataLoader,
    optimizer,
    device: str,
    criterion_cls,
    w_cls: float,
    w_soc: float,
    w_soh: float,
    grad_clip: float,
    train_soh_with_true_soc: bool,
    n_mc: int,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_cls = 0.0
    total_soc_nll = 0.0
    total_soh_nll = 0.0
    n_samples = 0

    y_cls_true = []
    y_cls_pred = []

    soc_true_all = []
    soc_pred_all = []

    soh_true_all = []
    soh_pred_all = []

    for x3, pt, y_cls, soc, soh in loader:
        x3 = x3.to(device)
        pt = pt.to(device)
        y_cls = y_cls.to(device)
        soc = soc.to(device)
        soh = soh.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        soc_tf = (
            soc
            if train_soh_with_true_soc
            else None
        )

        (
            logits,
            soc_pred,
            cond_soc,
            soh_mu,
            soh_logvar,
            _,
            _,
        ) = model(
            x_img=x3,
            x_pt=pt,
            soc_tf=soc_tf,
            n_mc=int(n_mc),
        )

        loss_cls = criterion_cls(
            logits,
            y_cls,
        )

        soc_log_prob = flow_log_prob(
            flow_module=model.soc_flow,
            target=soc,
            context=cond_soc,
        )

        loss_soc = -soc_log_prob.mean()

        loss_soh = heteroscedastic_nll(
            mu=soh_mu,
            logvar=soh_logvar,
            y=soh,
        )

        loss = (
            float(w_cls) * loss_cls
            + float(w_soc) * loss_soc
            + float(w_soh) * loss_soh
        )

        loss.backward()

        if (
            grad_clip
            and grad_clip > 0
        ):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(grad_clip),
            )

        optimizer.step()

        batch_size = int(
            y_cls.size(0)
        )

        total_loss += (
            float(loss.item())
            * batch_size
        )

        total_cls += (
            float(loss_cls.item())
            * batch_size
        )

        total_soc_nll += (
            float(loss_soc.item())
            * batch_size
        )

        total_soh_nll += (
            float(loss_soh.item())
            * batch_size
        )

        n_samples += batch_size

        y_cls_true.append(
            y_cls.detach().cpu().numpy()
        )

        y_cls_pred.append(
            logits.detach()
            .cpu()
            .argmax(dim=1)
            .numpy()
        )

        soc_true_all.append(
            soc.detach().cpu().numpy()
        )

        soc_pred_all.append(
            soc_pred.detach().cpu().numpy()
        )

        soh_true_all.append(
            soh.detach().cpu().numpy()
        )

        soh_pred_all.append(
            soh_mu.detach().cpu().numpy()
        )

    y_cls_true = np.concatenate(
        y_cls_true
    )

    y_cls_pred = np.concatenate(
        y_cls_pred
    )

    soc_true = np.concatenate(
        soc_true_all
    )

    soc_pred = np.concatenate(
        soc_pred_all
    )

    soh_true = np.concatenate(
        soh_true_all
    )

    soh_pred = np.concatenate(
        soh_pred_all
    )

    return {
        "loss": total_loss / n_samples,
        "cls_loss": total_cls / n_samples,
        "soc_nll": (
            total_soc_nll
            / n_samples
        ),
        "soh_nll": (
            total_soh_nll
            / n_samples
        ),
        "joint_nll": (
            total_soc_nll
            + total_soh_nll
        ) / n_samples,
        "cls_acc": float(
            accuracy_score(
                y_cls_true,
                y_cls_pred,
            )
        ),
        "soc_rmse": rmse(
            soc_true,
            soc_pred,
        ),
        "soc_mae": mae(
            soc_true,
            soc_pred,
        ),
        "soc_mape": mape(
            soc_true,
            soc_pred,
        ),
        "soc_medape": medape(
            soc_true,
            soc_pred,
        ),
        "soh_rmse": rmse(
            soh_true,
            soh_pred,
        ),
        "soh_mae": mae(
            soh_true,
            soh_pred,
        ),
        "soh_mape": mape(
            soh_true,
            soh_pred,
        ),
        "soh_medape": medape(
            soh_true,
            soh_pred,
        ),
    }


@torch.no_grad()
def eval_one_epoch(
    model: HybridFlowGaussianModel,
    loader: DataLoader,
    device: str,
    criterion_cls,
    w_cls: float,
    w_soc: float,
    w_soh: float,
    n_mc: int,
    soc_norm: Tuple[float, float],
    soh_norm: Tuple[float, float],
    normalize_soc: bool,
    zscore_normalize: bool,
    return_predictions: bool = False,
    return_probabilistic_samples: bool = False,
    flow_eval_samples: int = FLOW_EVAL_SAMPLES,
) -> Dict:
    model.eval()

    total_loss = 0.0
    total_cls = 0.0
    total_soc_nll = 0.0
    total_soh_nll = 0.0
    n_samples = 0

    y_cls_true_all = []
    y_cls_pred_all = []

    soc_true_all = []
    soc_pred_all = []
    soc_log_prob_all = []
    soc_samples_all = []

    soh_true_all = []
    soh_mu_all = []
    soh_logvar_all = []
    soh_sigma_all = []
    soh_log_prob_all = []

    for x3, pt, y_cls, soc, soh in loader:
        x3 = x3.to(device)
        pt = pt.to(device)
        y_cls = y_cls.to(device)
        soc = soc.to(device)
        soh = soh.to(device)

        (
            logits,
            soc_pred,
            cond_soc,
            soh_mu,
            soh_logvar,
            soh_sigma,
            _,
        ) = model(
            x_img=x3,
            x_pt=pt,
            soc_tf=None,
            n_mc=int(n_mc),
        )

        loss_cls = criterion_cls(
            logits,
            y_cls,
        )

        soc_log_prob = flow_log_prob(
            flow_module=model.soc_flow,
            target=soc,
            context=cond_soc,
        )

        loss_soc = -soc_log_prob.mean()

        loss_soh = heteroscedastic_nll(
            mu=soh_mu,
            logvar=soh_logvar,
            y=soh,
        )

        loss = (
            float(w_cls) * loss_cls
            + float(w_soc) * loss_soc
            + float(w_soh) * loss_soh
        )

        batch_size = int(
            y_cls.size(0)
        )

        total_loss += (
            float(loss.item())
            * batch_size
        )

        total_cls += (
            float(loss_cls.item())
            * batch_size
        )

        total_soc_nll += (
            float(loss_soc.item())
            * batch_size
        )

        total_soh_nll += (
            float(loss_soh.item())
            * batch_size
        )

        n_samples += batch_size

        y_cls_true_all.append(
            y_cls.detach().cpu().numpy()
        )

        y_cls_pred_all.append(
            logits.detach()
            .cpu()
            .argmax(dim=1)
            .numpy()
        )

        soc_true_all.append(
            soc.detach().cpu().numpy()
        )

        soc_pred_all.append(
            soc_pred.detach().cpu().numpy()
        )

        soc_log_prob_all.append(
            soc_log_prob.detach()
            .cpu()
            .numpy()
        )

        soh_true_all.append(
            soh.detach().cpu().numpy()
        )

        soh_mu_all.append(
            soh_mu.detach().cpu().numpy()
        )

        soh_logvar_all.append(
            soh_logvar.detach()
            .cpu()
            .numpy()
        )

        soh_sigma_all.append(
            soh_sigma.detach()
            .cpu()
            .numpy()
        )

        soh_dist = torch.distributions.Normal(
            soh_mu,
            torch.clamp(
                soh_sigma,
                min=SIGMA_MIN,
            ),
        )

        soh_log_prob_all.append(
            soh_dist.log_prob(soh)
            .detach()
            .cpu()
            .numpy()
        )

        if return_probabilistic_samples:
            soc_samples = sample_from_flow(
                flow_module=model.soc_flow,
                context=cond_soc,
                num_samples=int(
                    flow_eval_samples
                ),
            )

            soc_samples_all.append(
                soc_samples.detach()
                .cpu()
                .numpy()
            )

    y_cls_true = np.concatenate(
        y_cls_true_all
    )

    y_cls_pred = np.concatenate(
        y_cls_pred_all
    )

    soc_true_z = np.concatenate(
        soc_true_all
    )

    soc_pred_z = np.concatenate(
        soc_pred_all
    )

    soc_log_prob = np.concatenate(
        soc_log_prob_all
    )

    soh_true_z = np.concatenate(
        soh_true_all
    )

    soh_mu_z = np.concatenate(
        soh_mu_all
    )

    soh_logvar_z = np.concatenate(
        soh_logvar_all
    )

    soh_sigma_z = np.concatenate(
        soh_sigma_all
    )

    soh_log_prob = np.concatenate(
        soh_log_prob_all
    )

    soc_true_raw = inverse_single_target(
        soc_true_z,
        "SOC",
        soc_norm,
        soh_norm,
        normalize_soc,
        zscore_normalize,
    )

    soc_pred_raw = inverse_single_target(
        soc_pred_z,
        "SOC",
        soc_norm,
        soh_norm,
        normalize_soc,
        zscore_normalize,
    )

    soh_true_raw = inverse_single_target(
        soh_true_z,
        "SOH",
        soc_norm,
        soh_norm,
        normalize_soc,
        zscore_normalize,
    )

    soh_pred_raw = inverse_single_target(
        soh_mu_z,
        "SOH",
        soc_norm,
        soh_norm,
        normalize_soc,
        zscore_normalize,
    )

    out = {
        "loss": total_loss / n_samples,
        "cls_loss": total_cls / n_samples,
        "soc_nll": (
            total_soc_nll
            / n_samples
        ),
        "soh_nll": (
            total_soh_nll
            / n_samples
        ),
        "joint_nll": (
            total_soc_nll
            + total_soh_nll
        ) / n_samples,
        "cls_acc": float(
            accuracy_score(
                y_cls_true,
                y_cls_pred,
            )
        ),
        "soc_rmse_raw": rmse(
            soc_true_raw,
            soc_pred_raw,
        ),
        "soc_mae_raw": mae(
            soc_true_raw,
            soc_pred_raw,
        ),
        "soc_mape_raw": mape(
            soc_true_raw,
            soc_pred_raw,
        ),
        "soc_medape_raw": medape(
            soc_true_raw,
            soc_pred_raw,
        ),
        "soh_rmse_raw": rmse(
            soh_true_raw,
            soh_pred_raw,
        ),
        "soh_mae_raw": mae(
            soh_true_raw,
            soh_pred_raw,
        ),
        "soh_mape_raw": mape(
            soh_true_raw,
            soh_pred_raw,
        ),
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
                "soc_true_z": soc_true_z,
                "soc_pred_z": soc_pred_z,
                "soc_log_prob_z": soc_log_prob,
                "soc_true_raw": soc_true_raw,
                "soc_pred_raw": soc_pred_raw,
                "soh_true_z": soh_true_z,
                "soh_pred_z": soh_mu_z,
                "soh_logvar_z": soh_logvar_z,
                "soh_sigma_z": soh_sigma_z,
                "soh_log_prob_z": soh_log_prob,
                "soh_true_raw": soh_true_raw,
                "soh_pred_raw": soh_pred_raw,
            }
        )

        out["predictions"] = pred_df

    if return_probabilistic_samples:
        soc_samples_z = np.concatenate(
            soc_samples_all,
            axis=0,
        )

        soc_samples_raw = inverse_single_target(
            soc_samples_z,
            "SOC",
            soc_norm,
            soh_norm,
            normalize_soc,
            zscore_normalize,
        )

        soh_scale = target_scale_factor(
            "SOH",
            soc_norm,
            soh_norm,
            normalize_soc,
            zscore_normalize,
        )

        soh_sigma_raw = (
            soh_sigma_z
            * soh_scale
        )

        out["probabilistic"] = {
            "soc_samples_z": soc_samples_z,
            "soc_samples_raw": soc_samples_raw,
            "soh_mu_raw": soh_pred_raw,
            "soh_sigma_raw": soh_sigma_raw,
            "soc_log_prob_z": soc_log_prob,
            "soh_log_prob_z": soh_log_prob,
            "soc_true_raw": soc_true_raw,
            "soh_true_raw": soh_true_raw,
        }

    return out


# =============================================================================
# Probabilistic summary
# =============================================================================
def summarize_probabilistic_metrics(
    probabilistic: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    per_sample_frames = []

    # -------------------------------------------------------------------------
    # SOC flow
    # -------------------------------------------------------------------------
    soc_true = probabilistic[
        "soc_true_raw"
    ]

    soc_samples = probabilistic[
        "soc_samples_raw"
    ]

    soc_log_prob = probabilistic[
        "soc_log_prob_z"
    ]

    soc_crps = sample_crps(
        y=soc_true,
        samples=soc_samples,
    )

    soc_interval = (
        interval_metrics_from_samples(
            y=soc_true,
            samples=soc_samples,
            levels=INTERVAL_LEVELS,
        )
    )

    soc_row = {
        "model": "Flow-Gaussian hybrid",
        "target": "SOC",
        "distribution": "Conditional flow",
        "n_test": int(
            len(soc_true)
        ),
        "nll_normalized": float(
            -np.mean(soc_log_prob)
        ),
        "crps_raw": float(
            np.mean(soc_crps)
        ),
    }

    soc_per = pd.DataFrame(
        {
            "model": (
                "Flow-Gaussian hybrid"
            ),
            "target": "SOC",
            "sample_index": np.arange(
                len(soc_true)
            ),
            "true_raw": soc_true,
            "predictive_mean_raw": (
                np.mean(
                    soc_samples,
                    axis=1,
                )
            ),
            "predictive_std_raw": (
                np.std(
                    soc_samples,
                    axis=1,
                )
            ),
            "log_prob_normalized": (
                soc_log_prob
            ),
            "nll_normalized": (
                -soc_log_prob
            ),
            "crps_raw": soc_crps,
        }
    )

    for level in INTERVAL_LEVELS:
        key = int(
            round(level * 100)
        )

        coverage = float(
            np.mean(
                soc_interval[
                    f"covered_{key}"
                ]
            )
            * 100.0
        )

        width = float(
            np.mean(
                soc_interval[
                    f"width_{key}"
                ]
            )
        )

        soc_row[
            f"coverage_{key}_pct"
        ] = coverage

        soc_row[
            f"coverage_error_{key}_pp"
        ] = abs(
            coverage
            - 100.0 * level
        )

        soc_row[
            f"mean_interval_width_{key}_raw"
        ] = width

        soc_per[
            f"lower_{key}_raw"
        ] = soc_interval[
            f"lower_{key}"
        ]

        soc_per[
            f"upper_{key}_raw"
        ] = soc_interval[
            f"upper_{key}"
        ]

        soc_per[
            f"covered_{key}"
        ] = soc_interval[
            f"covered_{key}"
        ]

        soc_per[
            f"interval_width_{key}_raw"
        ] = soc_interval[
            f"width_{key}"
        ]

    rows.append(soc_row)
    per_sample_frames.append(soc_per)

    # -------------------------------------------------------------------------
    # SOH Gaussian
    # -------------------------------------------------------------------------
    soh_true = probabilistic[
        "soh_true_raw"
    ]

    soh_mu = probabilistic[
        "soh_mu_raw"
    ]

    soh_sigma = probabilistic[
        "soh_sigma_raw"
    ]

    soh_log_prob = probabilistic[
        "soh_log_prob_z"
    ]

    soh_crps = gaussian_crps(
        y=soh_true,
        mu=soh_mu,
        sigma=soh_sigma,
    )

    soh_interval = (
        interval_metrics_gaussian(
            y=soh_true,
            mu=soh_mu,
            sigma=soh_sigma,
            levels=INTERVAL_LEVELS,
        )
    )

    soh_row = {
        "model": "Flow-Gaussian hybrid",
        "target": "SOH",
        "distribution": (
            "Heteroscedastic Gaussian"
        ),
        "n_test": int(
            len(soh_true)
        ),
        "nll_normalized": float(
            -np.mean(soh_log_prob)
        ),
        "crps_raw": float(
            np.mean(soh_crps)
        ),
    }

    soh_per = pd.DataFrame(
        {
            "model": (
                "Flow-Gaussian hybrid"
            ),
            "target": "SOH",
            "sample_index": np.arange(
                len(soh_true)
            ),
            "true_raw": soh_true,
            "predictive_mean_raw": (
                soh_mu
            ),
            "predictive_std_raw": (
                soh_sigma
            ),
            "log_prob_normalized": (
                soh_log_prob
            ),
            "nll_normalized": (
                -soh_log_prob
            ),
            "crps_raw": soh_crps,
        }
    )

    for level in INTERVAL_LEVELS:
        key = int(
            round(level * 100)
        )

        coverage = float(
            np.mean(
                soh_interval[
                    f"covered_{key}"
                ]
            )
            * 100.0
        )

        width = float(
            np.mean(
                soh_interval[
                    f"width_{key}"
                ]
            )
        )

        soh_row[
            f"coverage_{key}_pct"
        ] = coverage

        soh_row[
            f"coverage_error_{key}_pp"
        ] = abs(
            coverage
            - 100.0 * level
        )

        soh_row[
            f"mean_interval_width_{key}_raw"
        ] = width

        soh_per[
            f"lower_{key}_raw"
        ] = soh_interval[
            f"lower_{key}"
        ]

        soh_per[
            f"upper_{key}_raw"
        ] = soh_interval[
            f"upper_{key}"
        ]

        soh_per[
            f"covered_{key}"
        ] = soh_interval[
            f"covered_{key}"
        ]

        soh_per[
            f"interval_width_{key}_raw"
        ] = soh_interval[
            f"width_{key}"
        ]

    rows.append(soh_row)
    per_sample_frames.append(soh_per)

    metrics_df = pd.DataFrame(rows)

    per_sample_df = pd.concat(
        per_sample_frames,
        axis=0,
        ignore_index=True,
    )

    return metrics_df, per_sample_df


# =============================================================================
# Main experiment
# =============================================================================
def run_experiment(
    data_root: str | Path = DATA_ROOT,
    pulse_list: Optional[List[int]] = None,
    exp_dir: str | Path = EXP_DIR,
    max_epochs: int = MAX_EPOCHS,
    resume: bool = RESUME,
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

    set_random_seed(SEED)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    run_config = {
        "data_root": str(data_root),
        "pulse_list": list(
            map(int, pulse_list)
        ),
        "exp_dir": str(exp_dir),
        "u_start": U_START,
        "u_end": U_END,
        "drop_first_class": DROP_FIRST_CLASS,
        "soc_col": SOC_COL,
        "soh_col": SOH_COL,
        "use_pt_as_feature": (
            USE_PT_AS_FEATURE
        ),
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "max_epochs": int(max_epochs),
        "early_stopping": (
            EARLY_STOPPING
        ),
        "patience": PATIENCE,
        "resume": bool(resume),
        "num_workers": NUM_WORKERS,
        "seed": SEED,
        "width": WIDTH,
        "blocks": BLOCKS,
        "drop2d": DROP2D,
        "head_dropout": HEAD_DROPOUT,
        "flow_layers": FLOW_LAYERS,
        "flow_bins": FLOW_BINS,
        "flow_tail_bound": (
            FLOW_TAIL_BOUND
        ),
        "soh_hidden": SOH_HIDDEN,
        "w_cls": W_CLS,
        "w_soc": W_SOC,
        "w_soh": W_SOH,
        "test_id_frac": TEST_ID_FRAC,
        "test_id_count": TEST_ID_COUNT,
        "normalize_soc": NORMALIZE_SOC,
        "zscore_normalize": (
            ZSCORE_NORMALIZE
        ),
        "normalize_u_with_train_stats": (
            NORMALIZE_U_WITH_TRAIN_STATS
        ),
        "train_soh_with_true_soc": (
            TRAIN_SOH_WITH_TRUE_SOC
        ),
        "eval_soc_mc_samples": (
            EVAL_SOC_MC_SAMPLES
        ),
        "flow_eval_samples": (
            FLOW_EVAL_SAMPLES
        ),
        "interval_levels": (
            INTERVAL_LEVELS
        ),
    }

    save_json(
        exp_dir / "run_config.json",
        run_config,
    )

    # -------------------------------------------------------------------------
    # Build raw data
    # -------------------------------------------------------------------------
    train_kwargs = {
        "data_root": str(data_root),
        "soc_list": list(
            range(5, 90, 5)
        ),
        "pulse_list": list(
            map(int, pulse_list)
        ),
        "u_start": U_START,
        "u_end": U_END,
        "drop_first_class": (
            DROP_FIRST_CLASS
        ),
    }

    (
        Xtr_raw,
        ytr_raw,
        mtr_raw,
        tag_tr,
        hit_tr,
    ) = load_or_build_cache(
        str(cache_dir),
        "raw_train",
        build_train_mix_soc_mix_pt,
        train_kwargs,
    )

    test_kwargs = {
        "data_root": str(data_root),
        "pulse_list": list(
            map(int, pulse_list)
        ),
        "u_start": U_START,
        "u_end": U_END,
        "drop_first_class": (
            DROP_FIRST_CLASS
        ),
    }

    (
        Xte_raw,
        yte_raw,
        mte_raw,
        tag_te,
        hit_te,
    ) = load_or_build_cache(
        str(cache_dir),
        "raw_test",
        build_test_random_mix_pt,
        test_kwargs,
    )

    print(
        f"[CACHE] Train tag: "
        f"{tag_tr} | hit={hit_tr}"
    )

    print(
        f"[CACHE] Test  tag: "
        f"{tag_te} | hit={hit_te}"
    )

    (
        Xtr_raw,
        ytr_raw,
        mtr_raw,
    ) = drop_nan_inf_rows(
        Xtr_raw,
        ytr_raw,
        mtr_raw,
        "RAW_TRAIN",
    )

    (
        Xte_raw,
        yte_raw,
        mte_raw,
    ) = drop_nan_inf_rows(
        Xte_raw,
        yte_raw,
        mte_raw,
        "RAW_TEST",
    )

    if (
        Xtr_raw.shape[1] != 41
        or Xte_raw.shape[1] != 41
    ):
        raise ValueError(
            "Expected U1-U41 inputs. "
            f"Got train={Xtr_raw.shape}, "
            f"test={Xte_raw.shape}."
        )

    # -------------------------------------------------------------------------
    # Test ID split
    # -------------------------------------------------------------------------
    all_ids = pd.concat(
        [
            mtr_raw["ID"],
            mte_raw["ID"],
        ],
        axis=0,
    ).astype(str).to_numpy()

    test_ids = pick_test_ids(
        all_ids=all_ids,
        test_id_frac=TEST_ID_FRAC,
        test_id_count=TEST_ID_COUNT,
        seed=SEED,
    )

    if (
        TEST_ID_COUNT
        and TEST_ID_COUNT > 0
    ):
        split_name = (
            f"testIDs_seed{SEED}_"
            f"n{TEST_ID_COUNT}"
        )
    else:
        split_name = (
            f"testIDs_seed{SEED}_"
            f"frac{TEST_ID_FRAC}"
        )

    split_path = (
        splits_dir
        / f"{split_name}.txt"
    )

    with open(
        split_path,
        "w",
        encoding="utf-8",
    ) as f:
        for test_id in test_ids:
            f.write(
                str(test_id) + "\n"
            )

    (
        Xtr,
        ytr_str,
        mtr,
        Xte,
        yte_str,
        mte,
    ) = apply_id_split(
        Xtr=Xtr_raw,
        ytr_str=ytr_raw,
        mtr=mtr_raw,
        Xte=Xte_raw,
        yte_str=yte_raw,
        mte=mte_raw,
        test_ids=test_ids,
    )

    if (
        len(ytr_str) == 0
        or len(yte_str) == 0
    ):
        raise RuntimeError(
            "Empty train or test data "
            "after applying ID split."
        )

    print(
        f"[DATA] Final TRAIN samples = "
        f"{len(ytr_str)} | unique IDs = "
        f"{mtr['ID'].astype(str).nunique()}"
    )

    print(
        f"[DATA] Final TEST samples = "
        f"{len(yte_str)} | unique IDs = "
        f"{mte['ID'].astype(str).nunique()}"
    )

    # -------------------------------------------------------------------------
    # U normalization
    # -------------------------------------------------------------------------
    if NORMALIZE_U_WITH_TRAIN_STATS:
        u_mean = Xtr.mean(
            axis=0,
            keepdims=True,
        )

        u_std = (
            Xtr.std(
                axis=0,
                keepdims=True,
            )
            + 1e-8
        )

        Xtr = (
            Xtr - u_mean
        ) / u_std

        Xte = (
            Xte - u_mean
        ) / u_std

        np.savez_compressed(
            exp_dir
            / "u41_norm_train_only.npz",
            u_mean=u_mean.astype(
                np.float32
            ),
            u_std=u_std.astype(
                np.float32
            ),
        )

        print(
            "[NORM] Applied U1-U41 "
            "train-only z-score."
        )

    # -------------------------------------------------------------------------
    # Target normalization
    # -------------------------------------------------------------------------
    soc_train = (
        mtr[SOC_COL]
        .astype(float)
        .to_numpy(dtype=np.float64)
    )

    if NORMALIZE_SOC:
        soc_train = (
            soc_train / 100.0
        )

    soc_norm = (
        float(soc_train.mean()),
        float(
            soc_train.std() + 1e-8
        ),
    )

    soh_train = (
        mtr[SOH_COL]
        .astype(float)
        .to_numpy(dtype=np.float64)
    )

    soh_norm = (
        float(soh_train.mean()),
        float(
            soh_train.std() + 1e-8
        ),
    )

    np.savez_compressed(
        exp_dir
        / "target_norm_train_only.npz",
        soc_mean=np.array(
            [soc_norm[0]],
            dtype=np.float32,
        ),
        soc_std=np.array(
            [soc_norm[1]],
            dtype=np.float32,
        ),
        soh_mean=np.array(
            [soh_norm[0]],
            dtype=np.float32,
        ),
        soh_std=np.array(
            [soh_norm[1]],
            dtype=np.float32,
        ),
    )

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------
    label_encoder = LabelEncoder()

    ytr_cls = label_encoder.fit_transform(
        ytr_str
    )

    train_classes = set(
        label_encoder.classes_.tolist()
    )

    mask_known = np.array(
        [
            label in train_classes
            for label in yte_str
        ],
        dtype=bool,
    )

    if not mask_known.all():
        n_removed = int(
            (~mask_known).sum()
        )

        print(
            f"[WARN] Removing {n_removed} "
            "test samples with unseen labels."
        )

        Xte = Xte[mask_known]
        yte_str = yte_str[mask_known]
        mte = (
            mte.loc[mask_known]
            .reset_index(drop=True)
        )

    yte_cls = label_encoder.transform(
        yte_str
    )

    class_names = list(
        label_encoder.classes_
    )

    num_classes = len(
        class_names
    )

    save_json(
        exp_dir
        / "label_mapping.json",
        {
            "classes": class_names,
            "split_name": split_name,
        },
    )

    # -------------------------------------------------------------------------
    # Pulse-width normalization
    # -------------------------------------------------------------------------
    if "pulse_ms" in mtr.columns:
        pt_col = "pulse_ms"
    elif "pulse_width_ms" in mtr.columns:
        pt_col = "pulse_width_ms"
    else:
        raise RuntimeError(
            "No pulse-width column found."
        )

    pt_train = np.log1p(
        mtr[pt_col]
        .astype(float)
        .to_numpy(dtype=np.float64)
    )

    pt_norm = (
        float(pt_train.mean()),
        float(
            pt_train.std() + 1e-8
        ),
    )

    # -------------------------------------------------------------------------
    # Datasets and loaders
    # -------------------------------------------------------------------------
    ds_tr = HierPulseDataset(
        X_u=Xtr,
        y_cls=ytr_cls,
        meta=mtr,
        soc_col=SOC_COL,
        soh_col=SOH_COL,
        pt_col=pt_col,
        use_pt_as_feature=(
            USE_PT_AS_FEATURE
        ),
        pt_norm=pt_norm,
        normalize_soc=NORMALIZE_SOC,
        zscore_normalize=(
            ZSCORE_NORMALIZE
        ),
        soc_norm=soc_norm,
        soh_norm=soh_norm,
    )

    ds_te = HierPulseDataset(
        X_u=Xte,
        y_cls=yte_cls,
        meta=mte,
        soc_col=SOC_COL,
        soh_col=SOH_COL,
        pt_col=pt_col,
        use_pt_as_feature=(
            USE_PT_AS_FEATURE
        ),
        pt_norm=pt_norm,
        normalize_soc=NORMALIZE_SOC,
        zscore_normalize=(
            ZSCORE_NORMALIZE
        ),
        soc_norm=soc_norm,
        soh_norm=soh_norm,
    )

    dl_tr = DataLoader(
        ds_tr,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    dl_te = DataLoader(
        ds_te,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model = HybridFlowGaussianModel(
        num_classes=num_classes,
        width=WIDTH,
        blocks=BLOCKS,
        drop2d=DROP2D,
        use_pt_as_feature=(
            USE_PT_AS_FEATURE
        ),
        head_dropout=HEAD_DROPOUT,
        flow_layers=FLOW_LAYERS,
        flow_bins=FLOW_BINS,
        flow_tail_bound=(
            FLOW_TAIL_BOUND
        ),
        soh_hidden=SOH_HIDDEN,
    ).to(device)

    criterion_cls = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(LR),
        weight_decay=float(
            WEIGHT_DECAY
        ),
    )

    last_path = (
        ckpt_dir / "last.pt"
    )

    best_path = (
        ckpt_dir / "best.pt"
    )

    start_epoch = 0
    best_score = float("inf")
    best_epoch = -1
    bad_count = 0

    if (
        resume
        and last_path.exists()
    ):
        ckpt = torch_load_compatible(
            last_path,
            map_location=device,
        )

        model.load_state_dict(
            extract_state_dict(ckpt)
        )

        if "optim" in ckpt:
            optimizer.load_state_dict(
                ckpt["optim"]
            )

        start_epoch = int(
            ckpt.get("epoch", -1)
            + 1
        )

        best_score = float(
            ckpt.get(
                "best_score",
                float("inf"),
            )
        )

        best_epoch = int(
            ckpt.get(
                "best_epoch",
                -1,
            )
        )

        bad_count = int(
            ckpt.get(
                "bad_count",
                0,
            )
        )

        print(
            f"[RESUME] start_epoch="
            f"{start_epoch}, "
            f"best_score={best_score:.6f}, "
            f"best_epoch={best_epoch}"
        )

    log_path = (
        logs_dir
        / "train_log.csv"
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    for epoch in range(
        start_epoch,
        int(max_epochs),
    ):
        epoch_start = time.time()

        tr = train_one_epoch(
            model=model,
            loader=dl_tr,
            optimizer=optimizer,
            device=device,
            criterion_cls=criterion_cls,
            w_cls=W_CLS,
            w_soc=W_SOC,
            w_soh=W_SOH,
            grad_clip=GRAD_CLIP,
            train_soh_with_true_soc=(
                TRAIN_SOH_WITH_TRUE_SOC
            ),
            n_mc=EVAL_SOC_MC_SAMPLES,
        )

        te = eval_one_epoch(
            model=model,
            loader=dl_te,
            device=device,
            criterion_cls=criterion_cls,
            w_cls=W_CLS,
            w_soc=W_SOC,
            w_soh=W_SOH,
            n_mc=EVAL_SOC_MC_SAMPLES,
            soc_norm=soc_norm,
            soh_norm=soh_norm,
            normalize_soc=NORMALIZE_SOC,
            zscore_normalize=(
                ZSCORE_NORMALIZE
            ),
            return_predictions=False,
            return_probabilistic_samples=False,
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        # Select the best checkpoint using TRAIN metrics only.
        # Lower is better.
        selection_score = (
            tr["joint_nll"]
            + 0.1 * (
                tr["soc_rmse"]
                + tr["soh_rmse"]
            )
            - 0.05 * tr["cls_acc"]
        )

        row = pd.DataFrame(
            [
                {
                    "epoch": int(epoch),
                    "train_loss": tr["loss"],
                    "train_cls_loss": (
                        tr["cls_loss"]
                    ),
                    "train_soc_nll": (
                        tr["soc_nll"]
                    ),
                    "train_soh_nll": (
                        tr["soh_nll"]
                    ),
                    "train_joint_nll": (
                        tr["joint_nll"]
                    ),
                    "train_cls_acc": (
                        tr["cls_acc"]
                    ),
                    "train_soc_rmse": (
                        tr["soc_rmse"]
                    ),
                    "train_soc_medape": (
                        tr["soc_medape"]
                    ),
                    "train_soh_rmse": (
                        tr["soh_rmse"]
                    ),
                    "train_soh_medape": (
                        tr["soh_medape"]
                    ),
                    "test_loss": te["loss"],
                    "test_cls_loss": (
                        te["cls_loss"]
                    ),
                    "test_soc_nll": (
                        te["soc_nll"]
                    ),
                    "test_soh_nll": (
                        te["soh_nll"]
                    ),
                    "test_joint_nll": (
                        te["joint_nll"]
                    ),
                    "test_cls_acc": (
                        te["cls_acc"]
                    ),
                    "test_soc_rmse_raw": (
                        te["soc_rmse_raw"]
                    ),
                    "test_soc_medape_raw": (
                        te["soc_medape_raw"]
                    ),
                    "test_soh_rmse_raw": (
                        te["soh_rmse_raw"]
                    ),
                    "test_soh_medape_raw": (
                        te["soh_medape_raw"]
                    ),
                    "train_selection_score": (
                        selection_score
                    ),
                    "best_score_so_far": min(
                        best_score,
                        selection_score,
                    ),
                    "epoch_duration_sec": (
                        elapsed
                    ),
                }
            ]
        )

        if not log_path.exists():
            row.to_csv(
                log_path,
                index=False,
                encoding="utf-8-sig",
            )
        else:
            row.to_csv(
                log_path,
                mode="a",
                header=False,
                index=False,
                encoding="utf-8-sig",
            )

        improved = (
            selection_score
            < best_score
        )

        if improved:
            best_score = float(
                selection_score
            )

            best_epoch = int(epoch)
            bad_count = 0

            torch.save(
                {
                    "epoch": int(epoch),
                    "model": (
                        model.state_dict()
                    ),
                    "optim": (
                        optimizer.state_dict()
                    ),
                    "best_score": (
                        best_score
                    ),
                    "best_epoch": (
                        best_epoch
                    ),
                    "run_config": (
                        run_config
                    ),
                },
                best_path,
            )
        else:
            bad_count += 1

        torch.save(
            {
                "epoch": int(epoch),
                "model": (
                    model.state_dict()
                ),
                "optim": (
                    optimizer.state_dict()
                ),
                "best_score": (
                    best_score
                ),
                "best_epoch": (
                    best_epoch
                ),
                "bad_count": (
                    bad_count
                ),
                "run_config": (
                    run_config
                ),
            },
            last_path,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"TE cls={te['cls_acc']:.4f} | "
            f"SOC NLL={te['soc_nll']:.4f} | "
            f"SOH NLL={te['soh_nll']:.4f} | "
            f"Joint NLL="
            f"{te['joint_nll']:.4f} | "
            f"SOC MedAPE(raw)="
            f"{te['soc_medape_raw']:.3f}% | "
            f"SOH MedAPE(raw)="
            f"{te['soh_medape_raw']:.3f}% | "
            f"train_select="
            f"{selection_score:.6f} | "
            f"time={elapsed:.2f}s"
        )

        if (
            EARLY_STOPPING
            and bad_count >= PATIENCE
        ):
            print(
                f"[EARLY STOP] "
                f"best_score={best_score:.6f} "
                f"at epoch={best_epoch}"
            )
            break

    # -------------------------------------------------------------------------
    # Final evaluation
    # -------------------------------------------------------------------------
    if best_path.exists():
        ckpt = torch_load_compatible(
            best_path,
            map_location=device,
        )

        model.load_state_dict(
            extract_state_dict(ckpt)
        )

        print(
            f"[BEST] Loaded best checkpoint "
            f"from epoch="
            f"{ckpt.get('epoch')} | "
            f"score="
            f"{ckpt.get('best_score')}"
        )

    te = eval_one_epoch(
        model=model,
        loader=dl_te,
        device=device,
        criterion_cls=criterion_cls,
        w_cls=W_CLS,
        w_soc=W_SOC,
        w_soh=W_SOH,
        n_mc=EVAL_SOC_MC_SAMPLES,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
        normalize_soc=NORMALIZE_SOC,
        zscore_normalize=(
            ZSCORE_NORMALIZE
        ),
        return_predictions=True,
        return_probabilistic_samples=True,
        flow_eval_samples=(
            FLOW_EVAL_SAMPLES
        ),
    )

    pred_df = te.pop(
        "predictions"
    )

    probabilistic = te.pop(
        "probabilistic"
    )

    pred_df.to_csv(
        metrics_dir
        / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (
        probabilistic_df,
        per_sample_prob_df,
    ) = summarize_probabilistic_metrics(
        probabilistic
    )

    probabilistic_df.to_csv(
        metrics_dir
        / "probabilistic_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_sample_prob_df.to_csv(
        metrics_dir
        / "per_sample_probabilistic_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_metrics = {
        "best_epoch": int(
            best_epoch
        ),
        "best_score": float(
            best_score
        ),
        "test_cls_acc": float(
            te["cls_acc"]
        ),
        "test_soc_nll": float(
            te["soc_nll"]
        ),
        "test_soh_nll": float(
            te["soh_nll"]
        ),
        "test_joint_nll": float(
            te["joint_nll"]
        ),
        "test_soc_rmse_raw": float(
            te["soc_rmse_raw"]
        ),
        "test_soc_mae_raw": float(
            te["soc_mae_raw"]
        ),
        "test_soc_mape_raw": float(
            te["soc_mape_raw"]
        ),
        "test_soc_medape_raw": float(
            te["soc_medape_raw"]
        ),
        "test_soh_rmse_raw": float(
            te["soh_rmse_raw"]
        ),
        "test_soh_mae_raw": float(
            te["soh_mae_raw"]
        ),
        "test_soh_mape_raw": float(
            te["soh_mape_raw"]
        ),
        "test_soh_medape_raw": float(
            te["soh_medape_raw"]
        ),
        "n_train": int(
            len(ds_tr)
        ),
        "n_test": int(
            len(ds_te)
        ),
        "num_classes": int(
            num_classes
        ),
        "device": device,
        "elapsed_sec": float(
            time.time()
            - start_time
        ),
    }

    for _, row in probabilistic_df.iterrows():
        target = str(
            row["target"]
        ).lower()

        final_metrics[
            f"{target}_nll_normalized"
        ] = float(
            row["nll_normalized"]
        )

        final_metrics[
            f"{target}_crps_raw"
        ] = float(
            row["crps_raw"]
        )

        final_metrics[
            f"{target}_coverage_90_pct"
        ] = float(
            row["coverage_90_pct"]
        )

        final_metrics[
            f"{target}_coverage_error_90_pp"
        ] = float(
            row[
                "coverage_error_90_pp"
            ]
        )

        final_metrics[
            f"{target}_mean_interval_width_90_raw"
        ] = float(
            row[
                "mean_interval_width_90_raw"
            ]
        )

    pd.DataFrame(
        [final_metrics]
    ).to_csv(
        metrics_dir
        / "final_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        metrics_dir
        / "final_metrics.json",
        final_metrics,
    )

    print(
        "\n[HYBRID PROBABILISTIC METRICS]"
    )

    print(
        probabilistic_df.to_string(
            index=False
        )
    )

    print(
        "\n[FINAL METRICS]"
    )

    for key, value in (
        final_metrics.items()
    ):
        print(
            f"{key}: {value}"
        )

    print(
        "\n[OK] Saved hybrid outputs "
        f"under: {exp_dir}"
    )

    return final_metrics


def main() -> None:
    run_experiment(
        data_root=DATA_ROOT,
        pulse_list=(
            DEFAULT_PULSE_LIST
        ),
        exp_dir=EXP_DIR,
        max_epochs=MAX_EPOCHS,
        resume=RESUME,
    )


if __name__ == "__main__":
    main()
