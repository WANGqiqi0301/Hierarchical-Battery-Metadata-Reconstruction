# ablation/input_representation_ablation.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import sys
import json
import time
import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# =============================================================================
# Project path
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Local imports
# =============================================================================

from utils.cache import ensure_dir, load_or_build_cache, drop_nan_inf_rows
from utils.seed import set_random_seed

from proposed_framework.data.build_dataset import (
    build_train_mix_soc_mix_pt,
    build_test_random_mix_pt,
    pick_test_ids,
    apply_id_split,
)

from proposed_framework.models.conditional_flow import Conditional1DFlow
from proposed_framework.training.trainer import train_one_epoch
from proposed_framework.training.evaluator import eval_one_epoch


# =============================================================================
# Constants
# =============================================================================

DEFAULT_PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]


# =============================================================================
# Raw 1D feature builder
# =============================================================================

def build_1d_features_from_u41(u: np.ndarray) -> np.ndarray:
    """
    Use raw U1-U41 directly as a one-dimensional input vector.

    Output shape per sample:
        (41,)
    """
    u = np.asarray(u, dtype=np.float32)

    if u.shape[0] != 41:
        raise ValueError(f"Expected 41 U values, got {u.shape[0]}.")

    return u.astype(np.float32)


# =============================================================================
# Dataset
# =============================================================================

class Raw1DDataset(Dataset):
    """
    Dataset for raw 1D U1-U41 representation ablation.

    Each sample returns:
    - x_vec: raw normalized U1-U41 vector, shape (41,)
    - pt: normalized pulse-width feature, shape (1,)
    - y_cls: material-capacity class label
    - soc: SOC target
    - soh: SOH target
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
        self.X_u = X_u
        self.y_cls = y_cls.astype(np.int64)
        self.meta = meta.reset_index(drop=True)

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
                    "zscore_normalize=True requires soc_norm and soh_norm "
                    "from training data."
                )

            soc_mean, soc_std = float(soc_norm[0]), float(soc_norm[1])
            soh_mean, soh_std = float(soh_norm[0]), float(soh_norm[1])

            soc = (soc - soc_mean) / (soc_std + 1e-8)
            soh = (soh - soh_mean) / (soh_std + 1e-8)

        self.soc = soc
        self.soh = soh

        self.use_pt = bool(use_pt_as_feature)

        if self.use_pt and pt_col in self.meta.columns:
            self.pt_ms = self.meta[pt_col].astype(float).to_numpy(dtype=np.float32)
            p = np.log1p(self.pt_ms)

            if pt_norm is None:
                self.pt_mean = float(p.mean())
                self.pt_std = float(p.std() + 1e-8)
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
        x_vec = torch.from_numpy(build_1d_features_from_u41(self.X_u[idx]))
        y_cls = torch.tensor(int(self.y_cls[idx]), dtype=torch.long)

        if self.use_pt and self.pt_ms is not None:
            p = (np.log1p(float(self.pt_ms[idx])) - self.pt_mean) / self.pt_std
            pt = torch.tensor([p], dtype=torch.float32)
        else:
            pt = torch.tensor([0.0], dtype=torch.float32)

        soc = torch.tensor(float(self.soc[idx]), dtype=torch.float32)
        soh = torch.tensor(float(self.soh[idx]), dtype=torch.float32)

        return x_vec, pt, y_cls, soc, soh


# =============================================================================
# Raw 1D model
# =============================================================================

class ResBlock1D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm1d(channels)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))

        return self.act(x + h)


class CNN1DEncoder(nn.Module):
    """
    1D CNN encoder for raw U1-U41 input.

    Input:
        x shape = (B, 41) or (B, 1, 41)

    Output:
        z shape = (B, width)
    """

    def __init__(
        self,
        input_dim: int = 41,
        width: int = 64,
        blocks: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = int(input_dim)

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=int(width),
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm1d(int(width)),
            nn.ReLU(inplace=True),
        )

        self.body = nn.Sequential(
            *[
                ResBlock1D(int(width), dropout=dropout)
                for _ in range(int(blocks))
            ]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)

        if x.ndim != 3:
            raise RuntimeError(f"Expected x ndim 2 or 3, got shape={tuple(x.shape)}.")

        z = self.stem(x)
        z = self.body(z)
        z = self.pool(z).flatten(1)

        return z


class Raw1DHierModel(nn.Module):
    """
    Hierarchical probabilistic model using raw 1D U1-U41 input.

    This keeps the same hierarchy as the proposed model:
    1. Material-capacity classification.
    2. SOC conditional flow.
    3. SOH conditional flow.

    Difference:
    - Encoder is 1D CNN over raw U1-U41 rather than 2D CNN over 3x5x8.
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 41,
        width: int = 64,
        blocks: int = 3,
        drop1d: float = 0.0,
        use_pt_as_feature: bool = True,
        soc_hidden: int = 64,
        soh_hidden: int = 64,
        head_dropout: float = 0.2,
        flow_layers: int = 6,
        flow_bins: int = 8,
        flow_tail_bound: float = 3.0,
    ):
        super().__init__()

        self.encoder = CNN1DEncoder(
            input_dim=input_dim,
            width=width,
            blocks=blocks,
            dropout=drop1d,
        )

        self.use_pt = bool(use_pt_as_feature)

        pt_dim = 1 if self.use_pt else 0
        p_dim = int(num_classes)

        # Keep material head consistent with the updated proposed model:
        # material classification uses encoder embedding plus optional pulse width.
        mat_input_dim = int(width) + pt_dim

        self.head_mat = nn.Sequential(
            nn.Linear(mat_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(64, num_classes),
        )

        soc_context_dim = int(width) + p_dim + pt_dim

        self.soc_flow = Conditional1DFlow(
            context_dim=soc_context_dim,
            hidden_features=int(soc_hidden),
            num_transforms=int(flow_layers),
            num_bins=int(flow_bins),
            tail_bound=float(flow_tail_bound),
        )

        soh_context_dim = int(width) + p_dim + 1 + pt_dim

        self.soh_flow = Conditional1DFlow(
            context_dim=soh_context_dim,
            hidden_features=int(soh_hidden),
            num_transforms=int(flow_layers),
            num_bins=int(flow_bins),
            tail_bound=float(flow_tail_bound),
        )

    @staticmethod
    def _sample_mean_1d(
        samples: torch.Tensor,
        batch_size: int,
        num_samples: int,
        name: str,
    ) -> torch.Tensor:
        if samples.ndim == 3:
            if samples.shape[0] == int(num_samples) and samples.shape[1] == batch_size:
                return samples.mean(dim=0).squeeze(-1)

            if samples.shape[0] == batch_size and samples.shape[1] == int(num_samples):
                return samples.mean(dim=1).squeeze(-1)

            samples = samples.reshape(int(num_samples), batch_size, 1)
            return samples.mean(dim=0).squeeze(-1)

        if samples.ndim == 2:
            samples = samples.view(int(num_samples), batch_size, 1)
            return samples.mean(dim=0).squeeze(-1)

        raise RuntimeError(f"Unexpected {name} sample shape: {tuple(samples.shape)}")

    def forward(
        self,
        x_img: torch.Tensor,
        x_pt: torch.Tensor,
        soc_tf: Optional[torch.Tensor] = None,
        n_mc: int = 16,
    ):
        """
        The argument name x_img is kept for compatibility with shared trainer/evaluator.
        Here it is actually raw 1D U1-U41 input.
        """
        z = self.encoder(x_img)
        batch_size = z.size(0)

        if self.use_pt:
            z_mat = torch.cat([z, x_pt], dim=1)
        else:
            z_mat = z

        logits_mat = self.head_mat(z_mat)
        p_mat = torch.softmax(logits_mat, dim=1)

        if self.use_pt:
            cond_soc = torch.cat([z, p_mat, x_pt], dim=1)
        else:
            cond_soc = torch.cat([z, p_mat], dim=1)

        soc_logp = None

        if soc_tf is not None:
            soc_tf = soc_tf.view(-1)
            soc_logp = self.soc_flow.log_prob(soc_tf, cond_soc)

        with torch.no_grad():
            soc_samples = self.soc_flow.sample(cond_soc, num_samples=int(n_mc))
            soc_pred = self._sample_mean_1d(
                samples=soc_samples,
                batch_size=batch_size,
                num_samples=int(n_mc),
                name="SOC",
            )

        soc_pred = soc_pred.view(-1)

        if soc_tf is not None:
            soc_value = soc_tf.detach().view(-1, 1)
        else:
            soc_value = soc_pred.detach().view(-1, 1)

        if self.use_pt:
            cond_soh = torch.cat([z, p_mat, soc_value, x_pt], dim=1)
        else:
            cond_soh = torch.cat([z, p_mat, soc_value], dim=1)

        with torch.no_grad():
            soh_samples = self.soh_flow.sample(cond_soh, num_samples=int(n_mc))
            soh_pred = self._sample_mean_1d(
                samples=soh_samples,
                batch_size=batch_size,
                num_samples=int(n_mc),
                name="SOH",
            )

        soh_pred = soh_pred.view(-1)

        return logits_mat, soc_pred, soc_logp, cond_soc, soh_pred, cond_soh


# =============================================================================
# Helpers
# =============================================================================

def _torch_load(path: str | Path, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _save_json(path: str | Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_existing_best_checkpoint(
    exp_dir: str | Path,
    preferred_stage: Optional[str] = None,
) -> Tuple[str, Path]:
    exp_dir = Path(exp_dir)
    stages = []

    metrics_json = _load_json(exp_dir / "metrics" / "final_metrics.json")
    if metrics_json.get("final_stage"):
        stages.append(str(metrics_json["final_stage"]))

    metrics_csv = exp_dir / "metrics" / "final_metrics.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        if len(df) and "final_stage" in df.columns:
            stages.append(str(df.iloc[0]["final_stage"]))

    if preferred_stage:
        stages.append(str(preferred_stage))

    run_cfg = _load_json(exp_dir / "run_config.json")
    if run_cfg.get("final_best_stage"):
        stages.append(str(run_cfg["final_best_stage"]))

    stages.extend(["finetune", "stage2_soh", "stage1_soc", "single"])
    for stage in dict.fromkeys(stages):
        path = exp_dir / "checkpoints" / stage / "best.pt"
        if path.exists():
            return stage, path

    raise FileNotFoundError(f"No best checkpoint found under {exp_dir / 'checkpoints'}")


def _material_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: Optional[LabelEncoder],
) -> float:
    if label_encoder is None:
        return float("nan")
    true_labels = label_encoder.inverse_transform(np.asarray(y_true, dtype=np.int64))
    pred_labels = label_encoder.inverse_transform(np.asarray(y_pred, dtype=np.int64))
    true_material = [str(x).split("_")[0] for x in true_labels]
    pred_material = [str(x).split("_")[0] for x in pred_labels]
    return float(accuracy_score(true_material, pred_material))


def _stage_score(stage: str, te: dict, alpha_score: float) -> float:
    if stage == "stage1_soc":
        return float(te["cls_acc"] - 0.3 * te["soc_rmse"])

    if stage == "stage2_soh":
        return float(-te["soh_rmse"])

    return float(
        te["cls_acc"]
        - float(alpha_score) * (te["soc_rmse"] + te["soh_rmse"])
    )



def _inverse_targets(
    soc_z: np.ndarray,
    soh_z: np.ndarray,
    soc_norm: Optional[Tuple[float, float]],
    soh_norm: Optional[Tuple[float, float]],
    normalize_soc: bool,
    zscore_normalize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert model-space SOC/SOH values back to raw reporting units.

    SOC is trained as SOC/100 when normalize_soc=True. If z-score
    normalization is also enabled, the inverse order is z-score inverse first,
    then multiply SOC by 100. SOH is reported in its original unit.
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
        soc *= 100.0
    soh *= 100.0
    return soc, soh


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_pred - y_true)))


def _safe_medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64))))


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0)


def _safe_medape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.median(np.abs((y_pred - y_true) / denom)) * 100.0)


@torch.no_grad()
def eval_one_epoch_raw_1d(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    w_cls: float,
    w_soc: float,
    w_soh: float,
    criterion_cls: nn.Module,
    criterion_reg: nn.Module,
    soc_nll_weight: float,
    soc_norm: Optional[Tuple[float, float]],
    soh_norm: Optional[Tuple[float, float]],
    normalize_soc: bool,
    zscore_normalize: bool,
    label_encoder: Optional[LabelEncoder] = None,
    n_mc: int = 16,
    use_pred_soc_for_soh: bool = False,
) -> dict:
    """
    Raw-1D evaluator with explicit n_mc control.

    It keeps the same output metric keys as the shared evaluator, but allows
    final stable evaluation with n_mc=500 without changing training settings.
    """
    model.eval()

    total_loss = 0.0
    total_n = 0

    y_true_all = []
    y_pred_all = []
    soc_true_all = []
    soc_pred_all = []
    soh_true_all = []
    soh_pred_all = []

    for x_img, x_pt, y_cls, soc_tf, soh_tf in loader:
        x_img = x_img.to(device)
        x_pt = x_pt.to(device)
        y_cls = y_cls.to(device)
        soc_tf = soc_tf.to(device).view(-1)
        soh_tf = soh_tf.to(device).view(-1)

        # Keep the original evaluator behavior by default. Only the final test
        # can opt into true hierarchical inference: predicted SOC -> SOH.
        model_soc_tf = None if use_pred_soc_for_soh else soc_tf

        logits_mat, soc_pred, soc_logp, cond_soc, soh_pred, cond_soh = model(
            x_img=x_img,
            x_pt=x_pt,
            soc_tf=model_soc_tf,
            n_mc=int(n_mc),
        )

        # When final-test inference uses predicted SOC for the SOH condition,
        # compute SOC log-probability separately so the reported evaluation
        # loss keeps the original SOC-NLL term.
        if use_pred_soc_for_soh:
            soc_logp = model.soc_flow.log_prob(soc_tf, cond_soc)

        loss = torch.zeros((), dtype=torch.float32, device=device)

        if float(w_cls) > 0:
            loss = loss + float(w_cls) * criterion_cls(logits_mat, y_cls)

        if float(w_soc) > 0:
            soc_reg = criterion_reg(soc_pred.view(-1), soc_tf.view(-1)).mean()
            if soc_logp is not None:
                soc_nll = -soc_logp.mean()
                soc_reg = soc_reg + float(soc_nll_weight) * soc_nll
            loss = loss + float(w_soc) * soc_reg

        if float(w_soh) > 0:
            soh_reg = criterion_reg(soh_pred.view(-1), soh_tf.view(-1)).mean()
            loss = loss + float(w_soh) * soh_reg

        bs = int(y_cls.shape[0])
        total_loss += float(loss.detach().cpu()) * bs
        total_n += bs

        y_true_all.append(y_cls.detach().cpu().numpy())
        y_pred_all.append(torch.argmax(logits_mat, dim=1).detach().cpu().numpy())
        soc_true_all.append(soc_tf.detach().cpu().numpy())
        soc_pred_all.append(soc_pred.detach().cpu().numpy())
        soh_true_all.append(soh_tf.detach().cpu().numpy())
        soh_pred_all.append(soh_pred.detach().cpu().numpy())

    if total_n <= 0:
        raise RuntimeError("Empty dataloader in evaluation.")

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    soc_true = np.concatenate(soc_true_all)
    soc_pred = np.concatenate(soc_pred_all)
    soh_true = np.concatenate(soh_true_all)
    soh_pred = np.concatenate(soh_pred_all)

    soc_true_raw, soh_true_raw = _inverse_targets(
        soc_true,
        soh_true,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
    )

    soc_pred_raw, soh_pred_raw = _inverse_targets(
        soc_pred,
        soh_pred,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
    )

    return {
        "loss": float(total_loss / total_n),
        "cls_acc": float(accuracy_score(y_true, y_pred)),
        "material_acc": _material_accuracy(y_true, y_pred, label_encoder),
        "soc_rmse": _safe_rmse(soc_true, soc_pred),
        "soc_mae": _safe_mae(soc_true, soc_pred),
        "soc_medae": _safe_medae(soc_true, soc_pred),
        "soc_mape": _safe_mape(soc_true, soc_pred),
        "soc_medape": _safe_medape(soc_true, soc_pred),
        "soh_rmse": _safe_rmse(soh_true, soh_pred),
        "soh_mae": _safe_mae(soh_true, soh_pred),
        "soh_medae": _safe_medae(soh_true, soh_pred),
        "soh_mape": _safe_mape(soh_true, soh_pred),
        "soh_medape": _safe_medape(soh_true, soh_pred),
        "soc_rmse_raw": _safe_rmse(soc_true_raw, soc_pred_raw),
        "soc_mae_raw": _safe_mae(soc_true_raw, soc_pred_raw),
        "soc_medae_raw": _safe_medae(soc_true_raw, soc_pred_raw),
        "soc_mape_raw": _safe_mape(soc_true_raw, soc_pred_raw),
        "soc_medape_raw": _safe_medape(soc_true_raw, soc_pred_raw),
        "soh_rmse_raw": _safe_rmse(soh_true_raw, soh_pred_raw),
        "soh_mae_raw": _safe_mae(soh_true_raw, soh_pred_raw),
        "soh_medae_raw": _safe_medae(soh_true_raw, soh_pred_raw),
        "soh_mape_raw": _safe_mape(soh_true_raw, soh_pred_raw),
        "soh_medape_raw": _safe_medape(soh_true_raw, soh_pred_raw),
    }


# =============================================================================
# Raw 1D experiment
# =============================================================================

def run_raw_1d_experiment(
    data_root: str | Path,
    pulse_list: List[int],
    exp_dir: str | Path,
    # U window
    u_start: int = 1,
    u_end: int = 41,
    drop_first_class: bool = True,
    # targets
    soc_col: str = "SOC",
    soh_col: str = "SOH",
    # features
    use_pt_as_feature: bool = True,
    # training
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
    max_epochs: int = 100,
    early_stopping: bool = True,
    patience: int = 20,
    resume: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    # model
    width: int = 64,
    blocks: int = 3,
    drop1d: float = 0.0,
    head_dropout: float = 0.2,
    # loss weights
    w_cls: float = 1.0,
    w_soc: float = 1.0,
    w_soh: float = 1.0,
    # split
    test_id_frac: float = 0.2,
    test_id_count: int = 0,
    val_id_frac: Optional[float] = None,
    val_id_count: int = 0,
    # target normalization
    normalize_soc: bool = True,
    zscore_normalize: bool = True,
    # two-stage
    two_stage: bool = True,
    stage1_epochs: int = 100,
    stage2_epochs: int = 50,
    finetune_epochs: int = 20,
    freeze_encoder_stage2: bool = True,
    freeze_mat_soc_stage2: bool = True,
    # prior bin weighting
    use_soc_prior_weighting: bool = True,
    use_soh_prior_weighting: bool = True,
    soc_prior_bins: int = 10,
    soh_prior_bins: int = 10,
    soc_prior_low: float = 0.5,
    soc_prior_mid: float = 1.0,
    soc_prior_high: float = 0.8,
    soh_prior_low: float = 0.8,
    soh_prior_mid: float = 1.0,
    soh_prior_high: float = 0.9,
    # scoring
    alpha_score: float = 0.1,
    final_best_stage: str = "finetune",
    # final stable evaluation
    final_eval_n_mc_soc: int = 500,
    final_eval_n_mc_soh: int = 500,
    summary_only: bool = False,
) -> dict:
    """
    Train and evaluate raw 1D U1-U41 representation.
    """
    start_time = time.time()

    data_root = Path(data_root)
    exp_dir = Path(exp_dir)

    if summary_only:
        cfg_path = exp_dir / "run_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing run_config.json for summary-only evaluation: {cfg_path}")
        cfg = _load_json(cfg_path)
        pulse_list = list(map(int, cfg.get("pulse_list", pulse_list)))
        u_start = int(cfg.get("u_start", u_start))
        u_end = int(cfg.get("u_end", u_end))
        drop_first_class = bool(cfg.get("drop_first_class", drop_first_class))
        soc_col = str(cfg.get("soc_col", soc_col))
        soh_col = str(cfg.get("soh_col", soh_col))
        use_pt_as_feature = bool(cfg.get("use_pt_as_feature", use_pt_as_feature))
        batch_size = int(cfg.get("batch_size", batch_size))
        seed = int(cfg.get("seed", seed))
        width = int(cfg.get("width", width))
        blocks = int(cfg.get("blocks", blocks))
        drop1d = float(cfg.get("drop1d", drop1d))
        head_dropout = float(cfg.get("head_dropout", head_dropout))
        w_cls = float(cfg.get("w_cls", w_cls))
        w_soc = float(cfg.get("w_soc", w_soc))
        w_soh = float(cfg.get("w_soh", w_soh))
        test_id_frac = float(cfg.get("test_id_frac", test_id_frac))
        test_id_count = int(cfg.get("test_id_count", test_id_count))
        val_id_frac = cfg.get("val_id_frac", val_id_frac)
        val_id_count = int(cfg.get("val_id_count", val_id_count))
        normalize_soc = bool(cfg.get("normalize_soc", normalize_soc))
        zscore_normalize = bool(cfg.get("zscore_normalize", zscore_normalize))
        two_stage = bool(cfg.get("two_stage", two_stage))
        final_best_stage = str(cfg.get("final_best_stage", final_best_stage))
        final_eval_n_mc_soc = int(cfg.get("final_eval_n_mc_soc", final_eval_n_mc_soc))
        final_eval_n_mc_soh = int(cfg.get("final_eval_n_mc_soh", final_eval_n_mc_soh))
        print(f"[SUMMARY-ONLY] Loaded experiment config: {cfg_path}")

    cache_dir = exp_dir / "cache"
    ckpt_dir = exp_dir / "checkpoints"
    logs_dir = exp_dir / "logs"
    splits_dir = exp_dir / "splits"
    metrics_dir = exp_dir / "metrics"

    ensure_dir(
        str(exp_dir),
        str(cache_dir),
        str(ckpt_dir),
        str(logs_dir),
        str(splits_dir),
        str(metrics_dir),
    )

    set_random_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print("[INFO] Input representation: raw_1d")

    run_config = {
        "input_representation": "raw_1d",
        "data_root": str(data_root),
        "pulse_list": list(map(int, pulse_list)),
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
        "drop1d": drop1d,
        "head_dropout": head_dropout,
        "w_cls": w_cls,
        "w_soc": w_soc,
        "w_soh": w_soh,
        "test_id_frac": test_id_frac,
        "test_id_count": test_id_count,
        "val_id_frac": val_id_frac,
        "val_id_count": val_id_count,
        "normalize_soc": normalize_soc,
        "zscore_normalize": zscore_normalize,
        "two_stage": two_stage,
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "finetune_epochs": finetune_epochs,
        "freeze_encoder_stage2": freeze_encoder_stage2,
        "freeze_mat_soc_stage2": freeze_mat_soc_stage2,
        "use_soc_prior_weighting": use_soc_prior_weighting,
        "use_soh_prior_weighting": use_soh_prior_weighting,
        "soc_prior_bins": soc_prior_bins,
        "soh_prior_bins": soh_prior_bins,
        "alpha_score": alpha_score,
        "final_best_stage": final_best_stage,
        "final_eval_n_mc_soc": int(final_eval_n_mc_soc),
        "final_eval_n_mc_soh": int(final_eval_n_mc_soh),
        "exp_dir": str(exp_dir),
    }

    if not summary_only:
        _save_json(exp_dir / "run_config.json", run_config)

    # =========================================================================
    # 1. Load raw train/test data
    # =========================================================================

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

    if Xtr_raw.shape[1] != 41 or Xte_raw.shape[1] != 41:
        raise ValueError(
            f"Expected X dimension = 41 for U1-U41. "
            f"Got train={Xtr_raw.shape}, test={Xte_raw.shape}."
        )

    Xtr_raw, ytr_raw, mtr_raw = drop_nan_inf_rows(
        Xtr_raw,
        ytr_raw,
        mtr_raw,
        name="RAW_TRAIN",
    )

    Xte_raw, yte_raw, mte_raw = drop_nan_inf_rows(
        Xte_raw,
        yte_raw,
        mte_raw,
        name="RAW_TEST",
    )

    if len(ytr_raw) == 0 or len(yte_raw) == 0:
        raise RuntimeError("Empty raw train or raw test after loading/cleaning.")

    if "ID" not in mtr_raw.columns or "ID" not in mte_raw.columns:
        raise RuntimeError("Metadata must contain an 'ID' column for group split.")

    for col in (soc_col, soh_col):
        if col not in mtr_raw.columns or col not in mte_raw.columns:
            raise RuntimeError(f"Metadata must contain column '{col}'.")

    # =========================================================================
    # 2. ID-level train / val / test split
    # =========================================================================

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
    test_ids = np.asarray(test_ids).astype(str)

    if test_id_count and test_id_count > 0:
        split_name = f"testIDs_seed{seed}_n{test_id_count}"
    else:
        split_name = f"testIDs_seed{seed}_frac{test_id_frac}"

    with open(splits_dir / f"{split_name}.txt", "w", encoding="utf-8") as f:
        for test_id in test_ids:
            f.write(str(test_id) + "\n")

    train_candidate_ids = (
        pd.Series(mtr_raw["ID"].astype(str).unique())
        .loc[lambda s: ~s.isin(set(test_ids))]
        .to_numpy()
    )

    if len(train_candidate_ids) == 0:
        raise RuntimeError("No train candidate IDs remain after test split.")

    if val_id_frac is None and not (val_id_count and val_id_count > 0):
        val_id_frac_eff = 0.1
    elif val_id_frac is None:
        val_id_frac_eff = 0.0
    else:
        val_id_frac_eff = float(val_id_frac)

    val_ids = pick_test_ids(
        all_ids=train_candidate_ids,
        test_id_frac=val_id_frac_eff,
        test_id_count=val_id_count,
        seed=seed + 1,
    )
    val_ids = np.asarray(val_ids).astype(str)

    if len(val_ids) == 0:
        raise RuntimeError(
            "Empty validation ID set. Please increase val_id_frac or val_id_count."
        )

    if val_id_count and val_id_count > 0:
        val_split_name = f"valIDs_seed{seed + 1}_n{val_id_count}"
    else:
        val_split_name = f"valIDs_seed{seed + 1}_frac{val_id_frac_eff}"

    with open(splits_dir / f"{val_split_name}.txt", "w", encoding="utf-8") as f:
        for val_id in val_ids:
            f.write(str(val_id) + "\n")

    test_id_set = set(map(str, test_ids))
    val_id_set = set(map(str, val_ids))

    train_mask = (
        ~mtr_raw["ID"].astype(str).isin(test_id_set)
        & ~mtr_raw["ID"].astype(str).isin(val_id_set)
    ).to_numpy()

    val_mask = mte_raw["ID"].astype(str).isin(val_id_set).to_numpy()
    test_mask = mte_raw["ID"].astype(str).isin(test_id_set).to_numpy()

    Xtr = Xtr_raw[train_mask]
    ytr_str = ytr_raw[train_mask]
    mtr = mtr_raw.loc[train_mask].reset_index(drop=True)

    Xval = Xte_raw[val_mask]
    yval_str = yte_raw[val_mask]
    mval = mte_raw.loc[val_mask].reset_index(drop=True)

    Xte = Xte_raw[test_mask]
    yte_str = yte_raw[test_mask]
    mte = mte_raw.loc[test_mask].reset_index(drop=True)

    if len(ytr_str) == 0 or len(yval_str) == 0 or len(yte_str) == 0:
        raise RuntimeError(
            "Empty train, val or test data after applying ID split. "
            f"n_train={len(ytr_str)}, n_val={len(yval_str)}, n_test={len(yte_str)}"
        )

    print(
        f"[DATA] Final TRAIN samples = {len(ytr_str)} | "
        f"unique IDs = {mtr['ID'].astype(str).nunique()}"
    )
    print(
        f"[DATA] Final VAL   samples = {len(yval_str)} | "
        f"unique IDs = {mval['ID'].astype(str).nunique()} "
        f"(from TEST_RANDOM, IDs drawn from training IDs)"
    )
    print(
        f"[DATA] Final TEST  samples = {len(yte_str)} | "
        f"unique IDs = {mte['ID'].astype(str).nunique()}"
    )

    # =========================================================================
    # 3. Train-only normalization of U1-U41
    # =========================================================================

    u_mean = Xtr.mean(axis=0, keepdims=True)
    u_std = Xtr.std(axis=0, keepdims=True) + 1e-8

    Xtr = (Xtr - u_mean) / u_std
    Xval = (Xval - u_mean) / u_std
    Xte = (Xte - u_mean) / u_std

    np.savez_compressed(
        exp_dir / "u41_norm_train_only.npz",
        u_mean=u_mean.astype(np.float32),
        u_std=u_std.astype(np.float32),
    )

    print("[NORM] Applied U1-U41 z-score using TRAIN statistics only.")

    # =========================================================================
    # 4. Target normalization
    # =========================================================================

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

    # =========================================================================
    # 5. Label encoding
    # =========================================================================

    label_encoder = LabelEncoder()
    ytr_cls = label_encoder.fit_transform(ytr_str)

    train_classes = set(label_encoder.classes_.tolist())

    mask_val_known = np.array(
        [label in train_classes for label in yval_str],
        dtype=bool,
    )

    if not mask_val_known.all():
        n_removed = int((~mask_val_known).sum())
        print(
            f"[WARN] Removing {n_removed} val samples with labels unseen in training."
        )

        Xval = Xval[mask_val_known]
        yval_str = yval_str[mask_val_known]
        mval = mval.loc[mask_val_known].reset_index(drop=True)

    mask_test_known = np.array(
        [label in train_classes for label in yte_str],
        dtype=bool,
    )

    if not mask_test_known.all():
        n_removed = int((~mask_test_known).sum())
        print(
            f"[WARN] Removing {n_removed} test samples with labels unseen in training."
        )

        Xte = Xte[mask_test_known]
        yte_str = yte_str[mask_test_known]
        mte = mte.loc[mask_test_known].reset_index(drop=True)

    if len(yval_str) == 0 or len(yte_str) == 0:
        raise RuntimeError(
            "No val or test samples remain after filtering unknown labels."
        )

    yval_cls = label_encoder.transform(yval_str)
    yte_cls = label_encoder.transform(yte_str)

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    _save_json(
        exp_dir / "label_mapping.json",
        {
            "classes": class_names,
            "split_name": split_name,
            "val_split_name": val_split_name,
            "input_representation": "raw_1d",
        },
    )

    print(f"[LABEL] Number of material-capacity classes: {num_classes}")
    print(f"[LABEL] Classes: {class_names}")

    # =========================================================================
    # 6. Pulse-width feature normalization
    # =========================================================================

    if use_pt_as_feature and "pulse_ms" in mtr.columns:
        pt_train_ms = mtr["pulse_ms"].astype(float).to_numpy(dtype=np.float32)
        pt_log = np.log1p(pt_train_ms)

        pt_norm = (
            float(pt_log.mean()),
            float(pt_log.std() + 1e-8),
        )
    else:
        pt_norm = (0.0, 1.0)

    # =========================================================================
    # 7. Dataset and dataloader
    # =========================================================================

    ds_tr = Raw1DDataset(
        X_u=Xtr,
        y_cls=ytr_cls,
        meta=mtr,
        soc_col=soc_col,
        soh_col=soh_col,
        use_pt_as_feature=use_pt_as_feature,
        pt_norm=pt_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
        soc_norm=soc_norm if zscore_normalize else None,
        soh_norm=soh_norm if zscore_normalize else None,
    )

    ds_val = Raw1DDataset(
        X_u=Xval,
        y_cls=yval_cls,
        meta=mval,
        soc_col=soc_col,
        soh_col=soh_col,
        use_pt_as_feature=use_pt_as_feature,
        pt_norm=pt_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
        soc_norm=soc_norm if zscore_normalize else None,
        soh_norm=soh_norm if zscore_normalize else None,
    )

    ds_te = Raw1DDataset(
        X_u=Xte,
        y_cls=yte_cls,
        meta=mte,
        soc_col=soc_col,
        soh_col=soh_col,
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

    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
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

    # =========================================================================
    # 8. Model
    # =========================================================================

    model = Raw1DHierModel(
        num_classes=num_classes,
        input_dim=41,
        width=width,
        blocks=blocks,
        drop1d=drop1d,
        use_pt_as_feature=use_pt_as_feature,
        head_dropout=head_dropout,
    ).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss(beta=1.0, reduction="none")
    soc_nll_weight = 1.0

    # =========================================================================
    # 9. Prior bin weighting
    # =========================================================================

    soc_bin_edges = None
    soc_bin_weights = None

    if use_soc_prior_weighting:
        soc_train_raw = mtr[soc_col].astype(float).to_numpy(dtype=np.float64)

        qs = np.quantile(
            soc_train_raw,
            np.linspace(0, 1, int(soc_prior_bins) + 1),
        )

        soc_bin_edges = [
            (float(qs[i]), float(qs[i + 1]))
            for i in range(len(qs) - 1)
        ]

        soc_bin_weights = np.full(
            (len(soc_bin_edges),),
            float(soc_prior_mid),
            dtype=np.float32,
        )

        if len(soc_bin_weights) >= 1:
            soc_bin_weights[0] = float(soc_prior_low)
            soc_bin_weights[-1] = float(soc_prior_high)

        soc_bin_weights = soc_bin_weights / float(np.mean(soc_bin_weights))

        pd.DataFrame(
            [
                {
                    "lo": lo,
                    "hi": hi,
                    "weight": float(weight),
                }
                for (lo, hi), weight in zip(soc_bin_edges, soc_bin_weights)
            ]
        ).to_csv(
            metrics_dir / "soc_bin_weights_prior.csv",
            index=False,
            encoding="utf-8-sig",
        )

    soh_bin_edges = None
    soh_bin_weights = None

    if use_soh_prior_weighting:
        soh_train_raw = mtr[soh_col].astype(float).to_numpy(dtype=np.float64)

        qs = np.quantile(
            soh_train_raw,
            np.linspace(0, 1, int(soh_prior_bins) + 1),
        )

        soh_bin_edges = [
            (float(qs[i]), float(qs[i + 1]))
            for i in range(len(qs) - 1)
        ]

        soh_bin_weights = np.full(
            (len(soh_bin_edges),),
            float(soh_prior_mid),
            dtype=np.float32,
        )

        if len(soh_bin_weights) >= 1:
            soh_bin_weights[0] = float(soh_prior_low)
            soh_bin_weights[-1] = float(soh_prior_high)

        soh_bin_weights = soh_bin_weights / float(np.mean(soh_bin_weights))

        pd.DataFrame(
            [
                {
                    "lo": lo,
                    "hi": hi,
                    "weight": float(weight),
                }
                for (lo, hi), weight in zip(soh_bin_edges, soh_bin_weights)
            ]
        ).to_csv(
            metrics_dir / "soh_bin_weights_prior.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # =========================================================================
    # 10. Stage helpers
    # =========================================================================

    def set_trainable(stage: str) -> None:
        if stage in {"stage1_soc", "finetune", "single"}:
            for param in model.parameters():
                param.requires_grad = True
            return

        if stage == "stage2_soh":
            for param in model.parameters():
                param.requires_grad = True

            if freeze_encoder_stage2:
                for param in model.encoder.parameters():
                    param.requires_grad = False

            if freeze_mat_soc_stage2:
                for param in model.head_mat.parameters():
                    param.requires_grad = False

                for param in model.soc_flow.parameters():
                    param.requires_grad = False

            for param in model.soh_flow.parameters():
                param.requires_grad = True

            return

        raise ValueError(f"Unknown training stage: {stage}")

    def make_optimizer():
        trainable_params = [
            param for param in model.parameters()
            if param.requires_grad
        ]

        if not trainable_params:
            raise RuntimeError("No trainable parameters found.")

        return torch.optim.AdamW(
            trainable_params,
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

    def stage_paths(stage: str):
        stage_ckpt_dir = ckpt_dir / stage
        ensure_dir(str(stage_ckpt_dir))

        last_path = stage_ckpt_dir / "last.pt"
        best_path = stage_ckpt_dir / "best.pt"
        log_path = logs_dir / f"train_log_{stage}.csv"

        return last_path, best_path, log_path

    def run_stage(
        stage: str,
        epochs: int,
        w_cls_s: float,
        w_soc_s: float,
        w_soh_s: float,
    ) -> Path:
        last_path, best_path, log_path = stage_paths(stage)

        set_trainable(stage)
        optimizer = make_optimizer()

        start_epoch = 0
        best_score = -1e9
        best_epoch = -1
        bad_count = 0

        if resume and last_path.exists():
            ckpt = _torch_load(last_path, map_location=device)

            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optim"])

            start_epoch = int(ckpt.get("epoch", 0) + 1)
            best_score = float(ckpt.get("best_score", -1e9))
            best_epoch = int(ckpt.get("best_epoch", -1))
            bad_count = int(ckpt.get("bad_count", 0))

            print(
                f"[RESUME-{stage}] start_epoch={start_epoch}, "
                f"best_score={best_score:.6f}, best_epoch={best_epoch}"
            )

        for epoch in range(start_epoch, int(epochs)):
            tr = train_one_epoch(
                model=model,
                loader=dl_tr,
                optimizer=optimizer,
                device=device,
                w_cls=w_cls_s,
                w_soc=w_soc_s,
                w_soh=w_soh_s,
                grad_clip=grad_clip,
                criterion_cls=criterion_cls,
                criterion_reg=criterion_reg,
                soc_nll_weight=soc_nll_weight,
                soc_bin_edges=(
                    soc_bin_edges
                    if use_soc_prior_weighting and w_soc_s > 0
                    else None
                ),
                soc_bin_weights=(
                    soc_bin_weights
                    if use_soc_prior_weighting and w_soc_s > 0
                    else None
                ),
                soc_norm=soc_norm if zscore_normalize else None,
                normalize_soc=normalize_soc,
                zscore_normalize=zscore_normalize,
                soh_bin_edges=(
                    soh_bin_edges
                    if use_soh_prior_weighting and w_soh_s > 0
                    else None
                ),
                soh_bin_weights=(
                    soh_bin_weights
                    if use_soh_prior_weighting and w_soh_s > 0
                    else None
                ),
                soh_norm=soh_norm if zscore_normalize else None,
            )

            val = eval_one_epoch_raw_1d(
                model=model,
                loader=dl_val,
                device=device,
                w_cls=w_cls_s,
                w_soc=w_soc_s,
                w_soh=w_soh_s,
                criterion_cls=criterion_cls,
                criterion_reg=criterion_reg,
                soc_nll_weight=soc_nll_weight,
                soc_norm=soc_norm if zscore_normalize else None,
                soh_norm=soh_norm if zscore_normalize else None,
                normalize_soc=normalize_soc,
                zscore_normalize=zscore_normalize,
                label_encoder=label_encoder,
                n_mc=16,
            )

            score = _stage_score(stage, val, alpha_score=alpha_score)

            row = pd.DataFrame(
                [
                    {
                        "stage": stage,
                        "epoch": epoch,
                        "train_loss": tr["loss"],
                        "train_cls_acc": tr["cls_acc"],
                        "train_soc_rmse": tr["soc_rmse"],
                        "train_soc_mae": tr["soc_mae"],
                        "train_soc_mape": tr["soc_mape"],
                        "train_soc_medape": tr.get("soc_medape", np.nan),
                        "train_soh_rmse": tr["soh_rmse"],
                        "train_soh_mae": tr["soh_mae"],
                        "train_soh_mape": tr["soh_mape"],
                        "train_soh_medape": tr.get("soh_medape", np.nan),
                        "val_loss": val["loss"],
                        "val_cls_acc": val["cls_acc"],
                        "val_material_acc": val.get("material_acc", np.nan),
                        "val_soc_rmse": val["soc_rmse"],
                        "val_soc_mae": val["soc_mae"],
                        "val_soc_medae": val.get("soc_medae", np.nan),
                        "val_soc_mape": val["soc_mape"],
                        "val_soc_medape": val.get("soc_medape", np.nan),
                        "val_soh_rmse": val["soh_rmse"],
                        "val_soh_mae": val["soh_mae"],
                        "val_soh_medae": val.get("soh_medae", np.nan),
                        "val_soh_mape": val["soh_mape"],
                        "val_soh_medape": val.get("soh_medape", np.nan),
                        "val_soc_rmse_raw": val["soc_rmse_raw"],
                        "val_soc_mae_raw": val["soc_mae_raw"],
                        "val_soc_medae_raw": val.get("soc_medae_raw", np.nan),
                        "val_soc_mape_raw": val["soc_mape_raw"],
                        "val_soc_medape_raw": val.get("soc_medape_raw", np.nan),
                        "val_soh_rmse_raw": val["soh_rmse_raw"],
                        "val_soh_mae_raw": val["soh_mae_raw"],
                        "val_soh_medae_raw": val.get("soh_medae_raw", np.nan),
                        "val_soh_mape_raw": val["soh_mape_raw"],
                        "val_soh_medape_raw": val.get("soh_medape_raw", np.nan),
                        "val_score": score,
                        "best_score_so_far": max(best_score, score),
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

            print(
                f"[{stage}] epoch {epoch:03d} | "
                f"VAL cls={val['cls_acc']:.4f} | "
                f"SOC MedAPE(raw)={val.get('soc_medape_raw', np.nan):.3f}% | "
                f"SOH MedAPE(raw)={val.get('soh_medape_raw', np.nan):.3f}% | "
                f"val_score={score:.6f}"
            )

            improved = score > best_score

            if improved:
                best_score = score
                best_epoch = epoch
                bad_count = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "best_score": best_score,
                        "best_epoch": best_epoch,
                    },
                    best_path,
                )
            else:
                bad_count += 1

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "bad_count": bad_count,
                },
                last_path,
            )

            if early_stopping and bad_count >= patience:
                print(
                    f"[EARLY STOP-{stage}] "
                    f"best_score={best_score:.6f} at epoch={best_epoch}"
                )
                break

        if best_path.exists():
            ckpt = _torch_load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"])

            print(
                f"[{stage}] Loaded BEST checkpoint from "
                f"epoch={ckpt.get('epoch')} | score={ckpt.get('best_score')}"
            )

        return best_path

    # =========================================================================
    # 11. Training or checkpoint-only evaluation
    # =========================================================================

    if summary_only:
        chosen, best_path = _find_existing_best_checkpoint(exp_dir, final_best_stage)
        print(f"[SUMMARY-ONLY] Skipping training and loading: {best_path}")
    else:
        stage_best_paths = {}
        if two_stage:
            stage_best_paths["stage1_soc"] = run_stage(
                stage="stage1_soc", epochs=int(stage1_epochs),
                w_cls_s=float(w_cls), w_soc_s=float(w_soc), w_soh_s=0.0,
            )
            stage_best_paths["stage2_soh"] = run_stage(
                stage="stage2_soh", epochs=int(stage2_epochs),
                w_cls_s=0.0, w_soc_s=0.0, w_soh_s=float(w_soh),
            )
            if finetune_epochs and finetune_epochs > 0:
                stage_best_paths["finetune"] = run_stage(
                    stage="finetune", epochs=int(finetune_epochs),
                    w_cls_s=float(w_cls) * 0.4, w_soc_s=float(w_soc), w_soh_s=float(w_soh),
                )
        else:
            stage_best_paths["single"] = run_stage(
                stage="single", epochs=int(max_epochs),
                w_cls_s=float(w_cls), w_soc_s=float(w_soc), w_soh_s=float(w_soh),
            )

        if not two_stage:
            chosen = "single"
        else:
            chosen = final_best_stage
            if chosen == "finetune" and "finetune" not in stage_best_paths:
                chosen = "stage2_soh" if "stage2_soh" in stage_best_paths else "stage1_soc"
            if chosen not in stage_best_paths:
                chosen = "stage2_soh" if "stage2_soh" in stage_best_paths else "stage1_soc"
        best_path = stage_best_paths[chosen]

    if not best_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {best_path}")

    ckpt = _torch_load(best_path, map_location=device)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    print(
        f"[FINAL] Using BEST checkpoint from stage='{chosen}' | "
        f"epoch={ckpt.get('epoch') if isinstance(ckpt, dict) else 'unknown'} | "
        f"score={ckpt.get('best_score') if isinstance(ckpt, dict) else 'unknown'}"
    )

    if chosen == "stage1_soc":
        w_cls_eval, w_soc_eval, w_soh_eval = float(w_cls), float(w_soc), 0.0
    elif chosen == "stage2_soh":
        w_cls_eval, w_soc_eval, w_soh_eval = 0.0, 0.0, float(w_soh)
    else:
        w_cls_eval, w_soc_eval, w_soh_eval = float(w_cls), float(w_soc), float(w_soh)

    # =========================================================================
    # 13. Final evaluation
    # =========================================================================

    final_n_mc = max(int(final_eval_n_mc_soc), int(final_eval_n_mc_soh))

    print(
        f"[FINAL] Stable test evaluation with "
        f"n_mc_soc={int(final_eval_n_mc_soc)}, "
        f"n_mc_soh={int(final_eval_n_mc_soh)} | "
        f"scheme=predicted_SOC_to_SOH"
    )

    te = eval_one_epoch_raw_1d(
        model=model,
        loader=dl_te,
        device=device,
        w_cls=w_cls_eval,
        w_soc=w_soc_eval,
        w_soh=w_soh_eval,
        criterion_cls=criterion_cls,
        criterion_reg=criterion_reg,
        soc_nll_weight=1.0,
        soc_norm=soc_norm if zscore_normalize else None,
        soh_norm=soh_norm if zscore_normalize else None,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
        label_encoder=label_encoder,
        n_mc=final_n_mc,
        use_pred_soc_for_soh=True,
    )

    elapsed_sec = time.time() - start_time

    out = {
        "input_representation": "raw_1d",
        "final_stage": chosen,
        "final_eval_scheme": "predicted_soc_to_soh",
        "final_eval_n_mc_used": int(final_n_mc),
        "evaluation_mode": "summary_only" if summary_only else "train_then_evaluate",
        "checkpoint_path": str(best_path),
        "test_cls_acc": float(te["cls_acc"]),
        "test_material_acc": float(te.get("material_acc", np.nan)),
        "test_soc_rmse": float(te["soc_rmse"]),
        "test_soc_mae": float(te["soc_mae"]),
        "test_soc_medae": float(te.get("soc_medae", np.nan)),
        "test_soc_mape": float(te["soc_mape"]),
        "test_soc_medape": float(te.get("soc_medape", np.nan)),
        "test_soh_rmse": float(te["soh_rmse"]),
        "test_soh_mae": float(te["soh_mae"]),
        "test_soh_medae": float(te.get("soh_medae", np.nan)),
        "test_soh_mape": float(te["soh_mape"]),
        "test_soh_medape": float(te.get("soh_medape", np.nan)),
        "test_soc_rmse_raw": float(te["soc_rmse_raw"]),
        "test_soc_mae_raw": float(te["soc_mae_raw"]),
        "test_soc_medae_raw": float(te.get("soc_medae_raw", np.nan)),
        "test_soc_mape_raw": float(te["soc_mape_raw"]),
        "test_soc_medape_raw": float(te.get("soc_medape_raw", np.nan)),
        "test_soh_rmse_raw": float(te["soh_rmse_raw"]),
        "test_soh_mae_raw": float(te["soh_mae_raw"]),
        "test_soh_medae_raw": float(te.get("soh_medae_raw", np.nan)),
        "test_soh_mape_raw": float(te["soh_mape_raw"]),
        "test_soh_medape_raw": float(te.get("soh_medape_raw", np.nan)),
        "n_train": int(len(ds_tr)),
        "n_test": int(len(ds_te)),
        "num_classes": int(num_classes),
        "device": device,
        "elapsed_sec": float(elapsed_sec),
    }

    pd.DataFrame([out]).to_csv(
        metrics_dir / "final_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _save_json(metrics_dir / "final_metrics.json", out)

    print("\n[FINAL METRICS]")
    for key, value in out.items():
        print(f"{key}: {value}")

    return out


# =============================================================================
# Input-representation ablation runner
# =============================================================================

def _add_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    summary = summary.copy()
    summary["fine_acc_pct"] = summary["test_cls_acc"].astype(float) * 100.0
    summary["material_acc_pct"] = summary["test_material_acc"].astype(float) * 100.0
    summary["soc_medae_pct"] = summary["test_soc_medae_raw"].astype(float)
    summary["soh_medae_pct"] = summary["test_soh_medae_raw"].astype(float)
    summary["soc_medape_pct"] = summary["test_soc_medape_raw"].astype(float)
    summary["soh_medape_pct"] = summary["test_soh_medape_raw"].astype(float)

    if "structured_3ch" in set(summary["input_representation"]):
        ref = summary.loc[summary["input_representation"] == "structured_3ch"].iloc[0]
        summary["fine_acc_change_pp_vs_structured"] = (summary["test_cls_acc"].astype(float) - float(ref["test_cls_acc"])) * 100.0
        summary["material_acc_change_pp_vs_structured"] = (summary["test_material_acc"].astype(float) - float(ref["test_material_acc"])) * 100.0
        summary["soc_medae_change_pp_vs_structured"] = summary["test_soc_medae_raw"].astype(float) - float(ref["test_soc_medae_raw"])
        summary["soh_medae_change_pp_vs_structured"] = summary["test_soh_medae_raw"].astype(float) - float(ref["test_soh_medae_raw"])
        summary["soc_medape_change_pp_vs_structured"] = summary["test_soc_medape_raw"].astype(float) - float(ref["test_soc_medape_raw"])
        summary["soh_medape_change_pp_vs_structured"] = summary["test_soh_medape_raw"].astype(float) - float(ref["test_soh_medape_raw"])

    return summary

def _summary_metrics_from_predictions(pred_path: Path) -> dict:
    df = pd.read_csv(pred_path)
    required = ["true_label", "pred_label", "soc_true", "soc_pred", "soh_true", "soh_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Prediction table is missing required columns: {missing}")

    soc_true = df["soc_true"].to_numpy(dtype=np.float64)
    soc_pred = df["soc_pred"].to_numpy(dtype=np.float64)
    soh_true = df["soh_true"].to_numpy(dtype=np.float64)
    soh_pred = df["soh_pred"].to_numpy(dtype=np.float64)

    if np.nanmedian(np.abs(soh_true)) <= 2.0:
        soh_true *= 100.0
        soh_pred *= 100.0

    true_material = df["true_label"].astype(str).str.split("_").str[0]
    pred_material = df["pred_label"].astype(str).str.split("_").str[0]

    return {
        "n": int(len(df)),
        "cls_acc": float((df["true_label"].astype(str) == df["pred_label"].astype(str)).mean()),
        "material_acc": float((true_material == pred_material).mean()),
        "soc_rmse": _safe_rmse(soc_true, soc_pred),
        "soc_mae": _safe_mae(soc_true, soc_pred),
        "soc_medae": _safe_medae(soc_true, soc_pred),
        "soc_mape": _safe_mape(soc_true, soc_pred),
        "soc_medape": _safe_medape(soc_true, soc_pred),
        "soh_rmse": _safe_rmse(soh_true, soh_pred),
        "soh_mae": _safe_mae(soh_true, soh_pred),
        "soh_medae": _safe_medae(soh_true, soh_pred),
        "soh_mape": _safe_mape(soh_true, soh_pred),
        "soh_medape": _safe_medape(soh_true, soh_pred),
    }


def load_structured_3ch_from_further_analysis(
    proposed_summary_path: str | Path,
) -> dict:
    proposed_summary_path = Path(proposed_summary_path)
    pred_path = proposed_summary_path.parent / "test_predictions_per_sample.csv"

    if pred_path.exists():
        row = _summary_metrics_from_predictions(pred_path)
        source = str(pred_path)
        source_kind = "per_sample_predictions"
    else:
        if not proposed_summary_path.exists():
            raise FileNotFoundError(
                f"Cannot find proposed summary or prediction table: {proposed_summary_path}"
            )

        df = pd.read_csv(proposed_summary_path)
        if df.empty or "split" not in df.columns:
            raise RuntimeError(f"Invalid proposed summary: {proposed_summary_path}")

        test_df = df.loc[df["split"].astype(str).str.lower() == "test"]
        if test_df.empty:
            raise RuntimeError("Proposed summary does not contain split == 'test'.")

        row = test_df.iloc[0].to_dict()
        required = ["n", "cls_acc", "soc_rmse", "soc_mae", "soc_mape", "soc_medape", "soh_rmse", "soh_mae", "soh_mape", "soh_medape"]
        missing = [c for c in required if c not in row]
        if missing:
            raise RuntimeError(f"Proposed summary is missing required columns: {missing}")

        row["material_acc"] = float(row.get("material_acc", np.nan))
        row["soc_medae"] = float(row.get("soc_medae", row.get("soc_ae_p50", np.nan)))
        row["soh_medae"] = float(row.get("soh_medae", row.get("soh_ae_p50", np.nan)))

        # Old summaries may store SOH absolute errors in ratio units.
        if np.isfinite(float(row["soh_mae"])) and float(row["soh_mae"]) < 1.0:
            for key in ("soh_rmse", "soh_mae", "soh_medae"):
                row[key] = float(row[key]) * 100.0

        source = str(proposed_summary_path)
        source_kind = "summary_csv"

    out = {
        "final_stage": "proposed_further_analysis",
        "final_eval_scheme": "predicted_soc_to_soh",
        "evaluation_mode": "loaded_existing_results",
        "checkpoint_path": "",
        "test_cls_acc": float(row["cls_acc"]),
        "test_material_acc": float(row["material_acc"]),
        "test_soc_rmse": float(row["soc_rmse"]),
        "test_soc_mae": float(row["soc_mae"]),
        "test_soc_medae": float(row["soc_medae"]),
        "test_soc_mape": float(row["soc_mape"]),
        "test_soc_medape": float(row["soc_medape"]),
        "test_soh_rmse": float(row["soh_rmse"]),
        "test_soh_mae": float(row["soh_mae"]),
        "test_soh_medae": float(row["soh_medae"]),
        "test_soh_mape": float(row["soh_mape"]),
        "test_soh_medape": float(row["soh_medape"]),
        "test_soc_rmse_raw": float(row["soc_rmse"]),
        "test_soc_mae_raw": float(row["soc_mae"]),
        "test_soc_medae_raw": float(row["soc_medae"]),
        "test_soc_mape_raw": float(row["soc_mape"]),
        "test_soc_medape_raw": float(row["soc_medape"]),
        "test_soh_rmse_raw": float(row["soh_rmse"]),
        "test_soh_mae_raw": float(row["soh_mae"]),
        "test_soh_medae_raw": float(row["soh_medae"]),
        "test_soh_mape_raw": float(row["soh_mape"]),
        "test_soh_medape_raw": float(row["soh_medape"]),
        "n_train": np.nan,
        "n_val": np.nan,
        "n_test": int(row["n"]),
        "num_classes": np.nan,
        "device": f"loaded_from_{source_kind}",
        "elapsed_sec": 0.0,
        "source_summary": source,
        "source_split": "test",
    }

    print(f"[LOAD] structured_3ch loaded from: {source}")
    return out

def run_input_representation_ablation(
    data_root: str | Path,
    output_root: str | Path,
    smoke: bool = False,
    resume: bool = True,
    config: str = "both",
    proposed_summary_path: str | Path | None = None,
    summary_only: bool = False,
) -> pd.DataFrame:
    """
    Run raw 1D vs structured 3-channel input representation ablation.

    raw_1d:
        U1-U41 -> 1D CNN encoder.

    structured_3ch:
        Directly loaded from proposed-method further-analysis summary.
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = str(config).lower()
    if config not in {"raw_1d", "structured_3ch", "both"}:
        raise ValueError("config must be one of {'raw_1d', 'structured_3ch', 'both'}.")

    if proposed_summary_path is None:
        proposed_summary_path = (
            PROJECT_ROOT
            / "results"
            / "proposed_framework"
            / "further_analysis"
            / "tables"
            / "proposed_method_summary.csv"
        )

    if smoke:
        pulse_list = [5000]

        raw_kwargs = {
            "batch_size": 32,
            "max_epochs": 1,
            "early_stopping": False,
            "patience": 1,
            "resume": False,
            "width": 32,
            "blocks": 1,
            "drop1d": 0.0,
            "head_dropout": 0.1,
            "two_stage": False,
            "stage1_epochs": 1,
            "stage2_epochs": 1,
            "finetune_epochs": 0,
            "use_soc_prior_weighting": False,
            "use_soh_prior_weighting": False,
            "final_best_stage": "single",
        }

    else:
        pulse_list = DEFAULT_PULSE_LIST

        raw_kwargs = {
            "batch_size": 128,
            "max_epochs": 400,
            "early_stopping": False,
            "patience": 20,
            "resume": resume,
            "width": 64,
            "blocks": 3,
            "drop1d": 0.0,
            "head_dropout": 0.2,
            "two_stage": True,
            "stage1_epochs": 200,
            "stage2_epochs": 200,
            "finetune_epochs": 30,
            "use_soc_prior_weighting": True,
            "use_soh_prior_weighting": True,
            "final_best_stage": "finetune",
        }

    raw_kwargs.update({
        "final_eval_n_mc_soc": 500,
        "final_eval_n_mc_soh": 500,
    })


    rows = []

    # -------------------------------------------------------------------------
    # 1. Raw 1D representation
    # -------------------------------------------------------------------------

    if config in {"raw_1d", "both"}:
        raw_exp_dir = output_root / "raw_1d"

        print("\n" + "=" * 90)
        print("[RUN] Input representation: raw_1d")
        print(f"[RUN] Pulse list: {pulse_list}")
        print(f"[RUN] Output directory: {raw_exp_dir}")
        print("=" * 90)

        raw_out = run_raw_1d_experiment(
            data_root=data_root,
            pulse_list=pulse_list,
            exp_dir=raw_exp_dir,

            u_start=1,
            u_end=41,
            drop_first_class=True,

            soc_col="SOC",
            soh_col="SOH",

            use_pt_as_feature=True,

            lr=3e-4,
            weight_decay=1e-4,
            grad_clip=5.0,
            num_workers=0,
            seed=42,

            w_cls=1.0,
            w_soc=1.0,
            w_soh=1.0,

            test_id_frac=0.2,
            test_id_count=0,
            val_id_frac=0.1,
            val_id_count=0,

            normalize_soc=True,
            zscore_normalize=True,

            freeze_encoder_stage2=True,
            freeze_mat_soc_stage2=True,

            soc_prior_bins=10,
            soh_prior_bins=10,
            soc_prior_low=0.5,
            soc_prior_mid=1.0,
            soc_prior_high=0.8,
            soh_prior_low=0.8,
            soh_prior_mid=1.0,
            soh_prior_high=0.9,

            alpha_score=0.1,
            summary_only=summary_only,

            **raw_kwargs,
        )

        rows.append(
            {
                "config": "raw_1d",
                "input_representation": "raw_1d",
                "pulse_widths_ms": ",".join(map(str, pulse_list)),
                "num_pulse_widths": len(pulse_list),
                **raw_out,
            }
        )

        partial = pd.DataFrame(rows)
        partial = _add_summary_columns(partial)

        partial.to_csv(
            output_root / "input_representation_ablation_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )

        _save_json(
            output_root / "input_representation_ablation_partial.json",
            rows,
        )

    # -------------------------------------------------------------------------
    # 2. Structured 3-channel representation
    # -------------------------------------------------------------------------

    if config in {"structured_3ch", "both"}:
        print("\n" + "=" * 90)
        print("[LOAD] Input representation: structured_3ch")
        print("[LOAD] Source: proposed further-analysis summary")
        print(f"[LOAD] Path: {proposed_summary_path}")
        print("=" * 90)

        structured_out = load_structured_3ch_from_further_analysis(
            proposed_summary_path=proposed_summary_path,
        )

        structured_row = {**structured_out}
        structured_row.update(
            {
                "config": "structured_3ch",
                "input_representation": "structured_3ch",
                "pulse_widths_ms": ",".join(map(str, pulse_list)),
                "num_pulse_widths": len(pulse_list),
            }
        )
        rows.append(structured_row)

    summary = pd.DataFrame(rows)
    summary = _add_summary_columns(summary)

    summary.to_csv(
        output_root / "input_representation_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _save_json(
        output_root / "input_representation_ablation_summary.json",
        summary.to_dict(orient="records"),
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or re-evaluate input-representation ablation.")
    parser.add_argument("--config", type=str, default="both", choices=["raw_1d", "structured_3ch", "both"])
    parser.add_argument("--summary-only", action="store_true", help="Skip raw-1D training and evaluate the existing best.pt checkpoint only.")
    parser.add_argument("--smoke", action="store_true", help="Run a very small smoke test.")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume when training.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "ablation" / "input_representation")
    parser.add_argument(
        "--proposed-summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "proposed_framework" / "further_analysis" / "tables" / "proposed_method_summary.csv",
        help="Path to proposed_method_summary.csv. A sibling test_predictions_per_sample.csv is preferred automatically.",
    )
    args = parser.parse_args()

    summary = run_input_representation_ablation(
        data_root=args.data_root,
        output_root=args.output_root,
        smoke=args.smoke,
        resume=not args.no_resume,
        config=args.config,
        proposed_summary_path=args.proposed_summary,
        summary_only=args.summary_only,
    )

    print("\n[SUMMARY]")
    print(summary)


if __name__ == "__main__":
    main()
