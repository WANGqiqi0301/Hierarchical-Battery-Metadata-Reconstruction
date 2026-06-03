# ablation/hierarchy_order_ablation.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import sys
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
from utils.metrics import rmse, mae, mape, medape

from proposed_framework.data.build_dataset import (
    build_train_mix_soc_mix_pt,
    build_test_random_mix_pt,
    pick_test_ids,
    apply_id_split,
)
from proposed_framework.data.pulse_dataset import HierPulseDataset
from proposed_framework.models.encoder import MicroResNetEncoder2D3Ch
from proposed_framework.models.conditional_flow import Conditional1DFlow


# =============================================================================
# Constants
# =============================================================================

DEFAULT_PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

ORDER_DESCRIPTIONS: Dict[str, str] = {
    "M_SOC_SOH": "Material -> SOC -> SOH, proposed physical order",
    "M_SOH_SOC": "Material -> SOH -> SOC",
    "SOC_M_SOH": "SOC -> Material -> SOH",
    "SOH_M_SOC": "SOH -> Material -> SOC",
    "PARALLEL": "Material, SOC and SOH predicted in parallel",
}

DEFAULT_ORDER_LIST = [
    "M_SOC_SOH",
    "M_SOH_SOC",
    "SOC_M_SOH",
    "SOH_M_SOC",
    "PARALLEL",
]


# =============================================================================
# Utility functions
# =============================================================================

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
        If normalize_soc=True, raw output is SOC in percentage.

    SOH:
        Raw output is original SOH unit.
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


def sample_flow_mean(
    flow: Conditional1DFlow,
    context: torch.Tensor,
    n_mc: int,
) -> torch.Tensor:
    """
    Compute MC mean from nflows samples robustly.

    nflows may return:
        (num_samples, B, 1)
        (B, num_samples, 1)
        (num_samples * B, 1)
    """
    batch_size = context.size(0)
    samples = flow.sample(context, num_samples=int(n_mc))

    if samples.ndim == 3:
        if samples.shape[0] == int(n_mc) and samples.shape[1] == batch_size:
            return samples.mean(dim=0).squeeze(-1).view(-1)

        if samples.shape[0] == batch_size and samples.shape[1] == int(n_mc):
            return samples.mean(dim=1).squeeze(-1).view(-1)

        samples = samples.reshape(int(n_mc), batch_size, 1)
        return samples.mean(dim=0).squeeze(-1).view(-1)

    if samples.ndim == 2:
        samples = samples.view(int(n_mc), batch_size, 1)
        return samples.mean(dim=0).squeeze(-1).view(-1)

    raise RuntimeError(f"Unexpected flow sample shape: {tuple(samples.shape)}")


def _cat(parts: List[Optional[torch.Tensor]]) -> torch.Tensor:
    valid = [p for p in parts if p is not None]
    return torch.cat(valid, dim=1)


def _save_json(path: str | Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _torch_load(path: str | Path, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# =============================================================================
# Hierarchy-order model
# =============================================================================

class HierOrderAblationModel(nn.Module):
    """
    Flexible model for hierarchy-order ablation.

    Supported orders
    ----------------
    M_SOC_SOH:
        Material -> SOC -> SOH
        M   = f(z, pt)
        SOC = f(z, M, pt)
        SOH = f(z, M, SOC, pt)

    M_SOH_SOC:
        Material -> SOH -> SOC
        M   = f(z, pt)
        SOH = f(z, M, pt)
        SOC = f(z, M, SOH, pt)

    SOC_M_SOH:
        SOC -> Material -> SOH
        SOC = f(z, pt)
        M   = f(z, SOC, pt)
        SOH = f(z, M, SOC, pt)

    SOH_M_SOC:
        SOH -> Material -> SOC
        SOH = f(z, pt)
        M   = f(z, SOH, pt)
        SOC = f(z, M, SOH, pt)

    PARALLEL:
        M   = f(z, pt)
        SOC = f(z, pt)
        SOH = f(z, pt)
    """

    def __init__(
        self,
        num_classes: int,
        order: str = "M_SOC_SOH",
        width: int = 24,
        blocks: int = 2,
        drop2d: float = 0.0,
        use_pt_as_feature: bool = True,
        head_dropout: float = 0.1,
        flow_hidden: int = 48,
        flow_layers: int = 3,
        flow_bins: int = 8,
        flow_tail_bound: float = 3.0,
    ):
        super().__init__()

        if order not in ORDER_DESCRIPTIONS:
            raise ValueError(
                f"Unknown order={order}. Choose from {list(ORDER_DESCRIPTIONS)}."
            )

        self.order = str(order)
        self.num_classes = int(num_classes)
        self.use_pt = bool(use_pt_as_feature)

        pt_dim = 1 if self.use_pt else 0
        p_dim = int(num_classes)

        self.encoder = MicroResNetEncoder2D3Ch(
            width=width,
            blocks=blocks,
            drop2d=drop2d,
        )

        # Material heads
        # Keep material heads consistent with the updated proposed setting:
        # when material is predicted directly from z, pulse width can be included.
        self.head_mat_from_z = nn.Sequential(
            nn.Linear(width + pt_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(64, num_classes),
        )

        self.head_mat_from_z_soc = nn.Sequential(
            nn.Linear(width + 1 + pt_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(64, num_classes),
        )

        self.head_mat_from_z_soh = nn.Sequential(
            nn.Linear(width + 1 + pt_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(64, num_classes),
        )

        # Context dimensions
        ctx_z_pt = width + pt_dim
        ctx_z_p_pt = width + p_dim + pt_dim
        ctx_z_p_soc_pt = width + p_dim + 1 + pt_dim
        ctx_z_p_soh_pt = width + p_dim + 1 + pt_dim

        # SOC flows
        self.soc_flow_z_pt = Conditional1DFlow(
            context_dim=ctx_z_pt,
            hidden_features=int(flow_hidden),
            num_transforms=int(flow_layers),
            num_bins=int(flow_bins),
            tail_bound=float(flow_tail_bound),
        )

        self.soc_flow_z_p_pt = Conditional1DFlow(
            context_dim=ctx_z_p_pt,
            hidden_features=int(flow_hidden),
            num_transforms=int(flow_layers),
            num_bins=int(flow_bins),
            tail_bound=float(flow_tail_bound),
        )

        self.soc_flow_z_p_soh_pt = Conditional1DFlow(
            context_dim=ctx_z_p_soh_pt,
            hidden_features=int(flow_hidden),
            num_transforms=int(flow_layers),
            num_bins=int(flow_bins),
            tail_bound=float(flow_tail_bound),
        )

        # SOH flows
        self.soh_flow_z_pt = Conditional1DFlow(
            context_dim=ctx_z_pt,
            hidden_features=int(flow_hidden),
            num_transforms=int(flow_layers),
            num_bins=int(flow_bins),
            tail_bound=float(flow_tail_bound),
        )

        self.soh_flow_z_p_pt = Conditional1DFlow(
            context_dim=ctx_z_p_pt,
            hidden_features=int(flow_hidden),
            num_transforms=int(flow_layers),
            num_bins=int(flow_bins),
            tail_bound=float(flow_tail_bound),
        )

        self.soh_flow_z_p_soc_pt = Conditional1DFlow(
            context_dim=ctx_z_p_soc_pt,
            hidden_features=int(flow_hidden),
            num_transforms=int(flow_layers),
            num_bins=int(flow_bins),
            tail_bound=float(flow_tail_bound),
        )

    def _pt(self, x_pt: torch.Tensor) -> Optional[torch.Tensor]:
        return x_pt if self.use_pt else None

    def forward(
        self,
        x_img: torch.Tensor,
        x_pt: torch.Tensor,
        soc_tf: Optional[torch.Tensor] = None,
        soh_tf: Optional[torch.Tensor] = None,
        n_mc: int = 8,
    ) -> Dict[str, torch.Tensor]:
        z = self.encoder(x_img)
        pt = self._pt(x_pt)

        soc_true = soc_tf.view(-1) if soc_tf is not None else None
        soh_true = soh_tf.view(-1) if soh_tf is not None else None

        soc_logp = None
        soh_logp = None

        # ---------------------------------------------------------------------
        # 1. Proposed physical order: Material -> SOC -> SOH
        # ---------------------------------------------------------------------
        if self.order == "M_SOC_SOH":
            mat_input = _cat([z, pt])
            logits = self.head_mat_from_z(mat_input)
            p_mat = torch.softmax(logits, dim=1)

            cond_soc = _cat([z, p_mat, pt])

            if soc_true is not None:
                soc_logp = self.soc_flow_z_p_pt.log_prob(soc_true, cond_soc)

            with torch.no_grad():
                soc_pred = sample_flow_mean(
                    self.soc_flow_z_p_pt,
                    cond_soc,
                    n_mc=n_mc,
                )

            soc_for_soh = (
                soc_true.detach().view(-1, 1)
                if soc_true is not None
                else soc_pred.detach().view(-1, 1)
            )

            cond_soh = _cat([z, p_mat, soc_for_soh, pt])

            if soh_true is not None:
                soh_logp = self.soh_flow_z_p_soc_pt.log_prob(soh_true, cond_soh)

            with torch.no_grad():
                soh_pred = sample_flow_mean(
                    self.soh_flow_z_p_soc_pt,
                    cond_soh,
                    n_mc=n_mc,
                )

        # ---------------------------------------------------------------------
        # 2. Material -> SOH -> SOC
        # ---------------------------------------------------------------------
        elif self.order == "M_SOH_SOC":
            mat_input = _cat([z, pt])
            logits = self.head_mat_from_z(mat_input)
            p_mat = torch.softmax(logits, dim=1)

            cond_soh = _cat([z, p_mat, pt])

            if soh_true is not None:
                soh_logp = self.soh_flow_z_p_pt.log_prob(soh_true, cond_soh)

            with torch.no_grad():
                soh_pred = sample_flow_mean(
                    self.soh_flow_z_p_pt,
                    cond_soh,
                    n_mc=n_mc,
                )

            soh_for_soc = (
                soh_true.detach().view(-1, 1)
                if soh_true is not None
                else soh_pred.detach().view(-1, 1)
            )

            cond_soc = _cat([z, p_mat, soh_for_soc, pt])

            if soc_true is not None:
                soc_logp = self.soc_flow_z_p_soh_pt.log_prob(soc_true, cond_soc)

            with torch.no_grad():
                soc_pred = sample_flow_mean(
                    self.soc_flow_z_p_soh_pt,
                    cond_soc,
                    n_mc=n_mc,
                )

        # ---------------------------------------------------------------------
        # 3. SOC -> Material -> SOH
        # ---------------------------------------------------------------------
        elif self.order == "SOC_M_SOH":
            cond_soc = _cat([z, pt])

            if soc_true is not None:
                soc_logp = self.soc_flow_z_pt.log_prob(soc_true, cond_soc)

            with torch.no_grad():
                soc_pred = sample_flow_mean(
                    self.soc_flow_z_pt,
                    cond_soc,
                    n_mc=n_mc,
                )

            soc_for_mat = (
                soc_true.detach().view(-1, 1)
                if soc_true is not None
                else soc_pred.detach().view(-1, 1)
            )

            mat_input = _cat([z, soc_for_mat, pt])
            logits = self.head_mat_from_z_soc(mat_input)
            p_mat = torch.softmax(logits, dim=1)

            cond_soh = _cat([z, p_mat, soc_for_mat, pt])

            if soh_true is not None:
                soh_logp = self.soh_flow_z_p_soc_pt.log_prob(soh_true, cond_soh)

            with torch.no_grad():
                soh_pred = sample_flow_mean(
                    self.soh_flow_z_p_soc_pt,
                    cond_soh,
                    n_mc=n_mc,
                )

        # ---------------------------------------------------------------------
        # 4. SOH -> Material -> SOC
        # ---------------------------------------------------------------------
        elif self.order == "SOH_M_SOC":
            cond_soh = _cat([z, pt])

            if soh_true is not None:
                soh_logp = self.soh_flow_z_pt.log_prob(soh_true, cond_soh)

            with torch.no_grad():
                soh_pred = sample_flow_mean(
                    self.soh_flow_z_pt,
                    cond_soh,
                    n_mc=n_mc,
                )

            soh_for_mat = (
                soh_true.detach().view(-1, 1)
                if soh_true is not None
                else soh_pred.detach().view(-1, 1)
            )

            mat_input = _cat([z, soh_for_mat, pt])
            logits = self.head_mat_from_z_soh(mat_input)
            p_mat = torch.softmax(logits, dim=1)

            cond_soc = _cat([z, p_mat, soh_for_mat, pt])

            if soc_true is not None:
                soc_logp = self.soc_flow_z_p_soh_pt.log_prob(soc_true, cond_soc)

            with torch.no_grad():
                soc_pred = sample_flow_mean(
                    self.soc_flow_z_p_soh_pt,
                    cond_soc,
                    n_mc=n_mc,
                )

        # ---------------------------------------------------------------------
        # 5. Parallel heads
        # ---------------------------------------------------------------------
        elif self.order == "PARALLEL":
            mat_input = _cat([z, pt])
            logits = self.head_mat_from_z(mat_input)
            p_mat = torch.softmax(logits, dim=1)

            cond_soc = _cat([z, pt])
            cond_soh = _cat([z, pt])

            if soc_true is not None:
                soc_logp = self.soc_flow_z_pt.log_prob(soc_true, cond_soc)

            if soh_true is not None:
                soh_logp = self.soh_flow_z_pt.log_prob(soh_true, cond_soh)

            with torch.no_grad():
                soc_pred = sample_flow_mean(
                    self.soc_flow_z_pt,
                    cond_soc,
                    n_mc=n_mc,
                )

                soh_pred = sample_flow_mean(
                    self.soh_flow_z_pt,
                    cond_soh,
                    n_mc=n_mc,
                )

        else:
            raise RuntimeError(f"Unhandled order={self.order}")

        return {
            "logits": logits,
            "soc_pred": soc_pred.view(-1),
            "soh_pred": soh_pred.view(-1),
            "soc_logp": soc_logp,
            "soh_logp": soh_logp,
        }


# =============================================================================
# Train and eval
# =============================================================================

def train_one_epoch_order(
    model: HierOrderAblationModel,
    loader: DataLoader,
    optimizer,
    device: str,
    criterion_cls,
    w_cls: float = 1.0,
    w_soc: float = 1.0,
    w_soh: float = 1.0,
    grad_clip: float = 5.0,
    n_mc: int = 4,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    n = 0

    y_true_cls_all = []
    y_pred_cls_all = []

    for x3, pt, y_cls, soc, soh in loader:
        x3 = x3.to(device)
        pt = pt.to(device)
        y_cls = y_cls.to(device)
        soc = soc.to(device).view(-1)
        soh = soh.to(device).view(-1)

        optimizer.zero_grad(set_to_none=True)

        out = model(
            x_img=x3,
            x_pt=pt,
            soc_tf=soc,
            soh_tf=soh,
            n_mc=n_mc,
        )

        logits = out["logits"]
        soc_logp = out["soc_logp"]
        soh_logp = out["soh_logp"]

        if soc_logp is None:
            raise RuntimeError("soc_logp is None during training.")
        if soh_logp is None:
            raise RuntimeError("soh_logp is None during training.")

        loss_cls = criterion_cls(logits, y_cls)
        loss_soc = (-soc_logp.view(-1)).mean()
        loss_soh = (-soh_logp.view(-1)).mean()

        loss = (
            float(w_cls) * loss_cls
            + float(w_soc) * loss_soc
            + float(w_soh) * loss_soh
        )

        loss.backward()

        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

        optimizer.step()

        batch_size = int(y_cls.size(0))
        total_loss += float(loss.item()) * batch_size
        n += batch_size

        y_true_cls_all.append(y_cls.detach().cpu().numpy())
        y_pred_cls_all.append(logits.detach().cpu().argmax(dim=1).numpy())

    y_true_cls = np.concatenate(y_true_cls_all)
    y_pred_cls = np.concatenate(y_pred_cls_all)

    return {
        "loss": total_loss / max(n, 1),
        "cls_acc": float(accuracy_score(y_true_cls, y_pred_cls)),
    }


@torch.no_grad()
def eval_order(
    model: HierOrderAblationModel,
    loader: DataLoader,
    device: str,
    criterion_cls,
    soc_norm: Optional[Tuple[float, float]],
    soh_norm: Optional[Tuple[float, float]],
    normalize_soc: bool = True,
    zscore_normalize: bool = True,
    n_mc: int = 16,
) -> Dict[str, float]:
    model.eval()

    loss_all = []
    n = 0

    y_true_cls_all = []
    y_pred_cls_all = []

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

        out = model(
            x_img=x3,
            x_pt=pt,
            soc_tf=None,
            soh_tf=None,
            n_mc=n_mc,
        )

        logits = out["logits"]
        soc_pred = out["soc_pred"].view(-1)
        soh_pred = out["soh_pred"].view(-1)

        loss_cls = criterion_cls(logits, y_cls)

        batch_size = int(y_cls.size(0))
        loss_all.append(float(loss_cls.item()) * batch_size)
        n += batch_size

        y_true_cls_all.append(y_cls.detach().cpu().numpy())
        y_pred_cls_all.append(logits.detach().cpu().argmax(dim=1).numpy())

        soc_true_all.append(soc.detach().cpu().numpy())
        soc_pred_all.append(soc_pred.detach().cpu().numpy())

        soh_true_all.append(soh.detach().cpu().numpy())
        soh_pred_all.append(soh_pred.detach().cpu().numpy())

    y_true_cls = np.concatenate(y_true_cls_all)
    y_pred_cls = np.concatenate(y_pred_cls_all)

    soc_true = np.concatenate(soc_true_all)
    soc_pred = np.concatenate(soc_pred_all)

    soh_true = np.concatenate(soh_true_all)
    soh_pred = np.concatenate(soh_pred_all)

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
        "loss": float(np.sum(loss_all) / max(n, 1)),
        "cls_acc": float(accuracy_score(y_true_cls, y_pred_cls)),

        "soc_rmse_raw": rmse(soc_true_raw, soc_pred_raw),
        "soc_mae_raw": mae(soc_true_raw, soc_pred_raw),
        "soc_mape_raw": mape(soc_true_raw, soc_pred_raw),
        "soc_medape_raw": medape(soc_true_raw, soc_pred_raw),

        "soh_rmse_raw": rmse(soh_true_raw, soh_pred_raw),
        "soh_mae_raw": mae(soh_true_raw, soh_pred_raw),
        "soh_mape_raw": mape(soh_true_raw, soh_pred_raw),
        "soh_medape_raw": medape(soh_true_raw, soh_pred_raw),
    }


# =============================================================================
# Data preparation
# =============================================================================

def prepare_hierarchy_order_data(
    data_root: str | Path,
    pulse_list: List[int],
    cache_dir: str | Path,
    splits_dir: str | Path,
    u_start: int = 1,
    u_end: int = 41,
    drop_first_class: bool = True,
    soc_col: str = "SOC",
    soh_col: str = "SOH",
    use_pt_as_feature: bool = True,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
    test_id_frac: float = 0.2,
    test_id_count: int = 0,
    normalize_soc: bool = True,
    zscore_normalize: bool = True,
) -> Dict:
    data_root = Path(data_root)
    cache_dir = Path(cache_dir)
    splits_dir = Path(splits_dir)

    ensure_dir(str(cache_dir), str(splits_dir))

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

    for col in ["ID", soc_col, soh_col]:
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
        split_name = f"hier_order_testIDs_seed{seed}_n{test_id_count}"
    else:
        split_name = f"hier_order_testIDs_seed{seed}_frac{test_id_frac}"

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
        f"[DATA] TRAIN samples = {len(ytr_str)} | "
        f"IDs = {mtr['ID'].astype(str).nunique()}"
    )
    print(
        f"[DATA] TEST  samples = {len(yte_str)} | "
        f"IDs = {mte['ID'].astype(str).nunique()}"
    )

    # U normalization using TRAIN stats only
    u_mean = Xtr.mean(axis=0, keepdims=True)
    u_std = Xtr.std(axis=0, keepdims=True) + 1e-8

    Xtr = (Xtr - u_mean) / u_std
    Xte = (Xte - u_mean) / u_std

    np.savez_compressed(
        Path(cache_dir).parent / "u41_norm_train_only_hierarchy_order.npz",
        u_mean=u_mean.astype(np.float32),
        u_std=u_std.astype(np.float32),
    )

    # Target normalization stats
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

    # Label encoding
    label_encoder = LabelEncoder()
    ytr_cls = label_encoder.fit_transform(ytr_str)

    train_classes = set(label_encoder.classes_.tolist())

    mask_known = np.array(
        [label in train_classes for label in yte_str],
        dtype=bool,
    )

    if not mask_known.all():
        n_removed = int((~mask_known).sum())
        print(
            f"[WARN] Removing {n_removed} test samples with labels unseen in training."
        )

        Xte = Xte[mask_known]
        yte_str = yte_str[mask_known]
        mte = mte.loc[mask_known].reset_index(drop=True)

    yte_cls = label_encoder.transform(yte_str)

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    _save_json(
        Path(cache_dir).parent / "label_mapping_hierarchy_order.json",
        {
            "classes": class_names,
            "split_name": split_name,
        },
    )

    # Pulse-width feature normalization
    if use_pt_as_feature and "pulse_ms" in mtr.columns:
        pt_train_ms = mtr["pulse_ms"].astype(float).to_numpy(dtype=np.float32)
        pt_log = np.log1p(pt_train_ms)

        pt_norm = (
            float(pt_log.mean()),
            float(pt_log.std() + 1e-8),
        )
    else:
        pt_norm = (0.0, 1.0)

    ds_tr = HierPulseDataset(
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

    ds_te = HierPulseDataset(
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

    dl_te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return {
        "dl_tr": dl_tr,
        "dl_te": dl_te,
        "num_classes": num_classes,
        "class_names": class_names,
        "soc_norm": soc_norm,
        "soh_norm": soh_norm,
        "normalize_soc": normalize_soc,
        "zscore_normalize": zscore_normalize,
        "n_train": int(len(ds_tr)),
        "n_test": int(len(ds_te)),
    }


# =============================================================================
# Main runner
# =============================================================================

def _add_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    summary = summary.copy()

    summary["mat_acc_pct"] = summary["cls_acc"].astype(float) * 100.0
    summary["soc_medape_pct"] = summary["soc_medape_raw"].astype(float)
    summary["soh_medape_pct"] = summary["soh_medape_raw"].astype(float)

    if "M_SOC_SOH" in set(summary["order"]):
        ref = summary.loc[summary["order"] == "M_SOC_SOH"].iloc[0]

        ref_acc = float(ref["cls_acc"])
        ref_soc = float(ref["soc_medape_raw"])
        ref_soh = float(ref["soh_medape_raw"])

        summary["mat_acc_change_pp_vs_proposed"] = (
            summary["cls_acc"].astype(float) - ref_acc
        ) * 100.0

        summary["soc_medape_change_pp_vs_proposed"] = (
            summary["soc_medape_raw"].astype(float) - ref_soc
        )

        summary["soh_medape_change_pp_vs_proposed"] = (
            summary["soh_medape_raw"].astype(float) - ref_soh
        )

    return summary


def run_hierarchy_order_ablation(
    data_root: str | Path,
    output_root: str | Path,
    orders: Optional[List[str]] = None,
    smoke: bool = False,
    # Fast training settings
    quick_epochs: int = 30,
    patience: int = 8,
    # Data settings
    u_start: int = 1,
    u_end: int = 41,
    drop_first_class: bool = True,
    soc_col: str = "SOC",
    soh_col: str = "SOH",
    use_pt_as_feature: bool = True,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
    test_id_frac: float = 0.2,
    test_id_count: int = 0,
    normalize_soc: bool = True,
    zscore_normalize: bool = True,
    # Model settings
    width: int = 24,
    blocks: int = 2,
    flow_hidden: int = 48,
    flow_layers: int = 3,
    flow_bins: int = 8,
    # Optimization
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
    # Loss weights
    w_cls: float = 1.0,
    w_soc: float = 1.0,
    w_soh: float = 1.0,
) -> pd.DataFrame:
    """
    Run fast hierarchy-order ablation.

    This experiment is for comparing relative hierarchy-order reasonableness,
    not for final best-performance training.
    """
    output_root = Path(output_root)
    data_root = Path(data_root)

    ensure_dir(str(output_root))

    set_random_seed(seed)

    if orders is None:
        orders = DEFAULT_ORDER_LIST

    for order in orders:
        if order not in ORDER_DESCRIPTIONS:
            raise ValueError(
                f"Unknown order={order}. Choose from {list(ORDER_DESCRIPTIONS)}."
            )

    if smoke:
        pulse_list = [5000]
        orders = ["M_SOC_SOH", "PARALLEL"]
        quick_epochs = 1
        patience = 1
        batch_size = 32
        width = 16
        blocks = 1
        flow_hidden = 32
        flow_layers = 1
    else:
        pulse_list = DEFAULT_PULSE_LIST

    cache_dir = output_root / "cache"
    splits_dir = output_root / "splits"

    data = prepare_hierarchy_order_data(
        data_root=data_root,
        pulse_list=pulse_list,
        cache_dir=cache_dir,
        splits_dir=splits_dir,
        u_start=u_start,
        u_end=u_end,
        drop_first_class=drop_first_class,
        soc_col=soc_col,
        soh_col=soh_col,
        use_pt_as_feature=use_pt_as_feature,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        test_id_frac=test_id_frac,
        test_id_count=test_id_count,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
    )

    dl_tr = data["dl_tr"]
    dl_te = data["dl_te"]
    num_classes = int(data["num_classes"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    criterion_cls = nn.CrossEntropyLoss()

    rows = []

    for order in orders:
        print("\n" + "=" * 90)
        print(f"[RUN] Hierarchy order: {order}")
        print(f"[RUN] Meaning: {ORDER_DESCRIPTIONS[order]}")
        print(f"[RUN] Output directory: {output_root / order}")
        print("=" * 90)

        set_random_seed(seed)

        order_dir = output_root / order
        ckpt_dir = order_dir / "checkpoints"
        logs_dir = order_dir / "logs"
        metrics_dir = order_dir / "metrics"

        ensure_dir(str(order_dir), str(ckpt_dir), str(logs_dir), str(metrics_dir))

        _save_json(
            order_dir / "run_config.json",
            {
                "order": order,
                "description": ORDER_DESCRIPTIONS[order],
                "pulse_list": list(map(int, pulse_list)),
                "quick_epochs": int(quick_epochs),
                "patience": int(patience),
                "batch_size": int(batch_size),
                "seed": int(seed),
                "width": int(width),
                "blocks": int(blocks),
                "flow_hidden": int(flow_hidden),
                "flow_layers": int(flow_layers),
                "flow_bins": int(flow_bins),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "grad_clip": float(grad_clip),
                "normalize_soc": bool(normalize_soc),
                "zscore_normalize": bool(zscore_normalize),
            },
        )

        model = HierOrderAblationModel(
            num_classes=num_classes,
            order=order,
            width=width,
            blocks=blocks,
            drop2d=0.0,
            use_pt_as_feature=use_pt_as_feature,
            head_dropout=0.1,
            flow_hidden=flow_hidden,
            flow_layers=flow_layers,
            flow_bins=flow_bins,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

        best_score = -1e18
        best_epoch = -1
        bad_count = 0
        best_state = None

        log_path = logs_dir / "train_log.csv"
        best_path = ckpt_dir / "best.pt"

        if log_path.exists():
            log_path.unlink()

        start_time = time.time()

        for epoch in range(int(quick_epochs)):
            tr = train_one_epoch_order(
                model=model,
                loader=dl_tr,
                optimizer=optimizer,
                device=device,
                criterion_cls=criterion_cls,
                w_cls=w_cls,
                w_soc=w_soc,
                w_soh=w_soh,
                grad_clip=grad_clip,
                n_mc=4,
            )

            te = eval_order(
                model=model,
                loader=dl_te,
                device=device,
                criterion_cls=criterion_cls,
                soc_norm=data["soc_norm"] if zscore_normalize else None,
                soh_norm=data["soh_norm"] if zscore_normalize else None,
                normalize_soc=normalize_soc,
                zscore_normalize=zscore_normalize,
                n_mc=12,
            )

            # Score for quick early stopping:
            # classification higher is better; SOC/SOH MedAPE lower is better.
            score = (
                float(te["cls_acc"])
                - 0.01 * float(te["soc_medape_raw"])
                - 0.02 * float(te["soh_medape_raw"])
            )

            row_epoch = {
                "order": order,
                "epoch": int(epoch),
                "train_loss": tr["loss"],
                "train_cls_acc": tr["cls_acc"],
                "test_loss": te["loss"],
                "test_cls_acc": te["cls_acc"],
                "test_soc_rmse_raw": te["soc_rmse_raw"],
                "test_soc_mae_raw": te["soc_mae_raw"],
                "test_soc_mape_raw": te["soc_mape_raw"],
                "test_soc_medape_raw": te["soc_medape_raw"],
                "test_soh_rmse_raw": te["soh_rmse_raw"],
                "test_soh_mae_raw": te["soh_mae_raw"],
                "test_soh_mape_raw": te["soh_mape_raw"],
                "test_soh_medape_raw": te["soh_medape_raw"],
                "score": score,
                "elapsed_sec": time.time() - start_time,
            }

            pd.DataFrame([row_epoch]).to_csv(
                log_path,
                mode="a",
                header=not log_path.exists(),
                index=False,
                encoding="utf-8-sig",
            )

            print(
                f"[{order}] epoch {epoch:03d} | "
                f"cls={te['cls_acc']:.4f} | "
                f"SOC MedAPE={te['soc_medape_raw']:.3f}% | "
                f"SOH MedAPE={te['soh_medape_raw']:.3f}% | "
                f"score={score:.6f}"
            )

            if score > best_score:
                best_score = score
                best_epoch = int(epoch)
                bad_count = 0

                best_state = {
                    "epoch": int(epoch),
                    "model": model.state_dict(),
                    "score": float(score),
                    "order": order,
                }

                torch.save(best_state, best_path)
            else:
                bad_count += 1

            if bad_count >= int(patience):
                print(
                    f"[EARLY STOP] {order} | "
                    f"best_epoch={best_epoch} | best_score={best_score:.6f}"
                )
                break

        if best_state is None and best_path.exists():
            best_state = _torch_load(best_path, map_location=device)

        if best_state is not None:
            model.load_state_dict(best_state["model"])

        te_best = eval_order(
            model=model,
            loader=dl_te,
            device=device,
            criterion_cls=criterion_cls,
            soc_norm=data["soc_norm"] if zscore_normalize else None,
            soh_norm=data["soh_norm"] if zscore_normalize else None,
            normalize_soc=normalize_soc,
            zscore_normalize=zscore_normalize,
            n_mc=24,
        )

        final_row = {
            "order": order,
            "description": ORDER_DESCRIPTIONS[order],
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "cls_acc": float(te_best["cls_acc"]),

            "soc_rmse_raw": float(te_best["soc_rmse_raw"]),
            "soc_mae_raw": float(te_best["soc_mae_raw"]),
            "soc_mape_raw": float(te_best["soc_mape_raw"]),
            "soc_medape_raw": float(te_best["soc_medape_raw"]),

            "soh_rmse_raw": float(te_best["soh_rmse_raw"]),
            "soh_mae_raw": float(te_best["soh_mae_raw"]),
            "soh_mape_raw": float(te_best["soh_mape_raw"]),
            "soh_medape_raw": float(te_best["soh_medape_raw"]),

            "n_train": int(data["n_train"]),
            "n_test": int(data["n_test"]),
            "num_classes": int(num_classes),
            "device": device,
            "elapsed_sec": float(time.time() - start_time),
        }

        pd.DataFrame([final_row]).to_csv(
            metrics_dir / "final_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        _save_json(metrics_dir / "final_metrics.json", final_row)

        rows.append(final_row)

        partial = pd.DataFrame(rows)
        partial = _add_summary_columns(partial)

        partial.to_csv(
            output_root / "hierarchy_order_ablation_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )

        _save_json(
            output_root / "hierarchy_order_ablation_partial.json",
            rows,
        )

        print(f"[BEST] {order}")
        print(final_row)

    summary = pd.DataFrame(rows)
    summary = _add_summary_columns(summary)

    summary.to_csv(
        output_root / "hierarchy_order_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _save_json(
        output_root / "hierarchy_order_ablation_summary.json",
        summary.to_dict(orient="records"),
    )

    return summary


def main() -> None:
    data_root = PROJECT_ROOT / "data"
    output_root = PROJECT_ROOT / "results" / "ablation" / "hierarchy_order_ablation"

    summary = run_hierarchy_order_ablation(
        data_root=data_root,
        output_root=output_root,
        orders=DEFAULT_ORDER_LIST,
        smoke=False,

        quick_epochs=30,
        patience=8,

        batch_size=128,
        num_workers=0,

        width=24,
        blocks=2,
        flow_hidden=48,
        flow_layers=3,
        flow_bins=8,

        lr=5e-4,
        weight_decay=1e-4,
        grad_clip=5.0,

        seed=42,

        normalize_soc=True,
        zscore_normalize=True,

        use_pt_as_feature=True,
    )

    print("\n[SUMMARY]")
    print(summary)


if __name__ == "__main__":
    main()