# ablation/channel_ablation.py
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
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


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
# Channel configurations
# =============================================================================

DEFAULT_PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

CHANNEL_MODES: Dict[str, Tuple[str, ...]] = {
    "ch1_only": ("ch1",),
    "ch12": ("ch1", "ch2"),
    "ch13": ("ch1", "ch3"),
    "full": ("ch1", "ch2", "ch3"),
}


# =============================================================================
# Feature builder
# =============================================================================

def build_channel_input_from_u41(
    u: np.ndarray,
    channel_mode: str = "full",
) -> np.ndarray:
    """
    Build channel-ablation input from U1-U41.

    Channel definitions
    -------------------
    ch1:
        Raw pulse-response voltage U2-U41 reshaped into 5x8.

    ch2:
        Differential / voltage-jump channel:
        U2-U1, U3-U2, ..., U41-U40, reshaped into 5x8.

    ch3:
        OCV-related rested-voltage baseline U1 repeated into 5x8.

    channel_mode
    ------------
    ch1_only:
        output shape = (1, 5, 8)

    ch12:
        output shape = (2, 5, 8)

    ch13:
        output shape = (2, 5, 8)

    full:
        output shape = (3, 5, 8)
    """
    u = np.asarray(u, dtype=np.float32)

    if u.shape[0] != 41:
        raise ValueError(f"Expected 41 U values, got {u.shape[0]}.")

    if channel_mode not in CHANNEL_MODES:
        raise ValueError(
            f"Unknown channel_mode={channel_mode}. "
            f"Choose from {list(CHANNEL_MODES.keys())}."
        )

    u1 = float(u[0])
    u2_41 = u[1:]

    ch1 = u2_41.reshape(5, 8)

    diff = np.empty(40, dtype=np.float32)
    diff[0] = u[1] - u[0]
    diff[1:] = u[2:] - u[1:-1]
    ch2 = diff.reshape(5, 8)

    ch3 = np.full((5, 8), u1, dtype=np.float32)

    channel_dict = {
        "ch1": ch1,
        "ch2": ch2,
        "ch3": ch3,
    }

    return np.stack(
        [channel_dict[name] for name in CHANNEL_MODES[channel_mode]],
        axis=0,
    ).astype(np.float32)


def get_input_channels(channel_mode: str) -> int:
    if channel_mode not in CHANNEL_MODES:
        raise ValueError(
            f"Unknown channel_mode={channel_mode}. "
            f"Choose from {list(CHANNEL_MODES.keys())}."
        )

    return len(CHANNEL_MODES[channel_mode])


# =============================================================================
# Dataset
# =============================================================================

class ChannelAblationDataset(Dataset):
    """
    Dataset for channel-ablation experiments.

    Each sample returns:
    - x: structured input with shape (C, 5, 8)
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
        channel_mode: str,
        pt_col: str = "pulse_ms",
        use_pt_as_feature: bool = True,
        pt_norm: Optional[Tuple[float, float]] = None,
        normalize_soc: bool = True,
        zscore_normalize: bool = True,
        soc_norm: Optional[Tuple[float, float]] = None,
        soh_norm: Optional[Tuple[float, float]] = None,
    ):
        if channel_mode not in CHANNEL_MODES:
            raise ValueError(
                f"Unknown channel_mode={channel_mode}. "
                f"Choose from {list(CHANNEL_MODES.keys())}."
            )

        self.X_u = X_u
        self.y_cls = y_cls.astype(np.int64)
        self.meta = meta.reset_index(drop=True)
        self.channel_mode = str(channel_mode)

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
        x = torch.from_numpy(
            build_channel_input_from_u41(
                self.X_u[idx],
                channel_mode=self.channel_mode,
            )
        )

        y_cls = torch.tensor(int(self.y_cls[idx]), dtype=torch.long)

        if self.use_pt and self.pt_ms is not None:
            p = (np.log1p(float(self.pt_ms[idx])) - self.pt_mean) / self.pt_std
            pt = torch.tensor([p], dtype=torch.float32)
        else:
            pt = torch.tensor([0.0], dtype=torch.float32)

        soc = torch.tensor(float(self.soc[idx]), dtype=torch.float32)
        soh = torch.tensor(float(self.soh[idx]), dtype=torch.float32)

        return x, pt, y_cls, soc, soh


# =============================================================================
# Variable-channel model
# =============================================================================

class ResBlock(nn.Module):
    def __init__(self, c: int, drop2d: float = 0.0):
        super().__init__()

        group_num = 8 if c % 8 == 0 else 4

        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(group_num, c)

        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(group_num, c)

        self.act = nn.ReLU(inplace=True)
        self.drop = (
            nn.Dropout2d(drop2d)
            if drop2d is not None and drop2d > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(self.conv1(x)))
        h = self.drop(h)
        h = self.gn2(self.conv2(h))

        return self.act(x + h)


class VariableChannelEncoder(nn.Module):
    """
    Micro-ResNet encoder that accepts 1, 2 or 3 input channels.
    """

    def __init__(
        self,
        input_channels: int,
        width: int = 32,
        blocks: int = 4,
        drop2d: float = 0.0,
    ):
        super().__init__()

        self.input_channels = int(input_channels)

        group_num = 8 if width % 8 == 0 else 4

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.input_channels,
                width,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(group_num, width),
            nn.ReLU(inplace=True),
        )

        self.body = nn.Sequential(
            *[ResBlock(width, drop2d=drop2d) for _ in range(int(blocks))]
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        z = self.stem(x_img)
        z = self.body(z)
        z = self.pool(z).flatten(1)

        return z


class ChannelAblationModel(nn.Module):
    """
    Hierarchical probabilistic model for channel ablation.

    Difference from the main proposed model:
    - input_channels can be 1, 2 or 3.

    Hierarchy:
    1. Material-capacity classification.
    2. SOC conditional flow.
    3. SOH conditional flow.
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int,
        width: int = 32,
        blocks: int = 4,
        drop2d: float = 0.0,
        use_pt_as_feature: bool = True,
        soc_hidden: int = 64,
        soh_hidden: int = 64,
        head_dropout: float = 0.2,
        flow_layers: int = 6,
        flow_bins: int = 8,
        flow_tail_bound: float = 3.0,
    ):
        super().__init__()

        self.encoder = VariableChannelEncoder(
            input_channels=int(input_channels),
            width=width,
            blocks=blocks,
            drop2d=drop2d,
        )

        self.use_pt = bool(use_pt_as_feature)

        pt_dim = 1 if self.use_pt else 0
        p_dim = int(num_classes)

        # Keep material head consistent with the updated proposed model:
        # material classification uses encoder embedding plus optional pulse width.
        mat_input_dim = width + pt_dim

        self.head_mat = nn.Sequential(
            nn.Linear(mat_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(64, num_classes),
        )

        soc_context_dim = width + p_dim + pt_dim

        self.soc_flow = Conditional1DFlow(
            context_dim=soc_context_dim,
            hidden_features=int(soc_hidden),
            num_transforms=int(flow_layers),
            num_bins=int(flow_bins),
            tail_bound=float(flow_tail_bound),
        )

        soh_context_dim = width + p_dim + 1 + pt_dim

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


def _stage_score(
    stage: str,
    te: dict,
    alpha_score: float,
) -> float:
    if stage == "stage1_soc":
        return float(te["cls_acc"] - 0.3 * te["soc_rmse"])

    if stage == "stage2_soh":
        return float(-te["soh_rmse"])

    return float(
        te["cls_acc"]
        - float(alpha_score) * (te["soc_rmse"] + te["soh_rmse"])
    )


# =============================================================================
# One channel-ablation experiment
# =============================================================================

def run_channel_experiment(
    data_root: str | Path,
    pulse_list: List[int],
    channel_mode: str,
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
    max_epochs: int = 200,
    early_stopping: bool = True,
    patience: int = 20,
    resume: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    # model
    width: int = 32,
    blocks: int = 4,
    drop2d: float = 0.0,
    head_dropout: float = 0.2,
    # losses
    w_cls: float = 1.0,
    w_soc: float = 1.0,
    w_soh: float = 1.0,
    # split
    test_id_frac: float = 0.2,
    test_id_count: int = 0,
    # target normalization
    normalize_soc: bool = True,
    zscore_normalize: bool = True,
    # two-stage
    two_stage: bool = True,
    stage1_epochs: int = 200,
    stage2_epochs: int = 200,
    finetune_epochs: int = 30,
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
) -> dict:
    """
    Train and evaluate one channel-ablation setting.
    """
    if channel_mode not in CHANNEL_MODES:
        raise ValueError(
            f"Unknown channel_mode={channel_mode}. "
            f"Choose from {list(CHANNEL_MODES.keys())}."
        )

    start_time = time.time()

    data_root = Path(data_root)
    exp_dir = Path(exp_dir)

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
    input_channels = get_input_channels(channel_mode)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Channel mode: {channel_mode} | input_channels={input_channels}")

    run_config = {
        "data_root": str(data_root),
        "pulse_list": list(map(int, pulse_list)),
        "channel_mode": channel_mode,
        "input_channels": int(input_channels),
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
        "w_cls": w_cls,
        "w_soc": w_soc,
        "w_soh": w_soh,
        "test_id_frac": test_id_frac,
        "test_id_count": test_id_count,
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
        "soc_prior_low": soc_prior_low,
        "soc_prior_mid": soc_prior_mid,
        "soc_prior_high": soc_prior_high,
        "soh_prior_low": soh_prior_low,
        "soh_prior_mid": soh_prior_mid,
        "soh_prior_high": soh_prior_high,
        "alpha_score": alpha_score,
        "final_best_stage": final_best_stage,
        "exp_dir": str(exp_dir),
    }

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
    # 2. ID-level split
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

    # =========================================================================
    # 3. Train-only normalization of U1-U41
    # =========================================================================

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

    if len(yte_str) == 0:
        raise RuntimeError("No test samples remain after filtering unknown labels.")

    yte_cls = label_encoder.transform(yte_str)

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    _save_json(
        exp_dir / "label_mapping.json",
        {
            "classes": class_names,
            "split_name": split_name,
            "channel_mode": channel_mode,
            "input_channels": int(input_channels),
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

    ds_tr = ChannelAblationDataset(
        X_u=Xtr,
        y_cls=ytr_cls,
        meta=mtr,
        soc_col=soc_col,
        soh_col=soh_col,
        channel_mode=channel_mode,
        use_pt_as_feature=use_pt_as_feature,
        pt_norm=pt_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
        soc_norm=soc_norm if zscore_normalize else None,
        soh_norm=soh_norm if zscore_normalize else None,
    )

    ds_te = ChannelAblationDataset(
        X_u=Xte,
        y_cls=yte_cls,
        meta=mte,
        soc_col=soc_col,
        soh_col=soh_col,
        channel_mode=channel_mode,
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

    # =========================================================================
    # 8. Model
    # =========================================================================

    model = ChannelAblationModel(
        num_classes=num_classes,
        input_channels=input_channels,
        width=width,
        blocks=blocks,
        drop2d=drop2d,
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

            te = eval_one_epoch(
                model=model,
                loader=dl_te,
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
            )

            score = _stage_score(stage, te, alpha_score=alpha_score)

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
                        "test_loss": te["loss"],
                        "test_cls_acc": te["cls_acc"],
                        "test_soc_rmse": te["soc_rmse"],
                        "test_soc_mae": te["soc_mae"],
                        "test_soc_mape": te["soc_mape"],
                        "test_soc_medape": te.get("soc_medape", np.nan),
                        "test_soh_rmse": te["soh_rmse"],
                        "test_soh_mae": te["soh_mae"],
                        "test_soh_mape": te["soh_mape"],
                        "test_soh_medape": te.get("soh_medape", np.nan),
                        "test_soc_rmse_raw": te["soc_rmse_raw"],
                        "test_soc_mae_raw": te["soc_mae_raw"],
                        "test_soc_mape_raw": te["soc_mape_raw"],
                        "test_soc_medape_raw": te.get("soc_medape_raw", np.nan),
                        "test_soh_rmse_raw": te["soh_rmse_raw"],
                        "test_soh_mae_raw": te["soh_mae_raw"],
                        "test_soh_mape_raw": te["soh_mape_raw"],
                        "test_soh_medape_raw": te.get("soh_medape_raw", np.nan),
                        "test_score": score,
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
                f"TE cls={te['cls_acc']:.4f} | "
                f"SOC MedAPE(raw)={te.get('soc_medape_raw', np.nan):.3f}% | "
                f"SOH MedAPE(raw)={te.get('soh_medape_raw', np.nan):.3f}% | "
                f"score={score:.6f}"
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
    # 11. Training
    # =========================================================================

    stage_best_paths = {}

    if two_stage:
        stage_best_paths["stage1_soc"] = run_stage(
            stage="stage1_soc",
            epochs=int(stage1_epochs),
            w_cls_s=float(w_cls),
            w_soc_s=float(w_soc),
            w_soh_s=0.0,
        )

        stage_best_paths["stage2_soh"] = run_stage(
            stage="stage2_soh",
            epochs=int(stage2_epochs),
            w_cls_s=0.0,
            w_soc_s=0.0,
            w_soh_s=float(w_soh),
        )

        if finetune_epochs and finetune_epochs > 0:
            stage_best_paths["finetune"] = run_stage(
                stage="finetune",
                epochs=int(finetune_epochs),
                w_cls_s=float(w_cls) * 0.4,
                w_soc_s=float(w_soc),
                w_soh_s=float(w_soh),
            )
    else:
        stage_best_paths["single"] = run_stage(
            stage="single",
            epochs=int(max_epochs),
            w_cls_s=float(w_cls),
            w_soc_s=float(w_soc),
            w_soh_s=float(w_soh),
        )

    # =========================================================================
    # 12. Final checkpoint selection
    # =========================================================================

    if not two_stage:
        chosen = "single"
    else:
        chosen = final_best_stage

        if chosen == "finetune" and "finetune" not in stage_best_paths:
            chosen = (
                "stage2_soh"
                if "stage2_soh" in stage_best_paths
                else "stage1_soc"
            )

        if chosen not in stage_best_paths:
            chosen = (
                "stage2_soh"
                if "stage2_soh" in stage_best_paths
                else "stage1_soc"
            )

    best_path = stage_best_paths[chosen]

    if best_path.exists():
        ckpt = _torch_load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        print(
            f"[FINAL] Using BEST checkpoint from stage='{chosen}' | "
            f"epoch={ckpt.get('epoch')} | score={ckpt.get('best_score')}"
        )

    if chosen == "stage1_soc":
        w_cls_eval = float(w_cls)
        w_soc_eval = float(w_soc)
        w_soh_eval = 0.0
    elif chosen == "stage2_soh":
        w_cls_eval = 0.0
        w_soc_eval = 0.0
        w_soh_eval = float(w_soh)
    else:
        w_cls_eval = float(w_cls)
        w_soc_eval = float(w_soc)
        w_soh_eval = float(w_soh)

    # =========================================================================
    # 13. Final evaluation
    # =========================================================================

    te = eval_one_epoch(
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
    )

    elapsed_sec = time.time() - start_time

    out = {
        "channel_mode": channel_mode,
        "input_channels": int(input_channels),
        "channels": "+".join(CHANNEL_MODES[channel_mode]),
        "final_stage": chosen,
        "test_cls_acc": float(te["cls_acc"]),

        "test_soc_rmse": float(te["soc_rmse"]),
        "test_soc_mae": float(te["soc_mae"]),
        "test_soc_mape": float(te["soc_mape"]),
        "test_soc_medape": float(te.get("soc_medape", np.nan)),

        "test_soh_rmse": float(te["soh_rmse"]),
        "test_soh_mae": float(te["soh_mae"]),
        "test_soh_mape": float(te["soh_mape"]),
        "test_soh_medape": float(te.get("soh_medape", np.nan)),

        "test_soc_rmse_raw": float(te["soc_rmse_raw"]),
        "test_soc_mae_raw": float(te["soc_mae_raw"]),
        "test_soc_mape_raw": float(te["soc_mape_raw"]),
        "test_soc_medape_raw": float(te.get("soc_medape_raw", np.nan)),

        "test_soh_rmse_raw": float(te["soh_rmse_raw"]),
        "test_soh_mae_raw": float(te["soh_mae_raw"]),
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
# Channel-ablation runner
# =============================================================================

def _add_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    summary = summary.copy()

    summary["mat_acc_pct"] = summary["test_cls_acc"].astype(float) * 100.0
    summary["soc_medape_pct"] = summary["test_soc_medape_raw"].astype(float)
    summary["soh_medape_pct"] = summary["test_soh_medape_raw"].astype(float)

    if "full" in set(summary["channel_mode"]):
        ref = summary.loc[summary["channel_mode"] == "full"].iloc[0]

        ref_acc = float(ref["test_cls_acc"])
        ref_soc = float(ref["test_soc_medape_raw"])
        ref_soh = float(ref["test_soh_medape_raw"])

        summary["mat_acc_change_pp_vs_full"] = (
            summary["test_cls_acc"].astype(float) - ref_acc
        ) * 100.0

        summary["soc_medape_change_pp_vs_full"] = (
            summary["test_soc_medape_raw"].astype(float) - ref_soc
        )

        summary["soh_medape_change_pp_vs_full"] = (
            summary["test_soh_medape_raw"].astype(float) - ref_soh
        )

    return summary


def run_channel_ablation(
    data_root: str | Path,
    output_root: str | Path,
    smoke: bool = False,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Run channel-ablation experiments.

    Full channel setting here is trained in the same script for direct comparison.
    If you already have proposed_framework full result, you may still keep this
    full row for consistency.
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if smoke:
        channel_modes = ["ch1_only", "full"]
        pulse_list = [5000]

        run_kwargs = {
            "batch_size": 32,
            "max_epochs": 1,
            "early_stopping": False,
            "patience": 1,
            "resume": False,
            "width": 16,
            "blocks": 1,
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
        channel_modes = ["ch1_only", "ch12", "ch13", "full"]
        pulse_list = DEFAULT_PULSE_LIST

        run_kwargs = {
            "batch_size": 128,
            "max_epochs": 400,
            "early_stopping": False,
            "patience": 20,
            "resume": resume,
            "width": 32,
            "blocks": 4,
            "head_dropout": 0.2,
            "two_stage": True,
            "stage1_epochs": 200,
            "stage2_epochs": 200,
            "finetune_epochs": 30,
            "use_soc_prior_weighting": True,
            "use_soh_prior_weighting": True,
            "final_best_stage": "finetune",
        }

    rows = []

    for channel_mode in channel_modes:
        exp_dir = output_root / channel_mode

        print("\n" + "=" * 90)
        print(f"[RUN] Channel ablation: {channel_mode}")
        print(f"[RUN] Channels: {CHANNEL_MODES[channel_mode]}")
        print(f"[RUN] Input channels: {get_input_channels(channel_mode)}")
        print(f"[RUN] Pulse list: {pulse_list}")
        print(f"[RUN] Output directory: {exp_dir}")
        print("=" * 90)

        out = run_channel_experiment(
            data_root=data_root,
            pulse_list=pulse_list,
            channel_mode=channel_mode,
            exp_dir=exp_dir,

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

            drop2d=0.0,

            w_cls=1.0,
            w_soc=1.0,
            w_soh=1.0,

            test_id_frac=0.2,
            test_id_count=0,

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

            **run_kwargs,
        )

        row = {
            "config": channel_mode,
            "pulse_widths_ms": ",".join(map(str, pulse_list)),
            "num_pulse_widths": len(pulse_list),
            **out,
        }

        rows.append(row)

        partial = pd.DataFrame(rows)
        partial = _add_summary_columns(partial)

        partial.to_csv(
            output_root / "channel_ablation_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )

        _save_json(
            output_root / "channel_ablation_partial.json",
            rows,
        )

    summary = pd.DataFrame(rows)
    summary = _add_summary_columns(summary)

    summary.to_csv(
        output_root / "channel_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _save_json(
        output_root / "channel_ablation_summary.json",
        summary.to_dict(orient="records"),
    )

    return summary


def main() -> None:
    data_root = PROJECT_ROOT / "data"
    output_root = PROJECT_ROOT / "results" / "ablation" / "channel_ablation"

    summary = run_channel_ablation(
        data_root=data_root,
        output_root=output_root,
        smoke=False,
        resume=True,
    )

    print("\n[SUMMARY]")
    print(summary)


if __name__ == "__main__":
    main()