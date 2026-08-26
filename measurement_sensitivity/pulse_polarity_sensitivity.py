# measurement_sensitivity/pulse_polarity_sensitivity.py
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
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader


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
from proposed_framework.models.hierarchical_model import Hier3HeadModel
from proposed_framework.training.trainer import train_one_epoch
from proposed_framework.training.evaluator import eval_one_epoch


# =============================================================================
# Configurations
# =============================================================================

DEFAULT_PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

POLARITY_CONFIGS: Dict[str, str] = {
    "positive_only": "positive",
    "negative_only": "negative",
}


# =============================================================================
# Feature builder for polarity sensitivity
# =============================================================================

def build_3ch_5x4_from_u41(
    u: np.ndarray,
    part: str,
) -> np.ndarray:
    """
    Convert U1-U41 voltage features into a polarity-specific 3-channel 5x4 input.

    Full representation before slicing:
        U2-U41 -> 5 x 8

    Polarity slicing:
        positive: take first 4 columns from each C-rate row
        negative: take last 4 columns from each C-rate row

    Output:
        shape = (3, 5, 4)

    Channel 1:
        Raw pulse response values.

    Channel 2:
        Voltage-jump / voltage-difference features.

    Channel 3:
        U1 rested-voltage baseline repeated.
    """
    u = np.asarray(u, dtype=np.float32)

    if u.shape[0] != 41:
        raise ValueError(f"Expected 41 U values, got {u.shape[0]}.")

    if part not in {"positive", "negative"}:
        raise ValueError("part must be either 'positive' or 'negative'.")

    u1 = float(u[0])
    u2_41 = u[1:]

    ch1_full = u2_41.reshape(5, 8)

    diff = np.empty(40, dtype=np.float32)
    diff[0] = u[1] - u[0]
    diff[1:] = u[2:] - u[1:-1]
    ch2_full = diff.reshape(5, 8)

    ch3_full = np.full((5, 8), u1, dtype=np.float32)

    if part == "positive":
        col_slice = slice(0, 4)
    else:
        col_slice = slice(4, 8)

    ch1 = ch1_full[:, col_slice]
    ch2 = ch2_full[:, col_slice]
    ch3 = ch3_full[:, col_slice]

    return np.stack([ch1, ch2, ch3], axis=0)


# =============================================================================
# Dataset
# =============================================================================

class PolarityPulseDataset(Dataset):
    """
    Dataset for positive-only / negative-only pulse-polarity sensitivity.

    Each sample returns:
    - x3: polarity-specific structured input, shape (3, 5, 4)
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
        part: str,
        pt_col: str = "pulse_ms",
        use_pt_as_feature: bool = True,
        pt_norm: Optional[Tuple[float, float]] = None,
        normalize_soc: bool = True,
        zscore_normalize: bool = True,
        soc_norm: Optional[Tuple[float, float]] = None,
        soh_norm: Optional[Tuple[float, float]] = None,
    ):
        if part not in {"positive", "negative"}:
            raise ValueError("part must be either 'positive' or 'negative'.")

        self.X_u = X_u
        self.y_cls = y_cls.astype(np.int64)
        self.meta = meta.reset_index(drop=True)
        self.part = part

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
                    "zscore_normalize=True requires soc_norm and soh_norm from training data."
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
        x3 = torch.from_numpy(
            build_3ch_5x4_from_u41(
                self.X_u[idx],
                part=self.part,
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

        return x3, pt, y_cls, soc, soh


# =============================================================================
# Helper functions
# =============================================================================
def _medae(a, b):
    return float(np.median(np.abs(np.asarray(a) - np.asarray(b))))

def _material_accuracy(true_labels, pred_labels):
    true_material = [x.split("_")[0] for x in true_labels]
    pred_material = [x.split("_")[0] for x in pred_labels]

    return float(
        np.mean(
            np.array(true_material) == np.array(pred_material)
        )
    )    
    
def _torch_load(path: str | Path, map_location: str):
    """
    Compatibility wrapper for torch.load across PyTorch versions.
    """
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
    """
    Stage-specific model selection score.
    """
    if stage == "stage1_soc":
        return float(te["cls_acc"] - 0.3 * te["soc_rmse"])

    if stage == "stage2_soh":
        return float(-te["soh_rmse"])

    return float(
        te["cls_acc"] - float(alpha_score) * (te["soc_rmse"] + te["soh_rmse"])
    )


# =============================================================================
# One polarity experiment
# =============================================================================

def run_polarity_experiment(
    data_root: str | Path,
    pulse_list: List[int],
    part: str,
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
    Train and evaluate one polarity-specific model.

    part:
        "positive" -> use first 4 columns from each 5x8 C-rate block.
        "negative" -> use last 4 columns from each 5x8 C-rate block.
    """
    if part not in {"positive", "negative"}:
        raise ValueError("part must be either 'positive' or 'negative'.")

    start_time = time.time()

    data_root = Path(data_root)
    exp_dir = Path(exp_dir)

    cache_dir = exp_dir / "cache"
    ckpt_dir = exp_dir / "checkpoints"
    logs_dir = exp_dir / "logs"
    splits_dir = exp_dir / "splits"
    metrics_dir = exp_dir / "metrics"

    for d in [exp_dir, cache_dir, ckpt_dir, logs_dir, splits_dir, metrics_dir]:
        ensure_dir(str(d))

    set_random_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    run_config = {
        "data_root": str(data_root),
        "pulse_list": list(map(int, pulse_list)),
        "part": part,
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
        "val_id_frac": test_id_frac if val_id_frac is None else val_id_frac,
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
    # 1. Load and cache raw train/test data
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
    # 2. ID-level train/val/test split, consistent with run_proposed_framework
    # =========================================================================
    # TEST IDs are held out at the cell-ID level and evaluated on TEST_RANDOM.
    # VAL IDs are drawn from the remaining non-test IDs and are also evaluated
    # on TEST_RANDOM. The final TRAIN set uses grid-SOC training samples from
    # the remaining IDs only.

    all_ids = pd.concat(
        [mtr_raw["ID"], mte_raw["ID"]],
        axis=0,
    ).astype(str).to_numpy()

    test_ids = np.array(
        pick_test_ids(
            all_ids=all_ids,
            test_id_frac=test_id_frac,
            test_id_count=test_id_count,
            seed=seed,
        ),
        dtype=object,
    )

    unique_all_ids = pd.Series(all_ids).astype(str).drop_duplicates().to_numpy()
    non_test_ids = np.array(
        [item for item in unique_all_ids if str(item) not in set(map(str, test_ids))],
        dtype=object,
    )

    if val_id_frac is None:
        val_id_frac_use = 0.1
    else:
        val_id_frac_use = float(val_id_frac)

    val_ids = np.array(
        pick_test_ids(
            all_ids=non_test_ids,
            test_id_frac=val_id_frac_use,
            test_id_count=val_id_count,
            seed=seed + 1,
        ),
        dtype=object,
    )

    if test_id_count and test_id_count > 0:
        split_name = f"testIDs_seed{seed}_n{test_id_count}"
    else:
        split_name = f"testIDs_seed{seed}_frac{test_id_frac}"

    if val_id_count and val_id_count > 0:
        val_split_name = f"valIDs_seed{seed + 1}_n{val_id_count}_from_non_test"
    else:
        val_split_name = f"valIDs_seed{seed + 1}_frac{val_id_frac_use}_from_non_test"

    with open(splits_dir / f"{split_name}.txt", "w", encoding="utf-8") as f:
        for test_id in test_ids:
            f.write(str(test_id) + "\n")

    with open(splits_dir / f"{val_split_name}.txt", "w", encoding="utf-8") as f:
        for val_id in val_ids:
            f.write(str(val_id) + "\n")

    test_id_set = set(map(str, test_ids))
    val_id_set = set(map(str, val_ids))

    train_mask = ~(
        mtr_raw["ID"].astype(str).isin(test_id_set)
        | mtr_raw["ID"].astype(str).isin(val_id_set)
    ).to_numpy()
    val_mask = mte_raw["ID"].astype(str).isin(val_id_set).to_numpy()
    test_mask = mte_raw["ID"].astype(str).isin(test_id_set).to_numpy()

    Xtr = Xtr_raw[train_mask]
    ytr_str = np.asarray(ytr_raw)[train_mask]
    mtr = mtr_raw.loc[train_mask].reset_index(drop=True)

    Xval = Xte_raw[val_mask]
    yval_str = np.asarray(yte_raw)[val_mask]
    mval = mte_raw.loc[val_mask].reset_index(drop=True)

    Xte = Xte_raw[test_mask]
    yte_str = np.asarray(yte_raw)[test_mask]
    mte = mte_raw.loc[test_mask].reset_index(drop=True)

    if len(ytr_str) == 0 or len(yval_str) == 0 or len(yte_str) == 0:
        raise RuntimeError("Empty train, validation, or test data after applying ID split.")

    print(
        f"[DATA] Final TRAIN samples = {len(ytr_str)} | "
        f"unique IDs = {mtr['ID'].astype(str).nunique()}"
    )
    print(
        f"[DATA] Final VAL   samples = {len(yval_str)} | "
        f"unique IDs = {mval['ID'].astype(str).nunique()} "
        f"(from TEST_RANDOM, IDs drawn from non-test IDs)"
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
            f"[WARN] Removing {n_removed} validation samples with labels unseen in training."
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
        raise RuntimeError("No validation or test samples remain after filtering unknown labels.")

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

    ds_tr = PolarityPulseDataset(
        X_u=Xtr,
        y_cls=ytr_cls,
        meta=mtr,
        soc_col=soc_col,
        soh_col=soh_col,
        part=part,
        use_pt_as_feature=use_pt_as_feature,
        pt_norm=pt_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
        soc_norm=soc_norm if zscore_normalize else None,
        soh_norm=soh_norm if zscore_normalize else None,
    )

    ds_val = PolarityPulseDataset(
        X_u=Xval,
        y_cls=yval_cls,
        meta=mval,
        soc_col=soc_col,
        soh_col=soh_col,
        part=part,
        use_pt_as_feature=use_pt_as_feature,
        pt_norm=pt_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
        soc_norm=soc_norm if zscore_normalize else None,
        soh_norm=soh_norm if zscore_normalize else None,
    )

    ds_te = PolarityPulseDataset(
        X_u=Xte,
        y_cls=yte_cls,
        meta=mte,
        soc_col=soc_col,
        soh_col=soh_col,
        part=part,
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

    model = Hier3HeadModel(
        num_classes=num_classes,
        width=width,
        blocks=blocks,
        drop2d=drop2d,
        use_pt_as_feature=use_pt_as_feature,
        head_dropout=head_dropout,
    ).to(device)

    criterion_cls = nn.CrossEntropyLoss()

    # Kept for compatibility with trainer/evaluator function signatures.
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

            va = eval_one_epoch(
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

            # Keep TEST metrics for reporting, but select the best checkpoint
            # using VALIDATION score, consistent with run_proposed_framework.
            train_score = _stage_score(stage, tr, alpha_score=alpha_score)
            val_score = _stage_score(stage, va, alpha_score=alpha_score)
            test_score = _stage_score(stage, te, alpha_score=alpha_score)
            score = val_score

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
                        "val_loss": va["loss"],
                        "val_cls_acc": va["cls_acc"],
                        "val_soc_rmse": va["soc_rmse"],
                        "val_soc_mae": va["soc_mae"],
                        "val_soc_mape": va["soc_mape"],
                        "val_soc_medape": va.get("soc_medape", np.nan),
                        "val_soh_rmse": va["soh_rmse"],
                        "val_soh_mae": va["soh_mae"],
                        "val_soh_mape": va["soh_mape"],
                        "val_soh_medape": va.get("soh_medape", np.nan),
                        "val_soc_rmse_raw": va["soc_rmse_raw"],
                        "val_soc_mae_raw": va["soc_mae_raw"],
                        "val_soc_mape_raw": va["soc_mape_raw"],
                        "val_soc_medape_raw": va.get("soc_medape_raw", np.nan),
                        "val_soh_rmse_raw": va["soh_rmse_raw"],
                        "val_soh_mae_raw": va["soh_mae_raw"],
                        "val_soh_mape_raw": va["soh_mape_raw"],
                        "val_soh_medape_raw": va.get("soh_medape_raw", np.nan),
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
                        "train_score": train_score,
                        "val_score": val_score,
                        "test_score": test_score,
                        "selection_score": score,
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
                f"VAL cls={va['cls_acc']:.4f} | "
                f"VAL SOC MedAPE(raw)={va.get('soc_medape_raw', np.nan):.3f}% | "
                f"VAL SOH MedAPE(raw)={va.get('soh_medape_raw', np.nan):.3f}% | "
                f"val_score={val_score:.6f} | "
                f"TEST cls={te['cls_acc']:.4f}"
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
                        "selection_metric": "val_score",
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
                    "selection_metric": "val_score",
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

    va = eval_one_epoch(
        model=model,
        loader=dl_val,
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
        "polarity": part,
        "final_stage": chosen,
        "val_cls_acc": float(va["cls_acc"]),
        "val_soc_medape_raw": float(va.get("soc_medape_raw", np.nan)),
        "val_soh_medape_raw": float(va.get("soh_medape_raw", np.nan)),
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
        "n_val": int(len(ds_val)),
        "n_test": int(len(ds_te)),
        "num_classes": int(num_classes),
        "input_shape": "3x5x4",
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
# Retrospective evaluation of existing checkpoints
# =============================================================================

def _find_existing_checkpoint(exp_dir: str | Path, preferred_stage: Optional[str] = None) -> Path:
    exp_dir = Path(exp_dir)
    stages = [preferred_stage] if preferred_stage else []
    stages += ["finetune", "stage2_soh", "stage1_soc", "single"]
    for stage in dict.fromkeys(s for s in stages if s):
        path = exp_dir / "checkpoints" / stage / "best.pt"
        if path.exists():
            return path
    raise FileNotFoundError(f"No best.pt found under {exp_dir / 'checkpoints'}")


def _inverse_eval_targets(
    soc_z: np.ndarray,
    soh_z: np.ndarray,
    soc_norm: Tuple[float, float],
    soh_norm: Tuple[float, float],
    normalize_soc: bool,
    zscore_normalize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    soc = np.asarray(soc_z, dtype=np.float64).reshape(-1)
    soh = np.asarray(soh_z, dtype=np.float64).reshape(-1)
    if zscore_normalize:
        soc = soc * float(soc_norm[1]) + float(soc_norm[0])
        soh = soh * float(soh_norm[1]) + float(soh_norm[0])
    if normalize_soc:
        soc *= 100.0
    soh *= 100.0
    return soc, soh


@torch.no_grad()
def evaluate_existing_polarity_checkpoint(
    data_root: str | Path,
    exp_dir: str | Path,
    part: str,
    batch_size: Optional[int] = None,
    num_workers: int = 0,
    n_mc: int = 500,
) -> dict:
    """Load an existing checkpoint and recompute test metrics without training."""
    start_time = time.time()
    data_root, exp_dir = Path(data_root), Path(exp_dir)
    config_path = exp_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    part = str(cfg.get("part", part))
    if part not in {"positive", "negative"}:
        raise ValueError(f"Invalid polarity part: {part}")

    pulse_list = list(map(int, cfg.get("pulse_list", DEFAULT_PULSE_LIST)))
    u_start = int(cfg.get("u_start", 1))
    u_end = int(cfg.get("u_end", 41))
    drop_first_class = bool(cfg.get("drop_first_class", True))
    soc_col = str(cfg.get("soc_col", "SOC"))
    soh_col = str(cfg.get("soh_col", "SOH"))
    use_pt_as_feature = bool(cfg.get("use_pt_as_feature", True))
    normalize_soc = bool(cfg.get("normalize_soc", True))
    zscore_normalize = bool(cfg.get("zscore_normalize", True))
    seed = int(cfg.get("seed", 42))
    test_id_frac = float(cfg.get("test_id_frac", 0.2))
    test_id_count = int(cfg.get("test_id_count", 0))
    val_id_frac = float(cfg.get("val_id_frac", 0.1))
    val_id_count = int(cfg.get("val_id_count", 0))
    batch_size = int(batch_size or cfg.get("batch_size", 128))

    cache_dir = exp_dir / "cache"
    train_kwargs = {
        "data_root": str(data_root),
        "soc_list": list(range(5, 90, 5)),
        "pulse_list": pulse_list,
        "u_start": u_start,
        "u_end": u_end,
        "drop_first_class": drop_first_class,
    }
    test_kwargs = {
        "data_root": str(data_root),
        "pulse_list": pulse_list,
        "u_start": u_start,
        "u_end": u_end,
        "drop_first_class": drop_first_class,
    }

    Xtr_raw, ytr_raw, mtr_raw, _, _ = load_or_build_cache(
        str(cache_dir), "raw_train", build_train_mix_soc_mix_pt, train_kwargs
    )
    Xte_raw, yte_raw, mte_raw, _, _ = load_or_build_cache(
        str(cache_dir), "raw_test", build_test_random_mix_pt, test_kwargs
    )
    Xtr_raw, ytr_raw, mtr_raw = drop_nan_inf_rows(
        Xtr_raw, ytr_raw, mtr_raw, name="RAW_TRAIN"
    )
    Xte_raw, yte_raw, mte_raw = drop_nan_inf_rows(
        Xte_raw, yte_raw, mte_raw, name="RAW_TEST"
    )

    all_ids = pd.concat([mtr_raw["ID"], mte_raw["ID"]], axis=0).astype(str).to_numpy()
    test_ids = np.asarray(
        pick_test_ids(
            all_ids=all_ids,
            test_id_frac=test_id_frac,
            test_id_count=test_id_count,
            seed=seed,
        ),
        dtype=object,
    )
    test_id_set = set(map(str, test_ids))
    unique_all_ids = pd.Series(all_ids).astype(str).drop_duplicates().to_numpy()
    non_test_ids = np.asarray(
        [item for item in unique_all_ids if str(item) not in test_id_set],
        dtype=object,
    )
    val_ids = np.asarray(
        pick_test_ids(
            all_ids=non_test_ids,
            test_id_frac=val_id_frac,
            test_id_count=val_id_count,
            seed=seed + 1,
        ),
        dtype=object,
    )
    val_id_set = set(map(str, val_ids))

    train_mask = ~(
        mtr_raw["ID"].astype(str).isin(test_id_set)
        | mtr_raw["ID"].astype(str).isin(val_id_set)
    ).to_numpy()
    test_mask = mte_raw["ID"].astype(str).isin(test_id_set).to_numpy()

    Xtr = Xtr_raw[train_mask]
    ytr_str = np.asarray(ytr_raw)[train_mask]
    mtr = mtr_raw.loc[train_mask].reset_index(drop=True)
    Xte = Xte_raw[test_mask]
    yte_str = np.asarray(yte_raw)[test_mask]
    mte = mte_raw.loc[test_mask].reset_index(drop=True)

    norm_path = exp_dir / "u41_norm_train_only.npz"
    if norm_path.exists():
        norm = np.load(norm_path)
        u_mean, u_std = norm["u_mean"].astype(np.float64), norm["u_std"].astype(np.float64)
    else:
        u_mean = Xtr.mean(axis=0, keepdims=True)
        u_std = Xtr.std(axis=0, keepdims=True) + 1e-8
        print(f"[WARN] Missing {norm_path}; recomputed normalization from TRAIN.")
    Xte = (Xte - u_mean) / (u_std + 1e-8)

    target_norm_path = exp_dir / "target_norm_train_only.npz"
    if target_norm_path.exists():
        target_norm = np.load(target_norm_path)
        soc_norm = (
            float(np.asarray(target_norm["soc_mean"]).reshape(-1)[0]),
            float(np.asarray(target_norm["soc_std"]).reshape(-1)[0]),
        )
        soh_norm = (
            float(np.asarray(target_norm["soh_mean"]).reshape(-1)[0]),
            float(np.asarray(target_norm["soh_std"]).reshape(-1)[0]),
        )
    else:
        soc_train = mtr[soc_col].astype(float).to_numpy(dtype=np.float64)
        if normalize_soc:
            soc_train /= 100.0
        soh_train = mtr[soh_col].astype(float).to_numpy(dtype=np.float64)
        soc_norm = (float(soc_train.mean()), float(soc_train.std() + 1e-8))
        soh_norm = (float(soh_train.mean()), float(soh_train.std() + 1e-8))
        print(f"[WARN] Missing {target_norm_path}; recomputed target normalization from TRAIN.")

    mapping_path = exp_dir / "label_mapping.json"
    label_encoder = LabelEncoder()
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            label_encoder.classes_ = np.asarray(json.load(f)["classes"], dtype=object)
    else:
        label_encoder.fit(ytr_str)
        print(f"[WARN] Missing {mapping_path}; rebuilt label mapping from TRAIN.")

    known_classes = set(label_encoder.classes_.tolist())
    known_mask = np.asarray([label in known_classes for label in yte_str], dtype=bool)
    if not known_mask.all():
        print(f"[WARN] Removing {int((~known_mask).sum())} test samples with unseen labels.")
        Xte = Xte[known_mask]
        yte_str = yte_str[known_mask]
        mte = mte.loc[known_mask].reset_index(drop=True)
    if len(yte_str) == 0:
        raise RuntimeError("No test samples remain after filtering unknown labels.")
    yte_cls = label_encoder.transform(yte_str)

    if use_pt_as_feature and "pulse_ms" in mtr.columns:
        pt_log = np.log1p(mtr["pulse_ms"].astype(float).to_numpy(dtype=np.float32))
        pt_norm = (float(pt_log.mean()), float(pt_log.std() + 1e-8))
    else:
        pt_norm = (0.0, 1.0)

    ds_te = PolarityPulseDataset(
        X_u=Xte,
        y_cls=yte_cls,
        meta=mte,
        soc_col=soc_col,
        soh_col=soh_col,
        part=part,
        use_pt_as_feature=use_pt_as_feature,
        pt_norm=pt_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
        soc_norm=soc_norm if zscore_normalize else None,
        soh_norm=soh_norm if zscore_normalize else None,
    )
    dl_te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Hier3HeadModel(
        num_classes=len(label_encoder.classes_),
        width=int(cfg.get("width", 32)),
        blocks=int(cfg.get("blocks", 4)),
        drop2d=float(cfg.get("drop2d", 0.0)),
        use_pt_as_feature=use_pt_as_feature,
        head_dropout=float(cfg.get("head_dropout", 0.2)),
    ).to(device)

    checkpoint_path = _find_existing_checkpoint(
        exp_dir, preferred_stage=cfg.get("final_best_stage", "finetune")
    )
    checkpoint = _torch_load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    cls_true, cls_pred = [], []
    soc_true_z, soc_pred_z = [], []
    soh_true_z, soh_pred_z = [], []

    for x3, pt, y_cls, soc, soh in dl_te:
        x3, pt = x3.to(device), pt.to(device)
        logits, soc_pred, _, _, soh_pred, _ = model(
            x_img=x3, x_pt=pt, soc_tf=None, n_mc=int(n_mc)
        )
        cls_true.append(y_cls.numpy().reshape(-1))
        cls_pred.append(logits.argmax(dim=1).cpu().numpy().reshape(-1))
        soc_true_z.append(soc.numpy().reshape(-1))
        soc_pred_z.append(soc_pred.cpu().numpy().reshape(-1))
        soh_true_z.append(soh.numpy().reshape(-1))
        soh_pred_z.append(soh_pred.cpu().numpy().reshape(-1))

    cls_true = np.concatenate(cls_true)
    cls_pred = np.concatenate(cls_pred)
    soc_true_z = np.concatenate(soc_true_z)
    soc_pred_z = np.concatenate(soc_pred_z)
    soh_true_z = np.concatenate(soh_true_z)
    soh_pred_z = np.concatenate(soh_pred_z)

    soc_true, soh_true = _inverse_eval_targets(
        soc_true_z, soh_true_z, soc_norm, soh_norm, normalize_soc, zscore_normalize
    )
    soc_pred, soh_pred = _inverse_eval_targets(
        soc_pred_z, soh_pred_z, soc_norm, soh_norm, normalize_soc, zscore_normalize
    )
    true_labels = label_encoder.inverse_transform(cls_true)
    pred_labels = label_encoder.inverse_transform(cls_pred)
    true_material = np.asarray([x.split("_")[0] for x in true_labels])
    pred_material = np.asarray([x.split("_")[0] for x in pred_labels])

    soc_abs = np.abs(soc_pred - soc_true)
    soh_abs = np.abs(soh_pred - soh_true)
    soc_ape = soc_abs / (np.abs(soc_true) + 1e-8) * 100.0
    soh_ape = soh_abs / (np.abs(soh_true) + 1e-8) * 100.0

    predictions = pd.DataFrame(
        {
            "true_label": true_labels,
            "pred_label": pred_labels,
            "true_material": true_material,
            "pred_material": pred_material,
            "material_correct": true_material == pred_material,
            "soc_true": soc_true,
            "soc_pred": soc_pred,
            "soh_true": soh_true,
            "soh_pred": soh_pred,
        }
    )
    if "ID" in mte.columns:
        predictions.insert(0, "ID", mte["ID"].astype(str).to_numpy())
    if "pulse_ms" in mte.columns:
        predictions["pulse_ms"] = mte["pulse_ms"].astype(float).to_numpy()

    metrics_dir = exp_dir / "metrics"
    ensure_dir(str(metrics_dir))
    predictions_path = metrics_dir / "retrospective_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    out = {
        "polarity": part,
        "checkpoint_stage": checkpoint_path.parent.name,
        "checkpoint_path": str(checkpoint_path),
        "test_cls_acc": float(np.mean(cls_true == cls_pred)),
        "test_material_acc": float(np.mean(true_material == pred_material)),
        "test_soc_medae_raw": float(np.median(soc_abs)),
        "test_soc_mae_raw": float(np.mean(soc_abs)),
        "test_soc_rmse_raw": float(np.sqrt(np.mean((soc_pred - soc_true) ** 2))),
        "test_soc_medape_raw": float(np.median(soc_ape)),
        "test_soc_mape_raw": float(np.mean(soc_ape)),
        "test_soh_medae_raw": float(np.median(soh_abs)),
        "test_soh_mae_raw": float(np.mean(soh_abs)),
        "test_soh_rmse_raw": float(np.sqrt(np.mean((soh_pred - soh_true) ** 2))),
        "test_soh_medape_raw": float(np.median(soh_ape)),
        "test_soh_mape_raw": float(np.mean(soh_ape)),
        "n_test": int(len(ds_te)),
        "num_classes": int(len(label_encoder.classes_)),
        "n_mc": int(n_mc),
        "device": device,
        "elapsed_sec": float(time.time() - start_time),
        "predictions_path": str(predictions_path),
    }
    pd.DataFrame([out]).to_csv(
        metrics_dir / "retrospective_final_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _save_json(metrics_dir / "retrospective_final_metrics.json", out)

    print(
        f"[RESULT] {exp_dir.name}: "
        f"fine={out['test_cls_acc']:.4f}, "
        f"material={out['test_material_acc']:.4f}, "
        f"SOC MedAE={out['test_soc_medae_raw']:.4f}, "
        f"SOH MedAE={out['test_soh_medae_raw']:.4f}"
    )
    return out


def run_polarity_summary_only(
    data_root: str | Path,
    output_root: str | Path,
    config: str = "all",
    n_mc: int = 500,
) -> pd.DataFrame:
    output_root = Path(output_root)
    selected = POLARITY_CONFIGS if config == "all" else {config: POLARITY_CONFIGS[config]}
    rows = []
    for config_name, part in selected.items():
        print("\n" + "=" * 90)
        print(f"[SUMMARY-ONLY] {config_name} | {part}")
        print("=" * 90)
        out = evaluate_existing_polarity_checkpoint(
            data_root=data_root,
            exp_dir=output_root / config_name,
            part=part,
            n_mc=n_mc,
        )
        rows.append({"config": config_name, "part": part, **out})

    summary = _add_summary_columns(pd.DataFrame(rows))
    summary.to_csv(
        output_root / "pulse_polarity_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _save_json(
        output_root / "pulse_polarity_sensitivity_summary.json",
        summary.to_dict(orient="records"),
    )
    return summary

# =============================================================================
# Polarity sensitivity runner
# =============================================================================

def _add_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    """Add percentage-form columns without changing the raw metrics."""
    if summary.empty:
        return summary
    summary = summary.copy()
    if "test_cls_acc" in summary:
        summary["fine_grained_acc_pct"] = summary["test_cls_acc"].astype(float) * 100.0
    if "test_material_acc" in summary:
        summary["material_acc_pct"] = summary["test_material_acc"].astype(float) * 100.0
    if "test_soc_medae_raw" in summary:
        summary["soc_medae_pct"] = summary["test_soc_medae_raw"].astype(float)
    if "test_soh_medae_raw" in summary:
        summary["soh_medae_pct"] = summary["test_soh_medae_raw"].astype(float)
    if "test_soc_medape_raw" in summary:
        summary["soc_medape_pct"] = summary["test_soc_medape_raw"].astype(float)
    if "test_soh_medape_raw" in summary:
        summary["soh_medape_pct"] = summary["test_soh_medape_raw"].astype(float)
    return summary


def run_pulse_polarity_sensitivity(
    data_root: str | Path,
    output_root: str | Path,
    smoke: bool = False,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Run positive-only and negative-only pulse polarity sensitivity experiments.

    Full bidirectional result should be taken from proposed_framework full setting.
    This script only trains/evaluates positive-only and negative-only models.
    Best checkpoints are selected by validation score, not test score.
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if smoke:
        configs = {
            "SMOKE_positive_only": "positive",
            "SMOKE_negative_only": "negative",
        }

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
        configs = POLARITY_CONFIGS
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

    for config_name, part in configs.items():
        exp_dir = output_root / config_name

        print("\n" + "=" * 90)
        print(f"[RUN] Pulse polarity configuration: {config_name}")
        print(f"[RUN] part: {part}")
        print(f"[RUN] pulse_list: {pulse_list}")
        print(f"[RUN] Output directory: {exp_dir}")
        print("=" * 90)

        out = run_polarity_experiment(
            data_root=data_root,
            pulse_list=pulse_list,
            part=part,
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

            **run_kwargs,
        )

        row = {
            "config": config_name,
            "part": part,
            "pulse_widths_ms": ",".join(map(str, pulse_list)),
            "num_pulse_widths": len(pulse_list),
            **out,
        }

        rows.append(row)

        partial = pd.DataFrame(rows)
        partial = _add_summary_columns(partial)

        partial.to_csv(
            output_root / "pulse_polarity_sensitivity_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )

        _save_json(
            output_root / "pulse_polarity_sensitivity_partial.json",
            rows,
        )

    summary = pd.DataFrame(rows)
    summary = _add_summary_columns(summary)

    summary.to_csv(
        output_root / "pulse_polarity_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _save_json(
        output_root / "pulse_polarity_sensitivity_summary.json",
        summary.to_dict(orient="records"),
    )

    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only evaluate existing checkpoints; do not train.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="all",
        choices=["all", "positive_only", "negative_only"],
    )
    parser.add_argument(
        "--n-mc",
        type=int,
        default=500,
        help="Monte Carlo samples used during retrospective evaluation.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override the current data directory.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else PROJECT_ROOT / "data"
    output_root = (
        PROJECT_ROOT
        / "results"
        / "measurement_sensitivity"
        / "pulse_polarity"
    )

    if args.summary_only:
        summary = run_polarity_summary_only(
            data_root=data_root,
            output_root=output_root,
            config=args.config,
            n_mc=args.n_mc,
        )
    elif args.config == "all":
        summary = run_pulse_polarity_sensitivity(
            data_root=data_root,
            output_root=output_root,
            smoke=False,
            resume=True,
        )
    else:
        part = POLARITY_CONFIGS[args.config]
        out = run_polarity_experiment(
            data_root=data_root,
            pulse_list=DEFAULT_PULSE_LIST,
            part=part,
            exp_dir=output_root / args.config,
            batch_size=128,
            max_epochs=400,
            early_stopping=False,
            resume=True,
            width=32,
            blocks=4,
            head_dropout=0.2,
            two_stage=True,
            stage1_epochs=200,
            stage2_epochs=200,
            finetune_epochs=30,
            final_best_stage="finetune",
        )
        summary = _add_summary_columns(pd.DataFrame([{"config": args.config, **out}]))
        summary.to_csv(
            output_root / f"pulse_polarity_sensitivity_{args.config}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print("\n[SUMMARY]")
    print(summary)


if __name__ == "__main__":
    main()
