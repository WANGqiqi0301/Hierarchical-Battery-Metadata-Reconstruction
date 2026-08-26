# proposed_framework/run_proposed_framework.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import sys
import os
import json
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader


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
from proposed_framework.data.pulse_dataset import HierPulseDataset
from proposed_framework.models.hierarchical_model import Hier3HeadModel
from proposed_framework.training.trainer import train_one_epoch
from proposed_framework.training.evaluator import eval_one_epoch


# =============================================================================
# Default experiment directory
# =============================================================================

DEFAULT_EXP_DIR = PROJECT_ROOT / "results" / "proposed_framework"


def _torch_load(path: str | Path, map_location: str):
    """
    Compatibility wrapper for torch.load across PyTorch versions.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def run_experiment(
    data_root: str,
    pulse_list: List[int],
    c_rate_combo: Optional[List[int]] = None,
    # U window
    u_start: int = 1,
    u_end: int = 41,
    drop_first_class: bool = True,
    # targets in metadata
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
    # model settings
    width: int = 32,
    blocks: int = 4,
    drop2d: float = 0.0,
    head_dropout: float = 0.2,
    # loss weights
    w_cls: float = 1.0,
    w_soc: float = 1.0,
    w_soh: float = 1.0,
    # group split config
    test_id_frac: float = 0.2,
    test_id_count: int = 0,
    # validation split config
    # IMPORTANT: TEST IDs are selected first exactly as before.
    # Validation IDs are then sampled only from the remaining TRAIN ID pool.
    val_id_frac: float = 0.1,
    val_id_count: int = 0,
    val_seed_offset: int = 1000,
    # target normalization
    normalize_soc: bool = True,
    zscore_normalize: bool = True,
    # two-stage training
    two_stage: bool = True,
    stage1_epochs: int = 200,
    stage2_epochs: int = 200,
    finetune_epochs: int = 30,
    freeze_encoder_stage2: bool = True,
    freeze_mat_soc_stage2: bool = True,
    # fixed prior bin weighting
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
    # scoring and final model selection
    alpha_score: float = 0.1,
    final_best_stage: str = "finetune",
    # output
    exp_dir: str | Path = DEFAULT_EXP_DIR,
):
    """
    Run the proposed hierarchical probabilistic framework.

    This function performs:
    1. Data loading and caching.
    2. ID-level group split.
    3. Train-only U1-U41 normalization.
    4. Material-capacity label encoding.
    5. Dataset and dataloader construction.
    6. Hierarchical conditional-flow model training.
    7. Final evaluation and metric saving.

    Parameters
    ----------
    data_root:
        Root folder containing material-capacity subfolders.

    pulse_list:
        Pulse widths used as input.

    exp_dir:
        Output directory for cache, checkpoints, logs, splits and metrics.

    Returns
    -------
    dict
        Final evaluation metrics.
    """
    start_time = time.time()

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
    print(f"[INFO] Device: {device}")

    soc_list = list(range(5, 90, 5))

    config_to_save = {
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
        "drop2d": drop2d,
        "head_dropout": head_dropout,
        "w_cls": w_cls,
        "w_soc": w_soc,
        "w_soh": w_soh,
        "test_id_frac": test_id_frac,
        "test_id_count": test_id_count,
        "val_id_frac": val_id_frac,
        "val_id_count": val_id_count,
        "val_seed_offset": val_seed_offset,
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

    with open(exp_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, ensure_ascii=False, indent=2)

    # =========================================================================
    # 1. Load and cache raw train/test datasets
    # =========================================================================

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
    # 2. Pick TEST IDs first, then split VALIDATION from the remaining TRAIN IDs
    # =========================================================================

    def _pick_val_ids_from_train_ids(
        train_ids: np.ndarray,
        val_id_frac: float = 0.1,
        val_id_count: int = 0,
        seed: int = 1042,
    ) -> np.ndarray:
        """
        Pick validation IDs only from the already-created training ID pool.

        This preserves the original TEST split exactly:
        1) pick TEST IDs using the same pick_test_ids(...) call as before;
        2) define the original training ID pool as IDs not in TEST;
        3) sample VALIDATION IDs only from that original training pool.
        """
        ids = np.array(pd.Series(train_ids).astype(str).unique(), dtype=object)
        n_all = len(ids)

        if n_all <= 1:
            raise RuntimeError("Not enough training IDs to create a validation split.")

        rng = np.random.RandomState(seed)
        rng.shuffle(ids)

        if val_id_count and val_id_count > 0:
            n_val = int(min(max(1, val_id_count), n_all - 1))
        else:
            n_val = int(max(1, round(n_all * float(val_id_frac))))
            n_val = min(n_val, n_all - 1)

        return ids[:n_val]

    def _apply_train_val_test_split_after_fixed_test(
        Xtr_raw: np.ndarray,
        ytr_raw: np.ndarray,
        mtr_raw: pd.DataFrame,
        Xte_raw: np.ndarray,
        yte_raw: np.ndarray,
        mte_raw: pd.DataFrame,
        test_ids: np.ndarray,
        val_ids: np.ndarray,
    ):
        """
        Three-way split with fixed original TEST IDs.

        TRAIN:
          grid-SOC training-sheet samples whose IDs are not TEST or VAL.
        VALIDATION:
          TEST_RANDOM samples whose IDs are selected from the original TRAIN ID pool.
        TEST:
          TEST_RANDOM samples whose IDs are exactly the original TEST IDs.
        """
        if "ID" not in mtr_raw.columns or "ID" not in mte_raw.columns:
            raise RuntimeError("Metadata must contain an 'ID' column for group split.")

        test_set = set(pd.Series(test_ids).astype(str).tolist())
        val_set = set(pd.Series(val_ids).astype(str).tolist())

        overlap = test_set & val_set
        if overlap:
            raise RuntimeError(
                f"TEST IDs and VAL IDs overlap: {list(sorted(overlap))[:10]}"
            )

        tr_ids = pd.Series(mtr_raw["ID"]).astype(str).to_numpy()
        te_ids = pd.Series(mte_raw["ID"]).astype(str).to_numpy()

        train_mask = ~pd.Series(tr_ids).isin(test_set | val_set).to_numpy()
        val_mask = pd.Series(te_ids).isin(val_set).to_numpy()
        test_mask = pd.Series(te_ids).isin(test_set).to_numpy()

        Xtr = Xtr_raw[train_mask]
        ytr = ytr_raw[train_mask]
        mtr = mtr_raw.loc[train_mask].reset_index(drop=True)

        Xval = Xte_raw[val_mask]
        yval = yte_raw[val_mask]
        mval = mte_raw.loc[val_mask].reset_index(drop=True)

        Xte = Xte_raw[test_mask]
        yte = yte_raw[test_mask]
        mte = mte_raw.loc[test_mask].reset_index(drop=True)

        train_id_set = set(mtr["ID"].astype(str).unique())
        val_id_set = set(mval["ID"].astype(str).unique())
        test_id_set = set(mte["ID"].astype(str).unique())

        if train_id_set & val_id_set:
            raise RuntimeError("ID leakage between TRAIN and VAL.")
        if train_id_set & test_id_set:
            raise RuntimeError("ID leakage between TRAIN and TEST.")
        if val_id_set & test_id_set:
            raise RuntimeError("ID leakage between VAL and TEST.")

        return Xtr, ytr, mtr, Xval, yval, mval, Xte, yte, mte

    all_ids = pd.concat(
        [mtr_raw["ID"], mte_raw["ID"]],
        axis=0,
    ).astype(str).to_numpy()

    # Important: this is exactly the same TEST-ID selection call/order as before.
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

    test_set = set(pd.Series(test_ids).astype(str).tolist())

    # Validation IDs are sampled only from IDs that still have grid-SOC training samples.
    original_train_ids = np.array(
        [
            train_id
            for train_id in pd.Series(mtr_raw["ID"]).astype(str).unique()
            if str(train_id) not in test_set
        ],
        dtype=object,
    )

    val_seed = int(seed) + int(val_seed_offset)
    val_ids = _pick_val_ids_from_train_ids(
        train_ids=original_train_ids,
        val_id_frac=val_id_frac,
        val_id_count=val_id_count,
        seed=val_seed,
    )

    if val_id_count and val_id_count > 0:
        val_split_name = f"valIDs_seed{val_seed}_n{val_id_count}"
    else:
        val_split_name = f"valIDs_seed{val_seed}_frac{val_id_frac}"

    with open(splits_dir / f"{val_split_name}.txt", "w", encoding="utf-8") as f:
        for val_id in val_ids:
            f.write(str(val_id) + "\n")

    with open(splits_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_split_is_original_first_split": True,
                "test_split_name": split_name,
                "val_split_name": val_split_name,
                "seed_test": int(seed),
                "seed_val": int(val_seed),
                "test_id_frac": float(test_id_frac),
                "test_id_count": int(test_id_count),
                "val_id_frac": float(val_id_frac),
                "val_id_count": int(val_id_count),
                "n_test_ids": int(len(test_ids)),
                "n_original_train_ids_before_val": int(len(original_train_ids)),
                "n_val_ids": int(len(val_ids)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    Xtr, ytr_str, mtr, Xval, yval_str, mval, Xte, yte_str, mte = (
        _apply_train_val_test_split_after_fixed_test(
            Xtr_raw=Xtr_raw,
            ytr_raw=ytr_raw,
            mtr_raw=mtr_raw,
            Xte_raw=Xte_raw,
            yte_raw=yte_raw,
            mte_raw=mte_raw,
            test_ids=test_ids,
            val_ids=val_ids,
        )
    )

    if len(ytr_str) == 0 or len(yval_str) == 0 or len(yte_str) == 0:
        raise RuntimeError("Empty train, validation or test data after applying ID split.")

    print(
        f"[DATA] Final TRAIN samples = {len(ytr_str)} | "
        f"unique IDs = {mtr['ID'].astype(str).nunique()}"
    )
    print(
        f"[DATA] Final VAL   samples = {len(yval_str)} | "
        f"unique IDs = {mval['ID'].astype(str).nunique()} "
        f"(from TEST_RANDOM, IDs drawn from original TRAIN IDs)"
    )
    print(
        f"[DATA] Final TEST  samples = {len(yte_str)} | "
        f"unique IDs = {mte['ID'].astype(str).nunique()} "
        f"(same TEST IDs as the original split)"
    )

    # =========================================================================
    # 3. Normalize U1-U41 using TRAIN statistics only
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
    # 4. Target normalization statistics from TRAIN only
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
    # 5. Encode material-capacity labels using TRAIN classes
    # =========================================================================

    label_encoder = LabelEncoder()
    ytr_cls = label_encoder.fit_transform(ytr_str)

    train_classes = set(label_encoder.classes_.tolist())

    def _keep_known_classes(X, y_str, meta, name: str):
        mask_known = np.array(
            [label in train_classes for label in y_str],
            dtype=bool,
        )

        if not mask_known.all():
            n_removed = int((~mask_known).sum())
            print(
                f"[WARN] Removing {n_removed} {name} samples "
                f"with labels unseen in training."
            )
            X = X[mask_known]
            y_str = y_str[mask_known]
            meta = meta.loc[mask_known].reset_index(drop=True)

        return X, y_str, meta

    Xval, yval_str, mval = _keep_known_classes(
        Xval,
        yval_str,
        mval,
        name="validation",
    )
    Xte, yte_str, mte = _keep_known_classes(
        Xte,
        yte_str,
        mte,
        name="test",
    )

    if len(yval_str) == 0 or len(yte_str) == 0:
        raise RuntimeError(
            "No validation or test samples remain after filtering unknown labels."
        )

    yval_cls = label_encoder.transform(yval_str)
    yte_cls = label_encoder.transform(yte_str)

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    with open(exp_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": class_names,
                "test_split_name": split_name,
                "val_split_name": val_split_name,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[LABEL] Number of material-capacity classes: {num_classes}")
    print(f"[LABEL] Classes: {class_names}")

    # =========================================================================
    # 6. Pulse-width feature normalization from TRAIN only
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
        c_rate_combo=c_rate_combo,
    )

    ds_val = HierPulseDataset(
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
        c_rate_combo=c_rate_combo,
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
        c_rate_combo=c_rate_combo,
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
    # 8. Model and losses
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
    # 9. Fixed prior bin weighting, computed from TRAIN only
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
    # 10. Stage helper functions
    # =========================================================================

    def set_trainable(stage: str) -> None:
        """
        Control trainable parameters for different training stages.
        """
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

        return stage_ckpt_dir, last_path, best_path, log_path

    def stage_score(stage: str, metrics: dict) -> float:
        if stage == "stage1_soc":
            return float(metrics["cls_acc"] - 0.3 * metrics["soc_rmse"])

        if stage == "stage2_soh":
            return float(-metrics["soh_rmse"])

        return float(
            metrics["cls_acc"]
            - float(alpha_score) * (metrics["soc_rmse"] + metrics["soh_rmse"])
        )


    def run_stage(
        stage: str,
        epochs: int,
        w_cls_s: float,
        w_soc_s: float,
        w_soh_s: float,
    ):
        """
        Run one training stage and return the best checkpoint path.
        """
        _, last_path, best_path, log_path = stage_paths(stage)

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

            train_score = stage_score(stage, tr)
            val_score = stage_score(stage, va)
            test_score = stage_score(stage, te)
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
                        "train_score": train_score,
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
                        "val_score": val_score,
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
                        "test_score": test_score,
                        "score": score,
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
                f"VAL cls={va['cls_acc']:.4f}, "
                f"SOC MedAPE(raw)={va.get('soc_medape_raw', np.nan):.3f}%, "
                f"SOH MedAPE(raw)={va.get('soh_medape_raw', np.nan):.3f}% | "
                f"TEST cls={te['cls_acc']:.4f}, "
                f"SOC MedAPE(raw)={te.get('soc_medape_raw', np.nan):.3f}%, "
                f"SOH MedAPE(raw)={te.get('soh_medape_raw', np.nan):.3f}% | "
                f"val_score={val_score:.6f}"
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
    # 11. Run training
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
    # 12. Choose final checkpoint
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
        "final_stage": chosen,
        "val_cls_acc": float(va["cls_acc"]),

        "val_soc_rmse": float(va["soc_rmse"]),
        "val_soc_mae": float(va["soc_mae"]),
        "val_soc_mape": float(va["soc_mape"]),
        "val_soc_medape": float(va.get("soc_medape", np.nan)),

        "val_soh_rmse": float(va["soh_rmse"]),
        "val_soh_mae": float(va["soh_mae"]),
        "val_soh_mape": float(va["soh_mape"]),
        "val_soh_medape": float(va.get("soh_medape", np.nan)),

        "val_soc_rmse_raw": float(va["soc_rmse_raw"]),
        "val_soc_mae_raw": float(va["soc_mae_raw"]),
        "val_soc_mape_raw": float(va["soc_mape_raw"]),
        "val_soc_medape_raw": float(va.get("soc_medape_raw", np.nan)),

        "val_soh_rmse_raw": float(va["soh_rmse_raw"]),
        "val_soh_mae_raw": float(va["soh_mae_raw"]),
        "val_soh_mape_raw": float(va["soh_mape_raw"]),
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
        "device": device,
        "elapsed_sec": float(elapsed_sec),
    }

    pd.DataFrame([out]).to_csv(
        metrics_dir / "final_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(metrics_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n[FINAL METRICS]")
    for key, value in out.items():
        print(f"{key}: {value}")

    return out


def main() -> None:
    """
    Default full-setting run.

    Change `data_root` if your data folder is not `code/data`.
    """
    data_root = PROJECT_ROOT / "data"

    run_experiment(
        data_root=str(data_root),
        pulse_list=[30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000],
        u_start=1,
        u_end=41,
        drop_first_class=True,
        soc_col="SOC",
        soh_col="SOH",
        use_pt_as_feature=True,
        batch_size=128,
        lr=3e-4,
        weight_decay=1e-4,
        grad_clip=5.0,
        max_epochs=200,
        early_stopping=True,
        patience=150,
        resume=True,
        num_workers=0,
        seed=42,
        width=32,
        blocks=4,
        drop2d=0.0,
        head_dropout=0.2,
        w_cls=1.0,
        w_soc=1.0,
        w_soh=1.0,
        test_id_frac=0.2,
        test_id_count=0,
        val_id_frac=0.1,
        val_id_count=0,
        val_seed_offset=1000,
        normalize_soc=True,
        zscore_normalize=True,
        two_stage=True,
        stage1_epochs=200,
        stage2_epochs=200,
        finetune_epochs=30,
        freeze_encoder_stage2=True,
        freeze_mat_soc_stage2=True,
        use_soc_prior_weighting=True,
        use_soh_prior_weighting=True,
        soc_prior_bins=10,
        soh_prior_bins=10,
        soc_prior_low=0.5,
        soc_prior_mid=1.0,
        soc_prior_high=0.8,
        soh_prior_low=0.8,
        soh_prior_mid=1.0,
        soh_prior_high=0.9,
        alpha_score=0.1,
        final_best_stage="finetune",
        exp_dir=DEFAULT_EXP_DIR,
    )


if __name__ == "__main__":
    main()