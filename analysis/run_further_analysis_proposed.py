# -*- coding: utf-8 -*-
"""
run_further_analysis_proposed.py

Generate per-sample prediction tables for the proposed framework.

This version matches the train/validation/test split and train-only
normalization used by proposed_framework/run_proposed_framework.py.

Outputs:
- further_analysis/tables/train_predictions_for_scatter.csv
- further_analysis/tables/val_predictions_per_sample.csv
- further_analysis/tables/test_predictions_per_sample.csv
- further_analysis/tables/proposed_method_summary.csv
- further_analysis/tables/further_analysis_inference_config.json

The proposed-method inference is performed in separated hierarchical steps:
- predicted material probability
- SOC flow with N_MC_SOC samples
- SOH flow with N_MC_SOH samples conditioned on predicted SOC
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import proposed_framework.run_proposed_framework as M


# =============================================================================
# Config
# =============================================================================
DEFAULT_EXP_DIR = getattr(
    M,
    "DEFAULT_EXP_DIR",
    PROJECT_ROOT / "results" / "proposed_framework",
)
EXP_DIR = str(DEFAULT_EXP_DIR)
DATA_ROOT = str(PROJECT_ROOT / "data")

BATCH_SIZE = 512
NUM_WORKERS = 0
SEED = 42
VAL_SEED_OFFSET = 1000
TEST_ID_FRAC = 0.2
TEST_ID_COUNT = 0
VAL_ID_FRAC = 0.1
VAL_ID_COUNT = 0

SOC_COL = "SOC"
SOH_COL = "SOH"
ID_COL = "ID"
PT_COL = "pulse_ms"
USE_PT_AS_FEATURE = True
NORMALIZE_SOC = True
ZSCORE_NORMALIZE = True
PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]
DROP_FIRST_CLASS = True
U_START = 1
U_END = 41

WIDTH = 32
BLOCKS = 4
DROP2D = 0.0
HEAD_DROPOUT = 0.2

N_MC_SOC = 500
N_MC_SOH = 500


# =============================================================================
# Utility functions
# =============================================================================
def torch_load(path: str | Path, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def medae(a, b):
    return float(np.median(np.abs(np.asarray(a) - np.asarray(b))))

def resolve_exp_dir() -> str:
    # run_config_path = os.path.join(EXP_DIR, "run_config.json")
    # if os.path.exists(run_config_path):
    #     with open(run_config_path, "r", encoding="utf-8") as f:
    #         cfg = json.load(f)
    #     return str(cfg.get("exp_dir", EXP_DIR))
    return str(EXP_DIR)
# def resolve_exp_dir() -> str:
#     return str(PROJECT_ROOT / "results" / "proposed_framework")


def resolve_checkpoint_path(exp_dir: str) -> str:
    final_metrics_path = os.path.join(exp_dir, "metrics", "final_metrics.json")
    run_config_path = os.path.join(exp_dir, "run_config.json")

    final_stage = None
    if os.path.exists(final_metrics_path):
        with open(final_metrics_path, "r", encoding="utf-8") as f:
            final_stage = json.load(f).get("final_stage")

    if final_stage is None and os.path.exists(run_config_path):
        with open(run_config_path, "r", encoding="utf-8") as f:
            final_stage = json.load(f).get("final_best_stage")

    candidates = []
    if final_stage:
        candidates.append(os.path.join(exp_dir, "checkpoints", str(final_stage), "best.pt"))
    candidates.extend([
        os.path.join(exp_dir, "checkpoints", "finetune", "best.pt"),
        os.path.join(exp_dir, "checkpoints", "stage2_soh", "best.pt"),
        os.path.join(exp_dir, "checkpoints", "stage1_soc", "best.pt"),
        os.path.join(exp_dir, "checkpoints", "single", "best.pt"),
    ])

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("No valid checkpoint was found under the experiment directory.")


def load_run_config(exp_dir: str) -> dict:
    path = os.path.join(exp_dir, "run_config.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_u41_train_only_norm(exp_dir: str, Xtr: np.ndarray, Xval: np.ndarray, Xte: np.ndarray):
    norm_path = os.path.join(exp_dir, "u41_norm_train_only.npz")
    if os.path.exists(norm_path):
        obj = np.load(norm_path)
        u_mean = obj["u_mean"].astype(np.float64)
        u_std = obj["u_std"].astype(np.float64)
        print(f"[NORM] Loaded train-only U(41) stats: {norm_path}")
    else:
        u_mean = Xtr.mean(axis=0, keepdims=True)
        u_std = Xtr.std(axis=0, keepdims=True) + 1e-8
        np.savez_compressed(
            norm_path,
            u_mean=u_mean.astype(np.float32),
            u_std=u_std.astype(np.float32),
        )
        print(f"[NORM] WARNING: norm file missing; recomputed and saved: {norm_path}")

    return (
        (Xtr - u_mean) / (u_std + 1e-8),
        (Xval - u_mean) / (u_std + 1e-8),
        (Xte - u_mean) / (u_std + 1e-8),
    )


def inverse_targets(
    soc_z: np.ndarray,
    soh_z: np.ndarray,
    soc_norm: Optional[Tuple[float, float]] = None,
    soh_norm: Optional[Tuple[float, float]] = None,
    normalize_soc: bool = True,
    zscore_normalize: bool = True,
):
    soc = np.asarray(soc_z, dtype=np.float64).copy()
    soh = np.asarray(soh_z, dtype=np.float64).copy()

    if zscore_normalize:
        if soc_norm is None or soh_norm is None:
            raise RuntimeError("zscore_normalize=True requires soc_norm and soh_norm.")
        soc = soc * float(soc_norm[1]) + float(soc_norm[0])
        soh = soh * float(soh_norm[1]) + float(soh_norm[0])

    if normalize_soc:
        soc = soc * 100.0

    soh = soh * 100.0

    return soc, soh


def pick_val_ids_from_train_ids(
    train_ids: np.ndarray,
    val_id_frac: float = 0.1,
    val_id_count: int = 0,
    seed: int = 1042,
) -> np.ndarray:
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


def apply_train_val_test_split_after_fixed_test(
    Xtr_raw: np.ndarray,
    ytr_raw: np.ndarray,
    mtr_raw: pd.DataFrame,
    Xte_raw: np.ndarray,
    yte_raw: np.ndarray,
    mte_raw: pd.DataFrame,
    test_ids: np.ndarray,
    val_ids: np.ndarray,
):
    test_set = set(pd.Series(test_ids).astype(str).tolist())
    val_set = set(pd.Series(val_ids).astype(str).tolist())

    overlap = test_set & val_set
    if overlap:
        raise RuntimeError(f"TEST IDs and VAL IDs overlap: {list(sorted(overlap))[:10]}")

    tr_ids = pd.Series(mtr_raw[ID_COL]).astype(str).to_numpy()
    te_ids = pd.Series(mte_raw[ID_COL]).astype(str).to_numpy()

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

    return Xtr, ytr, mtr, Xval, yval, mval, Xte, yte, mte


def model_uses_pt(model: torch.nn.Module) -> bool:
    if hasattr(model, "use_pt"):
        return bool(getattr(model, "use_pt"))
    if hasattr(model, "use_pt_as_feature"):
        return bool(getattr(model, "use_pt_as_feature"))
    return False


def first_linear_in_features(module: torch.nn.Module) -> Optional[int]:
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            return int(m.in_features)
    return None


@torch.no_grad()
def predict_material_prob(model: torch.nn.Module, x3: torch.Tensor, pt: torch.Tensor):
    z = model.encoder(x3)
    use_pt = model_uses_pt(model)
    in_features = first_linear_in_features(model.head_mat)
    z_dim = int(z.shape[1])
    pt_dim = int(pt.shape[1]) if pt is not None and pt.ndim == 2 else 0

    if use_pt and in_features == z_dim + pt_dim:
        logits = model.head_mat(torch.cat([z, pt], dim=1))
    elif in_features == z_dim:
        logits = model.head_mat(z)
    else:
        raise RuntimeError(
            "Cannot determine material-head input dimension. "
            f"z_dim={z_dim}, pt_dim={pt_dim}, head_first_linear_in_features={in_features}."
        )

    p = torch.softmax(logits, dim=1)
    return z, logits, p


@torch.no_grad()
def sample_flow_mean_1d(flow: torch.nn.Module, context: torch.Tensor, n_mc: int) -> torch.Tensor:
    n_mc = int(n_mc)
    batch_size = int(context.size(0))
    samples = flow.sample(context=context, num_samples=n_mc)

    if samples.ndim == 3:
        if samples.shape[0] == n_mc and samples.shape[1] == batch_size:
            return samples.mean(dim=0).squeeze(-1).view(-1)
        if samples.shape[0] == batch_size and samples.shape[1] == n_mc:
            return samples.mean(dim=1).squeeze(-1).view(-1)
        samples = samples.reshape(n_mc, batch_size, 1)
        return samples.mean(dim=0).squeeze(-1).view(-1)

    if samples.ndim == 2:
        samples = samples.view(n_mc, batch_size, 1)
        return samples.mean(dim=0).squeeze(-1).view(-1)

    raise RuntimeError(f"Unexpected flow sample shape: {tuple(samples.shape)}")


@torch.no_grad()
def infer_soc_given_p(model, x3, pt, p_used, n_mc: int):
    if hasattr(model, "infer_soc_given_p"):
        return model.infer_soc_given_p(x_img=x3, x_pt=pt, p_used=p_used, n_mc=int(n_mc))

    z = model.encoder(x3)
    if model_uses_pt(model):
        cond_soc = torch.cat([z, p_used, pt], dim=1)
    else:
        cond_soc = torch.cat([z, p_used], dim=1)

    soc_pred = sample_flow_mean_1d(model.soc_flow, cond_soc, n_mc=int(n_mc))
    return z, cond_soc, soc_pred.view(-1)


@torch.no_grad()
def infer_soh_given_p_and_soc(model, x3, pt, p_used, soc_val, n_mc: int):
    if hasattr(model, "infer_soh_given_p_and_soc"):
        return model.infer_soh_given_p_and_soc(
            x_img=x3,
            x_pt=pt,
            p_used=p_used,
            soc_val=soc_val,
            n_mc=int(n_mc),
        )

    z = model.encoder(x3)
    soc_val = soc_val.view(-1, 1)
    if model_uses_pt(model):
        cond_soh = torch.cat([z, p_used, soc_val, pt], dim=1)
    else:
        cond_soh = torch.cat([z, p_used, soc_val], dim=1)

    soh_pred = sample_flow_mean_1d(model.soh_flow, cond_soh, n_mc=int(n_mc))
    return cond_soh, soh_pred.view(-1)


def build_datasets():
    exp_dir = resolve_exp_dir()
    run_cfg = load_run_config(exp_dir)
    data_root = str(run_cfg.get("data_root", DATA_ROOT))
    pulse_list = list(map(int, run_cfg.get("pulse_list", PULSE_LIST)))
    u_start = int(run_cfg.get("u_start", U_START))
    u_end = int(run_cfg.get("u_end", U_END))
    drop_first_class = bool(run_cfg.get("drop_first_class", DROP_FIRST_CLASS))
    seed = int(run_cfg.get("seed", SEED))
    val_seed_offset = int(run_cfg.get("val_seed_offset", VAL_SEED_OFFSET))
    test_id_frac = float(run_cfg.get("test_id_frac", TEST_ID_FRAC))
    test_id_count = int(run_cfg.get("test_id_count", TEST_ID_COUNT))
    val_id_frac = float(run_cfg.get("val_id_frac", VAL_ID_FRAC))
    val_id_count = int(run_cfg.get("val_id_count", VAL_ID_COUNT))
    c_rate_combo = run_cfg.get("c_rate_combo", None)

    cache_dir = os.path.join(exp_dir, "cache")
    soc_list = list(range(5, 90, 5))

    train_kwargs = dict(
        data_root=data_root,
        soc_list=soc_list,
        pulse_list=pulse_list,
        u_start=u_start,
        u_end=u_end,
        drop_first_class=drop_first_class,
    )
    test_kwargs = dict(
        data_root=data_root,
        pulse_list=pulse_list,
        u_start=u_start,
        u_end=u_end,
        drop_first_class=drop_first_class,
    )

    Xtr_raw, ytr_raw, mtr_raw, _, _ = M.load_or_build_cache(
        cache_dir, "raw_train", M.build_train_mix_soc_mix_pt, train_kwargs
    )
    Xte_raw, yte_raw, mte_raw, _, _ = M.load_or_build_cache(
        cache_dir, "raw_test", M.build_test_random_mix_pt, test_kwargs
    )

    Xtr_raw, ytr_raw, mtr_raw = M.drop_nan_inf_rows(Xtr_raw, ytr_raw, mtr_raw, "RAW_TRAIN")
    Xte_raw, yte_raw, mte_raw = M.drop_nan_inf_rows(Xte_raw, yte_raw, mte_raw, "RAW_TEST")

    all_ids = pd.concat([mtr_raw[ID_COL], mte_raw[ID_COL]], axis=0).astype(str).to_numpy()
    test_ids = M.pick_test_ids(
        all_ids=all_ids,
        test_id_frac=test_id_frac,
        test_id_count=test_id_count,
        seed=seed,
    )

    test_set = set(pd.Series(test_ids).astype(str).tolist())
    original_train_ids = np.array(
        [
            train_id
            for train_id in pd.Series(mtr_raw[ID_COL]).astype(str).unique()
            if str(train_id) not in test_set
        ],
        dtype=object,
    )

    val_seed = int(seed) + int(val_seed_offset)
    val_ids = pick_val_ids_from_train_ids(
        train_ids=original_train_ids,
        val_id_frac=val_id_frac,
        val_id_count=val_id_count,
        seed=val_seed,
    )

    Xtr, ytr_str, mtr, Xval, yval_str, mval, Xte, yte_str, mte = (
        apply_train_val_test_split_after_fixed_test(
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

    Xtr, Xval, Xte = apply_u41_train_only_norm(exp_dir, Xtr, Xval, Xte)

    soc_tr = mtr[SOC_COL].astype(float).to_numpy(dtype=np.float64)
    if NORMALIZE_SOC:
        soc_tr = soc_tr / 100.0
    soc_norm = (float(soc_tr.mean()), float(soc_tr.std() + 1e-8))

    soh_tr = mtr[SOH_COL].astype(float).to_numpy(dtype=np.float64)
    soh_norm = (float(soh_tr.mean()), float(soh_tr.std() + 1e-8))

    if USE_PT_AS_FEATURE and PT_COL in mtr.columns:
        pt_train = mtr[PT_COL].astype(float).to_numpy(dtype=np.float32)
        pt_log = np.log1p(pt_train)
        pt_norm = (float(pt_log.mean()), float(pt_log.std() + 1e-8))
    else:
        pt_norm = (0.0, 1.0)

    le = LabelEncoder()
    ytr_cls = le.fit_transform(ytr_str)
    train_classes = set(le.classes_.tolist())

    def keep_known(X, y_str, meta, name):
        mask = np.array([label in train_classes for label in y_str], dtype=bool)
        if not mask.all():
            print(f"[WARN] Removing {int((~mask).sum())} {name} samples with unseen labels.")
            X = X[mask]
            y_str = y_str[mask]
            meta = meta.loc[mask].reset_index(drop=True)
        return X, y_str, meta

    Xval, yval_str, mval = keep_known(Xval, yval_str, mval, "validation")
    Xte, yte_str, mte = keep_known(Xte, yte_str, mte, "test")

    yval_cls = le.transform(yval_str)
    yte_cls = le.transform(yte_str)

    ds_tr = M.HierPulseDataset(
        X_u=Xtr,
        y_cls=ytr_cls,
        meta=mtr,
        soc_col=SOC_COL,
        soh_col=SOH_COL,
        use_pt_as_feature=USE_PT_AS_FEATURE,
        pt_norm=pt_norm,
        normalize_soc=NORMALIZE_SOC,
        zscore_normalize=ZSCORE_NORMALIZE,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
        c_rate_combo=c_rate_combo,
    )
    ds_val = M.HierPulseDataset(
        X_u=Xval,
        y_cls=yval_cls,
        meta=mval,
        soc_col=SOC_COL,
        soh_col=SOH_COL,
        use_pt_as_feature=USE_PT_AS_FEATURE,
        pt_norm=pt_norm,
        normalize_soc=NORMALIZE_SOC,
        zscore_normalize=ZSCORE_NORMALIZE,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
        c_rate_combo=c_rate_combo,
    )
    ds_te = M.HierPulseDataset(
        X_u=Xte,
        y_cls=yte_cls,
        meta=mte,
        soc_col=SOC_COL,
        soh_col=SOH_COL,
        use_pt_as_feature=USE_PT_AS_FEATURE,
        pt_norm=pt_norm,
        normalize_soc=NORMALIZE_SOC,
        zscore_normalize=ZSCORE_NORMALIZE,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
        c_rate_combo=c_rate_combo,
    )

    info = {
        "exp_dir": exp_dir,
        "run_cfg": run_cfg,
        "label_encoder": le,
        "num_classes": int(len(le.classes_)),
        "soc_norm": soc_norm,
        "soh_norm": soh_norm,
        "n_train": int(len(ds_tr)),
        "n_val": int(len(ds_val)),
        "n_test": int(len(ds_te)),
    }

    print(f"[DATA] Train samples: {len(ds_tr)}")
    print(f"[DATA] Val samples: {len(ds_val)}")
    print(f"[DATA] Test samples: {len(ds_te)}")
    print(f"[DATA] Num classes: {len(le.classes_)}")

    return ds_tr, ds_val, ds_te, mtr, mval, mte, info


def load_model(exp_dir: str, num_classes: int, device: str, run_cfg: dict):
    width = int(run_cfg.get("width", WIDTH))
    blocks = int(run_cfg.get("blocks", BLOCKS))
    drop2d = float(run_cfg.get("drop2d", DROP2D))
    head_dropout = float(run_cfg.get("head_dropout", HEAD_DROPOUT))

    model = M.Hier3HeadModel(
        num_classes=int(num_classes),
        width=width,
        blocks=blocks,
        drop2d=drop2d,
        use_pt_as_feature=USE_PT_AS_FEATURE,
        head_dropout=head_dropout,
    ).to(device)

    ckpt_path = resolve_checkpoint_path(exp_dir)
    ckpt = torch_load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format.")

    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[MODEL] Loaded checkpoint: {ckpt_path}")
    return model


def infer_rows(model, loader, meta_df, label_encoder, device):
    rows = []
    idx_base = 0
    with torch.no_grad():
        for x3, pt, y_cls, soc_z, soh_z in loader:
            batch_size = int(x3.size(0))
            x3 = x3.to(device)
            pt = pt.to(device)
            y_cls_device = y_cls.to(device).view(-1)
            soc_z = soc_z.to(device).view(-1)
            soh_z = soh_z.to(device).view(-1)

            _, logits_mat, p_pred = predict_material_prob(model, x3, pt)
            _, _, soc_pred_z_t = infer_soc_given_p(
                model=model,
                x3=x3,
                pt=pt,
                p_used=p_pred,
                n_mc=N_MC_SOC,
            )
            _, soh_pred_z_t = infer_soh_given_p_and_soc(
                model=model,
                x3=x3,
                pt=pt,
                p_used=p_pred,
                soc_val=soc_pred_z_t,
                n_mc=N_MC_SOH,
            )

            soc_pred_z = soc_pred_z_t.detach().cpu().numpy()
            soh_pred_z = soh_pred_z_t.detach().cpu().numpy()
            soc_true_z = soc_z.detach().cpu().numpy()
            soh_true_z = soh_z.detach().cpu().numpy()

            soc_true_raw, soh_true_raw = inverse_targets(
                soc_true_z,
                soh_true_z,
                soc_norm=infer_rows.soc_norm,
                soh_norm=infer_rows.soh_norm,
                normalize_soc=NORMALIZE_SOC,
                zscore_normalize=ZSCORE_NORMALIZE,
            )
            soc_pred_raw, soh_pred_raw = inverse_targets(
                soc_pred_z,
                soh_pred_z,
                soc_norm=infer_rows.soc_norm,
                soh_norm=infer_rows.soh_norm,
                normalize_soc=NORMALIZE_SOC,
                zscore_normalize=ZSCORE_NORMALIZE,
            )

            pred_cls = logits_mat.detach().cpu().argmax(dim=1).numpy()
            true_cls = y_cls_device.detach().cpu().numpy()
            meta_slice = meta_df.iloc[idx_base: idx_base + batch_size].reset_index(drop=True)
            idx_base += batch_size

            for i in range(batch_size):
                true_label = str(label_encoder.inverse_transform([int(true_cls[i])])[0])
                pred_label = str(label_encoder.inverse_transform([int(pred_cls[i])])[0])

                row = {
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "true_material": true_label.split("_")[0],
                    "pred_material": pred_label.split("_")[0],
                    "material_correct": true_label.split("_")[0] == pred_label.split("_")[0],
                    "soc_true": float(soc_true_raw[i]),
                    "soc_pred": float(soc_pred_raw[i]),
                    "soh_true": float(soh_true_raw[i]),
                    "soh_pred": float(soh_pred_raw[i]),
                }
                if ID_COL in meta_slice.columns:
                    row["ID"] = str(meta_slice.loc[i, ID_COL])
                if PT_COL in meta_slice.columns:
                    row["pulse_ms"] = float(meta_slice.loc[i, PT_COL])
                rows.append(row)

    return pd.DataFrame(rows)


def summarize_predictions(df: pd.DataFrame, split: str) -> dict:
    """Compute classification and SOC/SOH error metrics for one split."""
    if len(df) == 0:
        return {
            "split": split,
            "n": 0,
            "cls_acc": np.nan,
            "soc_mae": np.nan,
            "soc_rmse": np.nan,
            "soc_mape": np.nan,
            "soc_medape": np.nan,
            "soc_ape_p25": np.nan,
            "soc_ape_p50": np.nan,
            "soc_ape_p75": np.nan,
            "soc_ape_p90": np.nan,
            "soc_ape_p95": np.nan,
            "soc_ae_p25": np.nan,
            "soc_ae_p50": np.nan,
            "soc_ae_p75": np.nan,
            "soc_ae_p90": np.nan,
            "soc_ae_p95": np.nan,
            "soh_mae": np.nan,
            "soh_rmse": np.nan,
            "soh_mape": np.nan,
            "soh_medape": np.nan,
            "soh_ape_p25": np.nan,
            "soh_ape_p50": np.nan,
            "soh_ape_p75": np.nan,
            "soh_ape_p90": np.nan,
            "soh_ape_p95": np.nan,
            "soh_ae_p25": np.nan,
            "soh_ae_p50": np.nan,
            "soh_ae_p75": np.nan,
            "soh_ae_p90": np.nan,
            "soh_ae_p95": np.nan,
        }
    material_acc = np.mean(
        df["true_label"].str.split("_").str[0] ==
        df["pred_label"].str.split("_").str[0]
    )
    soc_true = df["soc_true"].to_numpy(dtype=np.float64)
    soc_pred = df["soc_pred"].to_numpy(dtype=np.float64)
    soh_true = df["soh_true"].to_numpy(dtype=np.float64)
    soh_pred = df["soh_pred"].to_numpy(dtype=np.float64)

    soc_err = soc_pred - soc_true
    soh_err = soh_pred - soh_true
    soc_abs = np.abs(soc_err)
    soh_abs = np.abs(soh_err)

    soc_ape = soc_abs / (np.abs(soc_true) + 1e-8) * 100.0
    soh_ape = soh_abs / (np.abs(soh_true) + 1e-8) * 100.0

    return {
        "split": split,
        "n": int(len(df)),
        "cls_acc": float((df["true_label"] == df["pred_label"]).mean()),
        "material_acc": float(material_acc),
        "soc_mae": float(np.mean(soc_abs)),
        "soc_medae": medae(soc_true, soc_pred),
        "soc_rmse": float(np.sqrt(np.mean(soc_err ** 2))),
        "soc_mape": float(np.mean(soc_ape)),
        "soc_medape": float(np.median(soc_ape)),
        "soc_ape_p25": float(np.percentile(soc_ape, 25)),
        "soc_ape_p50": float(np.percentile(soc_ape, 50)),
        "soc_ape_p75": float(np.percentile(soc_ape, 75)),
        "soc_ape_p90": float(np.percentile(soc_ape, 90)),
        "soc_ape_p95": float(np.percentile(soc_ape, 95)),
        "soc_ae_p25": float(np.percentile(soc_abs, 25)),
        "soc_ae_p50": float(np.percentile(soc_abs, 50)),
        "soc_ae_p75": float(np.percentile(soc_abs, 75)),
        "soc_ae_p90": float(np.percentile(soc_abs, 90)),
        "soc_ae_p95": float(np.percentile(soc_abs, 95)),

        "soh_mae": float(np.mean(soh_abs)),
        "soh_medae": medae(soh_true, soh_pred),
        "soh_rmse": float(np.sqrt(np.mean(soh_err ** 2))),
        "soh_mape": float(np.mean(soh_ape)),
        "soh_medape": float(np.median(soh_ape)),
        "soh_ape_p25": float(np.percentile(soh_ape, 25)),
        "soh_ape_p50": float(np.percentile(soh_ape, 50)),
        "soh_ape_p75": float(np.percentile(soh_ape, 75)),
        "soh_ape_p90": float(np.percentile(soh_ape, 90)),
        "soh_ape_p95": float(np.percentile(soh_ape, 95)),
        "soh_ae_p25": float(np.percentile(soh_abs, 25)),
        "soh_ae_p50": float(np.percentile(soh_abs, 50)),
        "soh_ae_p75": float(np.percentile(soh_abs, 75)),
        "soh_ae_p90": float(np.percentile(soh_abs, 90)),
        "soh_ae_p95": float(np.percentile(soh_abs, 95)),
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    ds_tr, ds_val, ds_te, mtr, mval, mte, info = build_datasets()
    exp_dir = info["exp_dir"]

    out_root = os.path.join(exp_dir, "further_analysis")
    tab_root = os.path.join(out_root, "tables")
    os.makedirs(tab_root, exist_ok=True)

    model = load_model(
        exp_dir=exp_dir,
        num_classes=info["num_classes"],
        device=device,
        run_cfg=info["run_cfg"],
    )

    dl_te = DataLoader(
        ds_te,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    dl_tr = DataLoader(
        ds_tr,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    infer_rows.soc_norm = info["soc_norm"]
    infer_rows.soh_norm = info["soh_norm"]

    df_test = infer_rows(model, dl_te, mte, info["label_encoder"], device)
    test_path = os.path.join(tab_root, "test_predictions_per_sample.csv")
    df_test.to_csv(test_path, index=False, encoding="utf-8-sig")

    df_val = infer_rows(model, dl_val, mval, info["label_encoder"], device)
    val_path = os.path.join(tab_root, "val_predictions_per_sample.csv")
    df_val.to_csv(val_path, index=False, encoding="utf-8-sig")

    df_train = infer_rows(model, dl_tr, mtr, info["label_encoder"], device)
    train_path = os.path.join(tab_root, "train_predictions_for_scatter.csv")
    df_train.to_csv(train_path, index=False, encoding="utf-8-sig")

    summary_rows = [
        summarize_predictions(df_train, "train"),
        summarize_predictions(df_val, "val"),
        summarize_predictions(df_test, "test"),
    ]
    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(tab_root, "proposed_method_summary.csv")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    meta = {
        "n_mc_soc": int(N_MC_SOC),
        "n_mc_soh": int(N_MC_SOH),
        "n_train": int(info["n_train"]),
        "n_val": int(info["n_val"]),
        "n_test": int(info["n_test"]),
        "num_classes": int(info["num_classes"]),
        "test_predictions_path": test_path,
        "val_predictions_path": val_path,
        "train_predictions_path": train_path,
        "summary_path": summary_path,
    }
    config_path = os.path.join(tab_root, "further_analysis_inference_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n[SUMMARY]")
    print(df_summary.to_string(index=False))

    print(f"[SAVED] {test_path}")
    print(f"[SAVED] {val_path}")
    print(f"[SAVED] {train_path}")
    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {config_path}")
    print(f"[OK] Generated CSVs under {tab_root}")

if __name__ == "__main__":
    main()
