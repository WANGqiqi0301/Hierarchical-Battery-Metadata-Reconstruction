# -*- coding: utf-8 -*-
"""
run_further_analysis_proposed.py

Generate per-sample prediction tables for the proposed framework (run_proposed_framework.py).
Outputs are identical to original i10 pipeline:

- test_predictions_per_sample.csv
- train_predictions_for_scatter.csv
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import proposed_framework.run_proposed_framework as M

# -----------------------------
# Config
# -----------------------------
EXP_DIR = getattr(M, "EXP_DIR", "results/proposed_framework")
DATA_ROOT = getattr(M, "DATA_ROOT", "data")

BATCH_SIZE = 512
NUM_WORKERS = 0
SEED = 42
SOC_COL = "SOC"
SOH_COL = "SOH"
ID_COL = "ID"
PT_COL = "pulse_ms"
USE_PT_AS_FEATURE = True
NORMALIZE_SOC = True
ZSCORE_NORMALIZE = True
PULSE_LIST = [30,50,70,100,300,500,700,1000,3000,5000]

# -----------------------------
# Helper functions
# -----------------------------
def safe_mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs((y_pred - y_true)/np.maximum(np.abs(y_true), eps))) * 100.0)

def safe_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_pred - y_true)))

def apply_u41_train_only_norm(exp_dir: str, Xtr: np.ndarray, Xte: np.ndarray):
    """Apply U(41) normalization using TRAIN-only statistics."""
    norm_path = os.path.join(exp_dir, "u41_norm_train_only.npz")
    if os.path.exists(norm_path):
        obj = np.load(norm_path)
        u_mean = obj["u_mean"].astype(np.float64)
        u_std  = obj["u_std"].astype(np.float64)
        print(f"[NORM] Loaded train-only U(41) stats: {norm_path}")
    else:
        u_mean = Xtr.mean(axis=0, keepdims=True)
        u_std  = Xtr.std(axis=0, keepdims=True) + 1e-8
        np.savez_compressed(norm_path, u_mean=u_mean.astype(np.float32), u_std=u_std.astype(np.float32))
        print(f"[NORM] WARNING: norm file missing; recomputed and saved: {norm_path}")
    Xtr_n = (Xtr - u_mean) / (u_std + 1e-8)
    Xte_n = (Xte - u_mean) / (u_std + 1e-8)
    return Xtr_n, Xte_n
def inverse_targets(
    soc_z: np.ndarray,
    soh_z: np.ndarray,
    soc_norm=None,
    soh_norm=None,
    normalize_soc: bool = True,
    zscore_normalize: bool = True,
):
    """
    Convert normalized SOC/SOH targets back to raw units.

    SOC output:
        - If normalize_soc=True, raw SOC is returned in percentage scale (%).
    SOH output:
        - Returned in original SOH unit.
    """
    soc = np.asarray(soc_z, dtype=np.float64)
    soh = np.asarray(soh_z, dtype=np.float64)

    if zscore_normalize:
        if soc_norm is None or soh_norm is None:
            raise RuntimeError(
                "zscore_normalize=True requires soc_norm and soh_norm."
            )

        soc_mean, soc_std = soc_norm
        soh_mean, soh_std = soh_norm

        soc = soc * soc_std + soc_mean
        soh = soh * soh_std + soh_mean

    if normalize_soc:
        soc = soc * 100.0

    return soc, soh
# -----------------------------
# Main
# -----------------------------
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_dir = os.path.join(EXP_DIR,"cache")
    out_root = os.path.join(EXP_DIR,"further_analysis")
    tab_root = os.path.join(out_root,"tables")
    os.makedirs(tab_root, exist_ok=True)

    # -----------------------------
    # 1) Load datasets
    # -----------------------------
    soc_list = list(range(5,90,5))
    train_kwargs = dict(data_root=DATA_ROOT, soc_list=soc_list, pulse_list=PULSE_LIST, u_start=1, u_end=41, drop_first_class=True)
    test_kwargs  = dict(data_root=DATA_ROOT, pulse_list=PULSE_LIST, u_start=1, u_end=41, drop_first_class=True)

    Xtr_raw, ytr_raw, mtr_raw, _, _ = M.load_or_build_cache(cache_dir,"raw_train",M.build_train_mix_soc_mix_pt,train_kwargs)
    Xte_raw, yte_raw, mte_raw, _, _ = M.load_or_build_cache(cache_dir,"raw_test",M.build_test_random_mix_pt,test_kwargs)

    Xtr_raw, ytr_raw, mtr_raw = M.drop_nan_inf_rows(Xtr_raw, ytr_raw, mtr_raw,"RAW_TRAIN")
    Xte_raw, yte_raw, mte_raw = M.drop_nan_inf_rows(Xte_raw, yte_raw, mte_raw,"RAW_TEST")

    all_ids = pd.concat([mtr_raw[ID_COL], mte_raw[ID_COL]], axis=0).astype(str).to_numpy()
    test_ids = M.pick_test_ids(all_ids, test_id_frac=0.2, test_id_count=0, seed=SEED)
    Xtr, ytr_str, mtr, Xte, yte_str, mte = M.apply_id_split(Xtr_raw, ytr_raw, mtr_raw, Xte_raw, yte_raw, mte_raw, test_ids=test_ids)

    # -----------------------------
    # 2) Apply train-only normalization
    # -----------------------------
    Xtr, Xte = apply_u41_train_only_norm(EXP_DIR, Xtr, Xte)

    # -----------------------------
    # 3) Compute target normalization (SOC/SOH) from TRAIN split
    # -----------------------------
    soc_tr = mtr[SOC_COL].astype(float).to_numpy(dtype=np.float64)
    if NORMALIZE_SOC:
        soc_tr = soc_tr / 100.0
    soc_norm = (float(soc_tr.mean()), float(soc_tr.std() + 1e-8))

    soh_tr = mtr[SOH_COL].astype(float).to_numpy(dtype=np.float64)
    soh_norm = (float(soh_tr.mean()), float(soh_tr.std() + 1e-8))

    # Optional PT normalization
    if USE_PT_AS_FEATURE and PT_COL in mtr.columns:
        pt_tr = mtr[PT_COL].astype(float).to_numpy(dtype=np.float32)
        pt_log = np.log1p(pt_tr)
        pt_norm = (float(pt_log.mean()), float(pt_log.std() + 1e-8))
    else:
        pt_norm = (0.0, 1.0)

    # -----------------------------
    # 4) Label encoding
    # -----------------------------
    le = LabelEncoder()
    _ = le.fit_transform(ytr_str)
    train_classes = set(le.classes_.tolist())
    mask_known = np.array([lbl in train_classes for lbl in yte_str],dtype=bool)
    Xte = Xte[mask_known]
    yte_str = yte_str[mask_known]
    mte = mte.loc[mask_known].reset_index(drop=True)
    yte_cls = le.transform(yte_str)
    K = len(le.classes_)

    # -----------------------------
    # 5) Load checkpoint
    # -----------------------------
    best_ckpt_path = os.path.join(EXP_DIR,"checkpoints","stage2_soh","best.pt")
    if not os.path.exists(best_ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {best_ckpt_path}")
    model = M.Hier3HeadModel(num_classes=K,width=32,blocks=4,drop2d=0.0,use_pt_as_feature=USE_PT_AS_FEATURE,head_dropout=0.2).to(device)
    ckpt = torch.load(best_ckpt_path,map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # -----------------------------
    # 6) Inference on TEST
    # -----------------------------
    rows = []
    ds_te = M.HierPulseDataset(
        X_u=Xte,
        y_cls=yte_cls,
        meta=mte,
        soc_col=SOC_COL,
        soh_col=SOH_COL,
        use_pt_as_feature=USE_PT_AS_FEATURE,
        pt_norm=pt_norm,
        normalize_soc=NORMALIZE_SOC,
        zscore_normalize=True,
        soc_norm=soc_norm,
        soh_norm=soh_norm
    )
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

    idx_base = 0
    with torch.no_grad():
        for x3, pt, y_cls, soc_z, soh_z in dl_te:
            bs = x3.size(0)
            x3 = x3.to(device)
            pt = pt.to(device)
            logits_mat, soc_pred_z_t, soc_logp, cond_soc, soh_pred_z_t, cond_soh = model(x3, pt, soc_tf=None, n_mc=16)
            soc_pred_z = soc_pred_z_t.detach().cpu().numpy()
            soh_pred_z = soh_pred_z_t.detach().cpu().numpy()
            soc_true_z = soc_z.detach().cpu().numpy()
            soh_true_z = soh_z.detach().cpu().numpy()

            soc_true_raw, soh_true_raw = inverse_targets(
                soc_true_z,
                soh_true_z,
                soc_norm=soc_norm,
                soh_norm=soh_norm,
                normalize_soc=NORMALIZE_SOC,
                zscore_normalize=True
            )
            soc_pred_raw, soh_pred_raw = inverse_targets(
                soc_pred_z,
                soh_pred_z,
                soc_norm=soc_norm,
                soh_norm=soh_norm,
                normalize_soc=NORMALIZE_SOC,
                zscore_normalize=True
            )

            meta_slice = mte.iloc[idx_base: idx_base+bs].reset_index(drop=True)
            idx_base += bs
            for i in range(bs):
                rows.append({
                    "ID": str(meta_slice.loc[i, ID_COL]) if ID_COL in meta_slice.columns else "",
                    "pulse_ms": float(meta_slice.loc[i, PT_COL]) if PT_COL in meta_slice.columns else np.nan,
                    "true_label": str(le.inverse_transform([y_cls[i]])[0]),
                    "pred_label": str(le.inverse_transform([logits_mat.detach().cpu().argmax(1)[i]])[0]),
                    "soc_true": float(soc_true_raw[i]),
                    "soc_pred": float(soc_pred_raw[i]),
                    "soh_true": float(soh_true_raw[i]),
                    "soh_pred": float(soh_pred_raw[i])
                })

    df_test = pd.DataFrame(rows)
    df_test.to_csv(os.path.join(tab_root,"test_predictions_per_sample.csv"),index=False,encoding="utf-8-sig")

    # -----------------------------
    # 7) Inference on TRAIN (scatter)
    # -----------------------------
    rows_tr = []
    ds_tr = M.HierPulseDataset(
        X_u=Xtr,
        y_cls=le.transform(ytr_str),
        meta=mtr,
        soc_col=SOC_COL,
        soh_col=SOH_COL,
        use_pt_as_feature=USE_PT_AS_FEATURE,
        pt_norm=pt_norm,
        normalize_soc=NORMALIZE_SOC,
        zscore_normalize=True,
        soc_norm=soc_norm,
        soh_norm=soh_norm
    )
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
    idx_base = 0
    with torch.no_grad():
        for x3, pt, y_cls, soc_z, soh_z in dl_tr:
            bs = x3.size(0)
            x3 = x3.to(device)
            pt = pt.to(device)
            logits_mat, soc_pred_z_t, soc_logp, cond_soc, soh_pred_z_t, cond_soh = model(x3, pt, soc_tf=None, n_mc=16)
            soc_pred_z = soc_pred_z_t.detach().cpu().numpy()
            soh_pred_z = soh_pred_z_t.detach().cpu().numpy()
            soc_true_z = soc_z.detach().cpu().numpy()
            soh_true_z = soh_z.detach().cpu().numpy()
            soc_true_raw, soh_true_raw = inverse_targets(
                soc_true_z,
                soh_true_z,
                soc_norm=soc_norm,
                soh_norm=soh_norm,
                normalize_soc=NORMALIZE_SOC,
                zscore_normalize=True
            )
            soc_pred_raw, soh_pred_raw = inverse_targets(
                soc_pred_z,
                soh_pred_z,
                soc_norm=soc_norm,
                soh_norm=soh_norm,
                normalize_soc=NORMALIZE_SOC,
                zscore_normalize=True
            )
            for i in range(bs):
                rows_tr.append({
                    "soc_true": float(soc_true_raw[i]),
                    "soc_pred": float(soc_pred_raw[i]),
                    "soh_true": float(soh_true_raw[i]),
                    "soh_pred": float(soh_pred_raw[i])
                })

    df_train = pd.DataFrame(rows_tr)
    df_train.to_csv(os.path.join(tab_root,"train_predictions_for_scatter.csv"),index=False,encoding="utf-8-sig")

    print(f"[OK] Generated CSVs under {tab_root}")


if __name__=="__main__":
    main()