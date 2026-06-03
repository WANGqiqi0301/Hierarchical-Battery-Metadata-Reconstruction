# -*- coding: utf-8 -*-
"""
analysis/error_propagation_analysis.py

Standalone counterfactual / error-propagation analysis.

This script does NOT train the proposed framework.
It loads an already trained proposed-framework checkpoint, rebuilds the test loader,
and evaluates four counterfactual settings:

    E0: oracle material + true SOC
    E1: predicted material + true SOC
    E2: oracle material + predicted SOC
    E3: predicted material + predicted SOC

Interpretation:
    E1 - E0: direct material-prediction error effect on SOH
    E2 - E0: SOC-prediction error propagation effect on SOH
    E3 - E0: total end-to-end gap

Run:
    python analysis/error_propagation_analysis.py
"""

from __future__ import annotations

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.preprocessing import LabelEncoder


# =============================================================================
# Project path
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Import project modules
# =============================================================================
from utils.cache import load_or_build_cache
from utils.data_loader import (
    build_train_mix_soc_mix_pt,
    build_test_random_mix_pt,
    drop_nan_inf_rows,
    pick_test_ids,
    apply_id_split,
)
from proposed_framework.data.pulse_dataset import HierPulseDataset
from proposed_framework.models.hierarchical_model import Hier3HeadModel


# =============================================================================
# User configuration
# =============================================================================
DATA_ROOT = r"F:\OneDrive_Personal\OneDrive\battery\second life battery\data"

EXP_DIR = os.path.join("results", "i10_normalization_flow")
CKPT_DIR = os.path.join(EXP_DIR, "checkpoints")

# 根据你 proposed framework 保存 checkpoint 的名字修改
CHECKPOINT_PATH = os.path.join(CKPT_DIR, "best_model.pt")

SAVE_DIR = os.path.join("results", "analysis", "error_propagation")

# Must match proposed-framework training config
PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

U_START = 1
U_END = 41
DROP_FIRST_CLASS = True

SOC_COL = "SOC"
SOH_COL = "SOH"

USE_PT_AS_FEATURE = True
NORMALIZE_SOC = True
ZSCORE_NORMALIZE = True

BATCH_SIZE = 128
NUM_WORKERS = 0
SEED = 42

TEST_ID_FRAC = 0.2
TEST_ID_COUNT = 0

# Must match proposed-framework model config
WIDTH = 32
BLOCKS = 4
DROP2D = 0.0
HEAD_DROPOUT = 0.2

SOC_HIDDEN = 64
SOH_HIDDEN = 64
FLOW_LAYERS = 6
FLOW_BINS = 8
FLOW_TAIL_BOUND = 3.0

# Counterfactual evaluation config
N_MC_SOC = 128
N_MC_SOH = 256
REPEATS = 5
BASE_SEED = 1000


# =============================================================================
# Basic utilities
# =============================================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_eval_seed(seed: int = 123) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.mean(np.abs(a - b)))


def mape(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    denom = np.maximum(np.abs(a), eps)
    return float(np.mean(np.abs((b - a) / denom)) * 100.0)


def median_ae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.median(np.abs(a - b)))


def median_ape(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    denom = np.maximum(np.abs(a), eps)
    return float(np.median(np.abs((b - a) / denom)) * 100.0)


def inverse_targets_np(
    soc_z: np.ndarray,
    soh_z: np.ndarray,
    soc_norm: Optional[Tuple[float, float]],
    soh_norm: Optional[Tuple[float, float]],
    normalize_soc: bool,
    zscore_normalize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse transform from training target space to raw units.

    SOC:
        training space = SOC / 100, then optional z-score.
        raw output = SOC percentage.

    SOH:
        training space = optional z-score.
        raw output = original SOH unit.
    """
    soc = np.asarray(soc_z, dtype=np.float64).copy()
    soh = np.asarray(soh_z, dtype=np.float64).copy()

    if zscore_normalize:
        if soc_norm is None or soh_norm is None:
            raise RuntimeError("zscore_normalize=True requires soc_norm and soh_norm.")

        soc_mean, soc_std = float(soc_norm[0]), float(soc_norm[1])
        soh_mean, soh_std = float(soh_norm[0]), float(soh_norm[1])

        soc = soc * soc_std + soc_mean
        soh = soh * soh_std + soh_mean

    if normalize_soc:
        soc = soc * 100.0

    return soc, soh


def onehot_from_y(y_cls: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(y_cls.view(-1), num_classes=int(num_classes)).float()


def to_numpy_1d(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().view(-1).numpy()


# =============================================================================
# Robust model inference helpers
# =============================================================================
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
def predict_material_prob(
    model: torch.nn.Module,
    x3: torch.Tensor,
    pt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predict material probability.

    Important:
    Older model may use:
        logits = head_mat(z)

    Newer model may use:
        logits = head_mat(torch.cat([z, pt], dim=1))

    This helper automatically handles both.
    """
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
            f"z_dim={z_dim}, pt_dim={pt_dim}, "
            f"head_first_linear_in_features={in_features}."
        )

    p = torch.softmax(logits, dim=1)
    return z, logits, p


@torch.no_grad()
def sample_flow_mean_1d(
    flow: torch.nn.Module,
    context: torch.Tensor,
    n_mc: int,
) -> torch.Tensor:
    """
    Return MC mean from a 1D conditional flow.

    Supports common nflows output shapes:
        (n_mc, B, 1)
        (B, n_mc, 1)
        (B * n_mc, 1)
    """
    n_mc = int(n_mc)
    B = int(context.size(0))

    s = flow.sample(context=context, num_samples=n_mc)

    if s.ndim == 3:
        if s.shape[0] == n_mc and s.shape[1] == B:
            return s.mean(dim=0).squeeze(-1).view(-1)
        if s.shape[0] == B and s.shape[1] == n_mc:
            return s.mean(dim=1).squeeze(-1).view(-1)

        s = s.reshape(n_mc, B, 1)
        return s.mean(dim=0).squeeze(-1).view(-1)

    if s.ndim == 2:
        s = s.view(n_mc, B, 1)
        return s.mean(dim=0).squeeze(-1).view(-1)

    raise RuntimeError(f"Unexpected flow sample shape: {tuple(s.shape)}")


@torch.no_grad()
def infer_soc_given_p(
    model: torch.nn.Module,
    x3: torch.Tensor,
    pt: torch.Tensor,
    p_used: torch.Tensor,
    n_mc: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Infer SOC using fixed material probability p_used.
    """
    if hasattr(model, "infer_soc_given_p"):
        return model.infer_soc_given_p(
            x_img=x3,
            x_pt=pt,
            p_used=p_used,
            n_mc=int(n_mc),
        )

    z = model.encoder(x3)

    if model_uses_pt(model):
        cond_soc = torch.cat([z, p_used, pt], dim=1)
    else:
        cond_soc = torch.cat([z, p_used], dim=1)

    soc_pred = sample_flow_mean_1d(model.soc_flow, cond_soc, n_mc=int(n_mc))
    return z, cond_soc, soc_pred.view(-1)


@torch.no_grad()
def infer_soh_given_p_and_soc(
    model: torch.nn.Module,
    x3: torch.Tensor,
    pt: torch.Tensor,
    p_used: torch.Tensor,
    soc_val: torch.Tensor,
    n_mc: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Infer SOH using fixed material probability p_used and fixed SOC condition.
    """
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


# =============================================================================
# Counterfactual E0-E3
# =============================================================================
@torch.no_grad()
def eval_counterfactual_E0E3(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
    soc_norm: Optional[Tuple[float, float]],
    soh_norm: Optional[Tuple[float, float]],
    normalize_soc: bool,
    zscore_normalize: bool,
    n_mc_soc: int = 128,
    n_mc_soh: int = 256,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Four counterfactual experiments:

    E0:
        p = oracle one-hot material
        SOH condition uses true SOC

    E1:
        p = predicted material softmax
        SOH condition uses true SOC

    E2:
        p = oracle one-hot material
        SOH condition uses predicted SOC under oracle material

    E3:
        p = predicted material softmax
        SOH condition uses predicted SOC under predicted material
        This is the end-to-end setting.
    """
    model.eval()

    store = {
        "E0": {"soc_t": [], "soc_p": [], "soh_t": [], "soh_p": []},
        "E1": {"soc_t": [], "soc_p": [], "soh_t": [], "soh_p": []},
        "E2": {"soc_t": [], "soc_p": [], "soh_t": [], "soh_p": []},
        "E3": {"soc_t": [], "soc_p": [], "soh_t": [], "soh_p": []},
    }

    for x3, pt, y_cls, soc_true, soh_true in loader:
        x3 = x3.to(device)
        pt = pt.to(device)
        y_cls = y_cls.to(device).view(-1)
        soc_true = soc_true.to(device).view(-1)
        soh_true = soh_true.to(device).view(-1)

        p_oracle = onehot_from_y(y_cls, num_classes=num_classes).to(device)

        _, _, p_pred = predict_material_prob(
            model=model,
            x3=x3,
            pt=pt,
        )

        _, _, soc_pred_oracle_p = infer_soc_given_p(
            model=model,
            x3=x3,
            pt=pt,
            p_used=p_oracle,
            n_mc=int(n_mc_soc),
        )

        _, _, soc_pred_pred_p = infer_soc_given_p(
            model=model,
            x3=x3,
            pt=pt,
            p_used=p_pred,
            n_mc=int(n_mc_soc),
        )

        _, soh_E0 = infer_soh_given_p_and_soc(
            model=model,
            x3=x3,
            pt=pt,
            p_used=p_oracle,
            soc_val=soc_true,
            n_mc=int(n_mc_soh),
        )

        _, soh_E1 = infer_soh_given_p_and_soc(
            model=model,
            x3=x3,
            pt=pt,
            p_used=p_pred,
            soc_val=soc_true,
            n_mc=int(n_mc_soh),
        )

        _, soh_E2 = infer_soh_given_p_and_soc(
            model=model,
            x3=x3,
            pt=pt,
            p_used=p_oracle,
            soc_val=soc_pred_oracle_p,
            n_mc=int(n_mc_soh),
        )

        _, soh_E3 = infer_soh_given_p_and_soc(
            model=model,
            x3=x3,
            pt=pt,
            p_used=p_pred,
            soc_val=soc_pred_pred_p,
            n_mc=int(n_mc_soh),
        )

        soc_true_np = to_numpy_1d(soc_true)
        soh_true_np = to_numpy_1d(soh_true)

        # E0/E2 use SOC prediction under oracle material.
        # E1/E3 use SOC prediction under predicted material.
        store["E0"]["soc_t"].append(soc_true_np)
        store["E0"]["soc_p"].append(to_numpy_1d(soc_pred_oracle_p))
        store["E0"]["soh_t"].append(soh_true_np)
        store["E0"]["soh_p"].append(to_numpy_1d(soh_E0))

        store["E1"]["soc_t"].append(soc_true_np)
        store["E1"]["soc_p"].append(to_numpy_1d(soc_pred_pred_p))
        store["E1"]["soh_t"].append(soh_true_np)
        store["E1"]["soh_p"].append(to_numpy_1d(soh_E1))

        store["E2"]["soc_t"].append(soc_true_np)
        store["E2"]["soc_p"].append(to_numpy_1d(soc_pred_oracle_p))
        store["E2"]["soh_t"].append(soh_true_np)
        store["E2"]["soh_p"].append(to_numpy_1d(soh_E2))

        store["E3"]["soc_t"].append(soc_true_np)
        store["E3"]["soc_p"].append(to_numpy_1d(soc_pred_pred_p))
        store["E3"]["soh_t"].append(soh_true_np)
        store["E3"]["soh_p"].append(to_numpy_1d(soh_E3))

    def summarize(tag: str) -> Dict[str, float]:
        soc_t = np.concatenate(store[tag]["soc_t"])
        soc_p = np.concatenate(store[tag]["soc_p"])
        soh_t = np.concatenate(store[tag]["soh_t"])
        soh_p = np.concatenate(store[tag]["soh_p"])

        soc_t_raw, soh_t_raw = inverse_targets_np(
            soc_t,
            soh_t,
            soc_norm=soc_norm,
            soh_norm=soh_norm,
            normalize_soc=normalize_soc,
            zscore_normalize=zscore_normalize,
        )

        soc_p_raw, soh_p_raw = inverse_targets_np(
            soc_p,
            soh_p,
            soc_norm=soc_norm,
            soh_norm=soh_norm,
            normalize_soc=normalize_soc,
            zscore_normalize=zscore_normalize,
        )

        return {
            "soc_rmse_raw": rmse(soc_t_raw, soc_p_raw),
            "soc_mae_raw": mae(soc_t_raw, soc_p_raw),
            "soc_mape_raw": mape(soc_t_raw, soc_p_raw),
            "soc_median_ae_raw": median_ae(soc_t_raw, soc_p_raw),
            "soc_median_ape_raw": median_ape(soc_t_raw, soc_p_raw),

            "soh_rmse_raw": rmse(soh_t_raw, soh_p_raw),
            "soh_mae_raw": mae(soh_t_raw, soh_p_raw),
            "soh_mape_raw": mape(soh_t_raw, soh_p_raw),
            "soh_median_ae_raw": median_ae(soh_t_raw, soh_p_raw),
            "soh_median_ape_raw": median_ape(soh_t_raw, soh_p_raw),
        }

    res = {k: summarize(k) for k in ["E0", "E1", "E2", "E3"]}

    direct_mat_to_soh = res["E1"]["soh_rmse_raw"] - res["E0"]["soh_rmse_raw"]
    soc_to_soh = res["E2"]["soh_rmse_raw"] - res["E0"]["soh_rmse_raw"]
    total_gap = res["E3"]["soh_rmse_raw"] - res["E0"]["soh_rmse_raw"]
    residual_gap = total_gap - direct_mat_to_soh - soc_to_soh

    attribution = {
        "direct_mat_to_soh_rmse_gap": float(direct_mat_to_soh),
        "soc_to_soh_rmse_gap": float(soc_to_soh),
        "total_end2end_rmse_gap": float(total_gap),
        "residual_or_interaction_rmse_gap": float(residual_gap),
    }

    return res, attribution


# =============================================================================
# Repeated summary
# =============================================================================
def summarize_repeated_counterfactual(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
    soc_norm: Optional[Tuple[float, float]],
    soh_norm: Optional[Tuple[float, float]],
    normalize_soc: bool,
    zscore_normalize: bool,
    n_mc_soc: int = 128,
    n_mc_soh: int = 256,
    repeats: int = 5,
    base_seed: int = 1000,
    save_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Repeat E0-E3 counterfactual evaluation with different MC seeds.

    Saves:
        counterfactual_runs.csv
        counterfactual_summary.csv
        attribution_summary.csv
    """
    run_rows = []
    attr_rows = []

    for r in range(int(repeats)):
        seed = int(base_seed + r)
        set_eval_seed(seed)

        res_cf, attr = eval_counterfactual_E0E3(
            model=model,
            loader=loader,
            device=device,
            num_classes=int(num_classes),
            soc_norm=soc_norm,
            soh_norm=soh_norm,
            normalize_soc=normalize_soc,
            zscore_normalize=zscore_normalize,
            n_mc_soc=int(n_mc_soc),
            n_mc_soh=int(n_mc_soh),
        )

        row = {
            "run": int(r),
            "seed": int(seed),
        }

        for exp in ["E0", "E1", "E2", "E3"]:
            for metric_name, value in res_cf[exp].items():
                row[f"{exp}__{metric_name}"] = float(value)

        run_rows.append(row)

        attr_row = {
            "run": int(r),
            "seed": int(seed),
        }
        for k, v in attr.items():
            attr_row[k] = float(v)
        attr_rows.append(attr_row)

    df_runs = pd.DataFrame(run_rows)
    df_attr_runs = pd.DataFrame(attr_rows)

    def agg_stats(series: pd.Series) -> Dict[str, float]:
        x = series.to_numpy(dtype=float)
        if x.size == 0:
            return {"mean": np.nan, "std": np.nan, "median": np.nan}
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)) if x.size >= 2 else 0.0,
            "median": float(np.median(x)),
        }

    summary_rows = []
    metric_cols = [
        c for c in df_runs.columns
        if c.startswith(("E0__", "E1__", "E2__", "E3__"))
    ]

    metric_cols = sorted(
        metric_cols,
        key=lambda s: (s.split("__", 1)[0], s.split("__", 1)[1]),
    )

    for col in metric_cols:
        exp, metric = col.split("__", 1)
        stats = agg_stats(df_runs[col])
        summary_rows.append({
            "experiment": exp,
            "metric": metric,
            "mean": stats["mean"],
            "std": stats["std"],
            "median": stats["median"],
        })

    df_summary = pd.DataFrame(summary_rows)

    attr_summary_rows = []
    attr_metric_cols = [
        c for c in df_attr_runs.columns
        if c not in ["run", "seed"]
    ]

    for col in attr_metric_cols:
        stats = agg_stats(df_attr_runs[col])
        attr_summary_rows.append({
            "metric": col,
            "mean": stats["mean"],
            "std": stats["std"],
            "median": stats["median"],
        })

    df_attr_summary = pd.DataFrame(attr_summary_rows)

    if save_dir is not None:
        ensure_dir(save_dir)

        runs_path = os.path.join(save_dir, "counterfactual_runs.csv")
        summary_path = os.path.join(save_dir, "counterfactual_summary.csv")
        attr_path = os.path.join(save_dir, "attribution_summary.csv")

        df_runs.to_csv(runs_path, index=False, encoding="utf-8-sig")
        df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        df_attr_summary.to_csv(attr_path, index=False, encoding="utf-8-sig")

        print(f"[SAVED] {runs_path}")
        print(f"[SAVED] {summary_path}")
        print(f"[SAVED] {attr_path}")

    return df_runs, df_summary, df_attr_summary


# =============================================================================
# Standalone experiment: rebuild test loader + load checkpoint
# =============================================================================
def build_test_loader_for_error_propagation():
    """
    Rebuild the same train/test split and test loader used by proposed framework.

    Important:
    This function does not train a model.
    It only reconstructs:
        - train-only U normalization
        - train-only target normalization
        - label encoder
        - test dataset / loader
    """
    cache_dir = os.path.join(EXP_DIR, "cache")
    splits_dir = os.path.join(EXP_DIR, "splits")
    ensure_dir(cache_dir, splits_dir)

    soc_list = list(range(5, 90, 5))

    train_kwargs = dict(
        data_root=DATA_ROOT,
        soc_list=soc_list,
        pulse_list=list(map(int, PULSE_LIST)),
        u_start=U_START,
        u_end=U_END,
        drop_first_class=DROP_FIRST_CLASS,
    )

    Xtr_raw, ytr_raw, mtr_raw, _, _ = load_or_build_cache(
        cache_dir,
        "raw_train",
        build_train_mix_soc_mix_pt,
        train_kwargs,
    )

    test_kwargs = dict(
        data_root=DATA_ROOT,
        pulse_list=list(map(int, PULSE_LIST)),
        u_start=U_START,
        u_end=U_END,
        drop_first_class=DROP_FIRST_CLASS,
    )

    Xte_raw, yte_raw, mte_raw, _, _ = load_or_build_cache(
        cache_dir,
        "raw_test",
        build_test_random_mix_pt,
        test_kwargs,
    )

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

    all_ids = pd.concat(
        [mtr_raw["ID"], mte_raw["ID"]],
        axis=0,
    ).astype(str).to_numpy()

    test_ids = pick_test_ids(
        all_ids=all_ids,
        test_id_frac=TEST_ID_FRAC,
        test_id_count=TEST_ID_COUNT,
        seed=SEED,
    )

    Xtr, ytr_str, mtr, Xte, yte_str, mte = apply_id_split(
        Xtr_raw,
        ytr_raw,
        mtr_raw,
        Xte_raw,
        yte_raw,
        mte_raw,
        test_ids=test_ids,
    )

    if len(ytr_str) == 0 or len(yte_str) == 0:
        raise RuntimeError("Empty train or test set after ID split.")

    # U normalization: train stats only
    u_mean = Xtr.mean(axis=0, keepdims=True)
    u_std = Xtr.std(axis=0, keepdims=True) + 1e-8

    Xtr = (Xtr - u_mean) / u_std
    Xte = (Xte - u_mean) / u_std

    # Target normalization: train stats only
    soc_tr = mtr[SOC_COL].astype(float).to_numpy(dtype=np.float64)
    if NORMALIZE_SOC:
        soc_tr = soc_tr / 100.0
    soc_norm = (float(soc_tr.mean()), float(soc_tr.std() + 1e-8))

    soh_tr = mtr[SOH_COL].astype(float).to_numpy(dtype=np.float64)
    soh_norm = (float(soh_tr.mean()), float(soh_tr.std() + 1e-8))

    # Label encoder: fit on train labels only
    le = LabelEncoder()
    ytr_cls = le.fit_transform(ytr_str)

    train_classes = set(le.classes_.tolist())
    mask_known = np.array(
        [lbl in train_classes for lbl in yte_str],
        dtype=bool,
    )

    Xte = Xte[mask_known]
    yte_str = yte_str[mask_known]
    mte = mte.loc[mask_known].reset_index(drop=True)

    yte_cls = le.transform(yte_str)
    num_classes = len(le.classes_)

    # Pulse-width normalization: train stats only
    if USE_PT_AS_FEATURE and "pulse_ms" in mtr.columns:
        pt_train = np.log1p(mtr["pulse_ms"].astype(float).to_numpy(dtype=np.float64))
        pt_norm = (float(pt_train.mean()), float(pt_train.std() + 1e-8))
    else:
        pt_norm = (0.0, 1.0)

    ds_te = HierPulseDataset(
        X_u=Xte,
        y_cls=yte_cls,
        meta=mte,
        soc_col=SOC_COL,
        soh_col=SOH_COL,
        pt_col="pulse_ms",
        use_pt_as_feature=USE_PT_AS_FEATURE,
        pt_norm=pt_norm,
        normalize_soc=NORMALIZE_SOC,
        zscore_normalize=ZSCORE_NORMALIZE,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
    )

    dl_te = DataLoader(
        ds_te,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    info = {
        "num_classes": int(num_classes),
        "classes": [str(x) for x in le.classes_.tolist()],
        "soc_norm": soc_norm,
        "soh_norm": soh_norm,
        "pt_norm": pt_norm,
        "n_test": int(len(ds_te)),
        "test_unique_ids": int(mte["ID"].astype(str).nunique()) if "ID" in mte.columns else None,
    }

    print(f"[DATA] Test samples: {info['n_test']}")
    print(f"[DATA] Test unique IDs: {info['test_unique_ids']}")
    print(f"[DATA] Num classes: {info['num_classes']}")

    return dl_te, info


def load_trained_model(
    checkpoint_path: str,
    num_classes: int,
    device: str,
) -> torch.nn.Module:
    model = Hier3HeadModel(
        num_classes=int(num_classes),
        width=WIDTH,
        blocks=BLOCKS,
        drop2d=DROP2D,
        use_pt_as_feature=USE_PT_AS_FEATURE,
        soc_hidden=SOC_HIDDEN,
        soh_hidden=SOH_HIDDEN,
        head_dropout=HEAD_DROPOUT,
        flow_layers=FLOW_LAYERS,
        flow_bins=FLOW_BINS,
        flow_tail_bound=FLOW_TAIL_BOUND,
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please update CHECKPOINT_PATH at the top of this script."
        )

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format.")

    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"[MODEL] Loaded checkpoint: {checkpoint_path}")
    return model


def run_error_propagation_experiment():
    ensure_dir(SAVE_DIR)
    set_eval_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    dl_te, info = build_test_loader_for_error_propagation()

    model = load_trained_model(
        checkpoint_path=CHECKPOINT_PATH,
        num_classes=info["num_classes"],
        device=device,
    )

    df_runs, df_summary, df_attr_summary = summarize_repeated_counterfactual(
        model=model,
        loader=dl_te,
        device=device,
        num_classes=info["num_classes"],
        soc_norm=info["soc_norm"] if ZSCORE_NORMALIZE else None,
        soh_norm=info["soh_norm"] if ZSCORE_NORMALIZE else None,
        normalize_soc=NORMALIZE_SOC,
        zscore_normalize=ZSCORE_NORMALIZE,
        n_mc_soc=N_MC_SOC,
        n_mc_soh=N_MC_SOH,
        repeats=REPEATS,
        base_seed=BASE_SEED,
        save_dir=SAVE_DIR,
    )

    config = {
        "data_root": DATA_ROOT,
        "exp_dir": EXP_DIR,
        "checkpoint_path": CHECKPOINT_PATH,
        "pulse_list": list(map(int, PULSE_LIST)),
        "u_start": U_START,
        "u_end": U_END,
        "drop_first_class": DROP_FIRST_CLASS,
        "soc_col": SOC_COL,
        "soh_col": SOH_COL,
        "use_pt_as_feature": USE_PT_AS_FEATURE,
        "normalize_soc": NORMALIZE_SOC,
        "zscore_normalize": ZSCORE_NORMALIZE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "test_id_frac": TEST_ID_FRAC,
        "test_id_count": TEST_ID_COUNT,
        "model": {
            "width": WIDTH,
            "blocks": BLOCKS,
            "drop2d": DROP2D,
            "head_dropout": HEAD_DROPOUT,
            "soc_hidden": SOC_HIDDEN,
            "soh_hidden": SOH_HIDDEN,
            "flow_layers": FLOW_LAYERS,
            "flow_bins": FLOW_BINS,
            "flow_tail_bound": FLOW_TAIL_BOUND,
        },
        "counterfactual": {
            "n_mc_soc": N_MC_SOC,
            "n_mc_soh": N_MC_SOH,
            "repeats": REPEATS,
            "base_seed": BASE_SEED,
        },
        "data_info": {
            "num_classes": info["num_classes"],
            "classes": info["classes"],
            "soc_norm": [float(info["soc_norm"][0]), float(info["soc_norm"][1])],
            "soh_norm": [float(info["soh_norm"][0]), float(info["soh_norm"][1])],
            "pt_norm": [float(info["pt_norm"][0]), float(info["pt_norm"][1])],
            "n_test": info["n_test"],
            "test_unique_ids": info["test_unique_ids"],
        },
    }

    config_path = os.path.join(SAVE_DIR, "error_propagation_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {config_path}")

    print("\n===== Attribution summary =====")
    print(df_attr_summary.to_string(index=False))

    print("\n===== E0-E3 SOH RMSE raw =====")
    tab = df_summary[
        (df_summary["metric"] == "soh_rmse_raw")
    ].copy()
    print(tab.to_string(index=False))

    print("\n[DONE] Error-propagation analysis finished.")
    print(f"[OUT] {SAVE_DIR}")

    return df_runs, df_summary, df_attr_summary


if __name__ == "__main__":
    run_error_propagation_experiment()