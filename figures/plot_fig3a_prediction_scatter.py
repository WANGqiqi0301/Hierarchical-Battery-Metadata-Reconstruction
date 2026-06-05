# -*- coding: utf-8 -*-
"""
figures/plot_fig3a_prediction_scatter.py

Reproduce Figure 3a:
- SOH train prediction scatter
- SOH test prediction scatter
- SOC train prediction scatter
- SOC test prediction scatter

This script:
1) reads raw data and trained-model artifacts from an experiment directory,
2) regenerates or loads prediction npz files,
3) plots four KDE-colored scatter figures,
4) saves a combined 2x2 figure.

Typical usage:

    python figures/plot_fig3a_prediction_scatter.py ^
        --data_root data ^
        --exp_dir results/i10_normalization_flow ^
        --force_zscore ^
        --force_regenerate

Quiet by default. Add --verbose for detailed logs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde
from sklearn.preprocessing import LabelEncoder


# =============================================================================
# Project root
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Imports from organized repository
# =============================================================================

from utils.cache import load_or_build_cache, drop_nan_inf_rows
from proposed_framework.data.build_dataset import (
    build_train_mix_soc_mix_pt,
    build_test_random_mix_pt,
    pick_test_ids,
    apply_id_split,
)
from proposed_framework.data.pulse_dataset import HierPulseDataset
from proposed_framework.models.hierarchical_model import Hier3HeadModel


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

DEFAULT_CONFIG = {
    "seed": 42,
    "test_id_frac": 0.2,
    "test_id_count": 0,
    "u_start": 1,
    "u_end": 41,
    "drop_first_class": True,
    "soc_col": "SOC",
    "soh_col": "SOH",
    "use_pt_as_feature": True,
    "normalize_soc": True,

    # Old i10_normalization_flow used target z-score normalization.
    "zscore_normalize": True,

    "width": 32,
    "blocks": 4,
    "drop2d": 0.0,
    "head_dropout": 0.2,
    "batch_size": 256,
    "n_mc_test": 16,
    "pulse_list": DEFAULT_PULSE_LIST,

    # plotting
    "hex_colors": ["#732C7C", "#4B7DA6", "#65A5D9", "#41C28A", "#FFEF30"],
    "dot_size": 120,
    "font": "Arial",
    "bw_method": 0.1,
}


# =============================================================================
# Basic utilities
# =============================================================================

def log(message: str, verbose: bool = False) -> None:
    if verbose:
        print(message)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _torch_load(path: str | Path, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(
    fig: plt.Figure,
    out_base: str | Path,
    dpi: int = 300,
    transparent: bool = False,
) -> None:
    out_base = Path(out_base)
    ensure_dir(out_base.parent)

    fig.savefig(str(out_base) + ".png", dpi=dpi, bbox_inches="tight", transparent=transparent)
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight", transparent=transparent)
    fig.savefig(str(out_base) + ".svg", bbox_inches="tight", transparent=transparent)


# =============================================================================
# Target inverse transform
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
    Inverse transform from training target space back to original units.

    Output:
    - SOC: percentage scale if normalize_soc=True.
    - SOH: original metadata unit, often 0-1 for old i10.
    """
    soc = soc_z.astype(np.float64).copy()
    soh = soh_z.astype(np.float64).copy()

    if zscore_normalize:
        if soc_norm is None or soh_norm is None:
            raise RuntimeError("zscore_normalize=True but soc_norm/soh_norm is missing.")

        soc = soc * float(soc_norm[1]) + float(soc_norm[0])
        soh = soh * float(soh_norm[1]) + float(soh_norm[0])

    if normalize_soc:
        soc = soc * 100.0

    return soc, soh


def load_target_norm_from_file(
    exp_dir: Path,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    path = exp_dir / "target_norm_train_only.npz"
    if not path.exists():
        return None, None

    d = np.load(path)
    soc_norm = (float(d["soc_mean"][0]), float(d["soc_std"][0]))
    soh_norm = (float(d["soh_mean"][0]), float(d["soh_std"][0]))
    return soc_norm, soh_norm


def compute_target_norm_from_train_meta(
    mtr: pd.DataFrame,
    soc_col: str,
    soh_col: str,
    normalize_soc: bool,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    soc = mtr[soc_col].astype(float).to_numpy(dtype=np.float64)
    if normalize_soc:
        soc = soc / 100.0

    soh = mtr[soh_col].astype(float).to_numpy(dtype=np.float64)

    soc_norm = (float(soc.mean()), float(soc.std() + 1e-8))
    soh_norm = (float(soh.mean()), float(soh.std() + 1e-8))

    return soc_norm, soh_norm


# =============================================================================
# Runtime config
# =============================================================================

def resolve_runtime_config(exp_dir: Path, verbose: bool = False) -> dict:
    """
    Load run_config.json if it exists.
    If absent, use defaults matching old i10 behavior.
    """
    cfg = DEFAULT_CONFIG.copy()

    run_cfg = load_json(exp_dir / "run_config.json")

    if run_cfg:
        for k in [
            "seed",
            "test_id_frac",
            "test_id_count",
            "u_start",
            "u_end",
            "drop_first_class",
            "soc_col",
            "soh_col",
            "use_pt_as_feature",
            "normalize_soc",
            "zscore_normalize",
            "width",
            "blocks",
            "drop2d",
            "head_dropout",
            "batch_size",
        ]:
            if k in run_cfg:
                cfg[k] = run_cfg[k]

        if "pulse_list" in run_cfg:
            cfg["pulse_list"] = run_cfg["pulse_list"]

        log("[CONFIG] Loaded run_config.json from experiment directory.", verbose)
    else:
        log("[CONFIG] run_config.json not found; using defaults for old i10 behavior.", verbose)

    return cfg


# =============================================================================
# Data loading and split
# =============================================================================

def load_u_norm_from_file(exp_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    path = exp_dir / "u41_norm_train_only.npz"
    if not path.exists():
        return None, None

    d = np.load(path)
    return d["u_mean"], d["u_std"]


def normalize_u_features(
    Xtr: np.ndarray,
    Xte: np.ndarray,
    exp_dir: Path,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    u_mean, u_std = load_u_norm_from_file(exp_dir)

    if u_mean is None or u_std is None:
        log("[NORM] u41_norm_train_only.npz not found. Computing from current train split.", verbose)
        u_mean = Xtr.mean(axis=0, keepdims=True)
        u_std = Xtr.std(axis=0, keepdims=True) + 1e-8

        np.savez_compressed(
            exp_dir / "u41_norm_train_only.npz",
            u_mean=u_mean.astype(np.float32),
            u_std=u_std.astype(np.float32),
        )
    else:
        log("[NORM] Loaded U1-U41 train-only normalization from file.", verbose)

    Xtr_norm = (Xtr - u_mean) / (u_std + 1e-8)
    Xte_norm = (Xte - u_mean) / (u_std + 1e-8)

    return Xtr_norm, Xte_norm


def get_split_name_from_label_mapping(exp_dir: Path) -> Optional[str]:
    label_mapping = load_json(exp_dir / "label_mapping.json")
    return label_mapping.get("split_name", None)


def get_classes_from_label_mapping(exp_dir: Path) -> Optional[List[str]]:
    label_mapping = load_json(exp_dir / "label_mapping.json")
    return label_mapping.get("classes", None)


def load_raw_data_and_split(
    data_root: str | Path,
    exp_dir: Path,
    cfg: dict,
    verbose: bool = False,
):
    cache_dir = ensure_dir(exp_dir / "cache")

    train_kwargs = {
        "data_root": str(data_root),
        "soc_list": list(range(5, 90, 5)),
        "pulse_list": list(map(int, cfg["pulse_list"])),
        "u_start": int(cfg["u_start"]),
        "u_end": int(cfg["u_end"]),
        "drop_first_class": bool(cfg["drop_first_class"]),
    }

    test_kwargs = {
        "data_root": str(data_root),
        "pulse_list": list(map(int, cfg["pulse_list"])),
        "u_start": int(cfg["u_start"]),
        "u_end": int(cfg["u_end"]),
        "drop_first_class": bool(cfg["drop_first_class"]),
    }

    log("[DATA] Loading or building raw_train cache...", verbose)
    Xtr_raw, ytr_raw, mtr_raw, tag_tr, hit_tr = load_or_build_cache(
        str(cache_dir),
        "raw_train",
        build_train_mix_soc_mix_pt,
        train_kwargs,
    )

    log("[DATA] Loading or building raw_test cache...", verbose)
    Xte_raw, yte_raw, mte_raw, tag_te, hit_te = load_or_build_cache(
        str(cache_dir),
        "raw_test",
        build_test_random_mix_pt,
        test_kwargs,
    )

    log(f"[CACHE] Train tag: {tag_tr} | hit={hit_tr}", verbose)
    log(f"[CACHE] Test  tag: {tag_te} | hit={hit_te}", verbose)

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
        raise RuntimeError("Empty train/test data after loading raw datasets.")

    split_name = get_split_name_from_label_mapping(exp_dir)
    split_path = exp_dir / "splits" / f"{split_name}.txt" if split_name else None

    if split_path is not None and split_path.exists():
        log(f"[SPLIT] Loading split from label_mapping split_name: {split_path}", verbose)
        with open(split_path, "r", encoding="utf-8") as f:
            test_ids = [line.strip() for line in f if line.strip()]
    else:
        fallback_split_path = (
            exp_dir
            / "splits"
            / f"testIDs_seed{cfg['seed']}_frac{cfg['test_id_frac']}.txt"
        )

        if fallback_split_path.exists():
            log(f"[SPLIT] Loading fallback split: {fallback_split_path}", verbose)
            with open(fallback_split_path, "r", encoding="utf-8") as f:
                test_ids = [line.strip() for line in f if line.strip()]
        else:
            log("[SPLIT] Existing split file not found. Creating split from IDs.", verbose)
            all_ids = pd.concat(
                [mtr_raw["ID"], mte_raw["ID"]],
                axis=0,
            ).astype(str).to_numpy()

            test_ids = pick_test_ids(
                all_ids=all_ids,
                test_id_frac=float(cfg["test_id_frac"]),
                test_id_count=int(cfg["test_id_count"]),
                seed=int(cfg["seed"]),
            )

            ensure_dir(exp_dir / "splits")
            with open(fallback_split_path, "w", encoding="utf-8") as f:
                for test_id in test_ids:
                    f.write(str(test_id) + "\n")

            log(f"[SPLIT] Saved new split: {fallback_split_path}", verbose)

    Xtr, ytr_str, mtr, Xte, yte_str, mte = apply_id_split(
        Xtr=Xtr_raw,
        ytr_str=ytr_raw,
        mtr=mtr_raw,
        Xte=Xte_raw,
        yte_str=yte_raw,
        mte=mte_raw,
        test_ids=np.array(test_ids, dtype=object),
    )

    train_classes = set(pd.Series(ytr_str).astype(str).tolist())
    mask_known = np.array([label in train_classes for label in yte_str], dtype=bool)

    if not mask_known.all():
        n_removed = int((~mask_known).sum())
        log(f"[WARN] Removing {n_removed} test samples with unseen labels.", verbose)
        Xte = Xte[mask_known]
        yte_str = yte_str[mask_known]
        mte = mte.loc[mask_known].reset_index(drop=True)

    log(f"[DATA] Final TRAIN samples = {len(ytr_str)}", verbose)
    log(f"[DATA] Final TEST  samples = {len(yte_str)}", verbose)

    return Xtr, ytr_str, mtr, Xte, yte_str, mte


# =============================================================================
# Model and dataset
# =============================================================================

def build_pt_norm(mtr: pd.DataFrame) -> Tuple[float, float]:
    if "pulse_ms" not in mtr.columns:
        return (0.0, 1.0)

    pt_train_ms = mtr["pulse_ms"].astype(float).to_numpy(dtype=np.float32)
    pt_log = np.log1p(pt_train_ms)

    return (float(pt_log.mean()), float(pt_log.std() + 1e-8))


def build_dataset_for_inference(
    X_u: np.ndarray,
    meta: pd.DataFrame,
    pt_norm: Optional[Tuple[float, float]],
    cfg: dict,
    soc_norm: Tuple[float, float],
    soh_norm: Tuple[float, float],
):
    return HierPulseDataset(
        X_u=X_u,
        y_cls=np.zeros(len(X_u), dtype=np.int64),
        meta=meta,
        soc_col=cfg["soc_col"],
        soh_col=cfg["soh_col"],
        use_pt_as_feature=bool(cfg["use_pt_as_feature"]),
        pt_norm=pt_norm,
        normalize_soc=bool(cfg["normalize_soc"]),
        zscore_normalize=bool(cfg["zscore_normalize"]),
        soc_norm=soc_norm if bool(cfg["zscore_normalize"]) else None,
        soh_norm=soh_norm if bool(cfg["zscore_normalize"]) else None,
        c_rate_combo=None,
    )


def get_num_classes(exp_dir: Path, ytr_str: np.ndarray) -> int:
    classes = get_classes_from_label_mapping(exp_dir)

    if classes is not None:
        return len(classes)

    le = LabelEncoder()
    le.fit(pd.Series(ytr_str).astype(str))
    return len(le.classes_)


def load_model(
    exp_dir: Path,
    cfg: dict,
    num_classes: int,
    device: str,
    verbose: bool = False,
) -> torch.nn.Module:
    model = Hier3HeadModel(
        num_classes=num_classes,
        width=int(cfg["width"]),
        blocks=int(cfg["blocks"]),
        drop2d=float(cfg["drop2d"]),
        use_pt_as_feature=bool(cfg["use_pt_as_feature"]),
        head_dropout=float(cfg["head_dropout"]),
    ).to(device)

    candidate_ckpts = [
        exp_dir / "checkpoints" / "finetune" / "best.pt",
        exp_dir / "checkpoints" / "stage2_soh" / "best.pt",
        exp_dir / "checkpoints" / "stage1_soc" / "best.pt",
    ]

    ckpt_path = None
    for p in candidate_ckpts:
        if p.exists():
            ckpt_path = p
            break

    if ckpt_path is None:
        raise FileNotFoundError(
            "No checkpoint found. Tried:\n"
            + "\n".join(str(p) for p in candidate_ckpts)
        )

    log(f"[MODEL] Loading checkpoint: {ckpt_path}", verbose)
    ckpt = _torch_load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model


# =============================================================================
# Prediction generation
# =============================================================================

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dataset,
    device: str,
    cfg: dict,
    soc_norm: Tuple[float, float],
    soh_norm: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
        soc_true_raw, soc_pred_raw, soh_true_raw, soh_pred_raw

    SOC is returned in percentage scale.
    SOH is returned in original metadata unit, often 0-1 for old i10.
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
    )

    soc_true_all, soc_pred_all = [], []
    soh_true_all, soh_pred_all = [], []

    for x3, pt, _, soc_z, soh_z in loader:
        x3 = x3.to(device)
        pt = pt.to(device)

        _, soc_pred_z, _, _, soh_pred_z, _ = model(
            x_img=x3,
            x_pt=pt,
            soc_tf=None,
            n_mc=int(cfg["n_mc_test"]),
        )

        soc_true_np = soc_z.detach().cpu().numpy().reshape(-1)
        soh_true_np = soh_z.detach().cpu().numpy().reshape(-1)
        soc_pred_np = soc_pred_z.detach().cpu().numpy().reshape(-1)
        soh_pred_np = soh_pred_z.detach().cpu().numpy().reshape(-1)

        soc_true_raw, soh_true_raw = inverse_targets(
            soc_true_np,
            soh_true_np,
            soc_norm=soc_norm,
            soh_norm=soh_norm,
            normalize_soc=bool(cfg["normalize_soc"]),
            zscore_normalize=bool(cfg["zscore_normalize"]),
        )
        soc_pred_raw, soh_pred_raw = inverse_targets(
            soc_pred_np,
            soh_pred_np,
            soc_norm=soc_norm,
            soh_norm=soh_norm,
            normalize_soc=bool(cfg["normalize_soc"]),
            zscore_normalize=bool(cfg["zscore_normalize"]),
        )

        soc_true_all.append(soc_true_raw)
        soc_pred_all.append(soc_pred_raw)
        soh_true_all.append(soh_true_raw)
        soh_pred_all.append(soh_pred_raw)

    soc_true = np.concatenate(soc_true_all)
    soc_pred = np.concatenate(soc_pred_all)
    soh_true = np.concatenate(soh_true_all)
    soh_pred = np.concatenate(soh_pred_all)

    return soc_true, soc_pred, soh_true, soh_pred


def load_existing_prediction_cache(
    pred_cache_dir: Path,
    verbose: bool = False,
) -> Optional[Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]]:
    required = {
        ("soc", "train"): pred_cache_dir / "soc_train_predictions.npz",
        ("soc", "test"): pred_cache_dir / "soc_test_predictions.npz",
        ("soh", "train"): pred_cache_dir / "soh_train_predictions.npz",
        ("soh", "test"): pred_cache_dir / "soh_test_predictions.npz",
    }

    if not all(p.exists() for p in required.values()):
        return None

    log("[CACHE] Existing prediction npz files found. Loading directly.", verbose)

    predictions = {}

    for key, path in required.items():
        d = np.load(path)
        predictions[key] = (d["true"], d["pred"])

    return predictions


def generate_prediction_npz_files(
    data_root: str | Path,
    exp_dir: Path,
    pred_cache_dir: Path,
    cfg: dict,
    force_regenerate: bool,
    pt_norm_mode: str,
    verbose: bool = False,
) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    ensure_dir(pred_cache_dir)

    if not force_regenerate:
        cached = load_existing_prediction_cache(pred_cache_dir, verbose=verbose)
        if cached is not None:
            return cached

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[DEVICE] {device}", verbose)

    Xtr, ytr_str, mtr, Xte, yte_str, mte = load_raw_data_and_split(
        data_root=data_root,
        exp_dir=exp_dir,
        cfg=cfg,
        verbose=verbose,
    )

    Xtr_norm, Xte_norm = normalize_u_features(
        Xtr=Xtr,
        Xte=Xte,
        exp_dir=exp_dir,
        verbose=verbose,
    )

    soc_norm, soh_norm = load_target_norm_from_file(exp_dir)

    if soc_norm is None or soh_norm is None:
        log("[NORM] target_norm_train_only.npz not found. Computing from current train split.", verbose)
        soc_norm, soh_norm = compute_target_norm_from_train_meta(
            mtr=mtr,
            soc_col=cfg["soc_col"],
            soh_col=cfg["soh_col"],
            normalize_soc=bool(cfg["normalize_soc"]),
        )

        np.savez_compressed(
            exp_dir / "target_norm_train_only.npz",
            soc_mean=np.array([soc_norm[0]], dtype=np.float32),
            soc_std=np.array([soc_norm[1]], dtype=np.float32),
            soh_mean=np.array([soh_norm[0]], dtype=np.float32),
            soh_std=np.array([soh_norm[1]], dtype=np.float32),
        )

    log(f"[NORM] SOC norm = mean {soc_norm[0]:.6f}, std {soc_norm[1]:.6f}", verbose)
    log(f"[NORM] SOH norm = mean {soh_norm[0]:.6f}, std {soh_norm[1]:.6f}", verbose)
    log(f"[CONFIG] normalize_soc={cfg['normalize_soc']} | zscore_normalize={cfg['zscore_normalize']}", verbose)

    num_classes = get_num_classes(exp_dir, ytr_str)
    model = load_model(
        exp_dir=exp_dir,
        cfg=cfg,
        num_classes=num_classes,
        device=device,
        verbose=verbose,
    )

    if pt_norm_mode == "train":
        pt_norm = build_pt_norm(mtr)
        log(f"[NORM] pulse width train-only norm = mean {pt_norm[0]:.6f}, std {pt_norm[1]:.6f}", verbose)
    elif pt_norm_mode == "none":
        pt_norm = None
        log("[NORM] pt_norm=None, dataset computes pulse-width norm from its own meta.", verbose)
    else:
        raise ValueError("pt_norm_mode must be 'train' or 'none'.")

    ds_tr = build_dataset_for_inference(
        X_u=Xtr_norm,
        meta=mtr,
        pt_norm=pt_norm,
        cfg=cfg,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
    )

    ds_te = build_dataset_for_inference(
        X_u=Xte_norm,
        meta=mte,
        pt_norm=pt_norm,
        cfg=cfg,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
    )

    soc_true_tr, soc_pred_tr, soh_true_tr, soh_pred_tr = run_inference(
        model=model,
        dataset=ds_tr,
        device=device,
        cfg=cfg,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
    )

    soc_true_te, soc_pred_te, soh_true_te, soh_pred_te = run_inference(
        model=model,
        dataset=ds_te,
        device=device,
        cfg=cfg,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
    )

    np.savez_compressed(pred_cache_dir / "soc_train_predictions.npz", true=soc_true_tr, pred=soc_pred_tr)
    np.savez_compressed(pred_cache_dir / "soc_test_predictions.npz", true=soc_true_te, pred=soc_pred_te)
    np.savez_compressed(pred_cache_dir / "soh_train_predictions.npz", true=soh_true_tr, pred=soh_pred_tr)
    np.savez_compressed(pred_cache_dir / "soh_test_predictions.npz", true=soh_true_te, pred=soh_pred_te)

    log(f"[SAVE] Prediction caches written to: {pred_cache_dir}", verbose)

    return {
        ("soc", "train"): (soc_true_tr, soc_pred_tr),
        ("soc", "test"): (soc_true_te, soc_pred_te),
        ("soh", "train"): (soh_true_tr, soh_pred_tr),
        ("soh", "test"): (soh_true_te, soh_pred_te),
    }


# =============================================================================
# Plotting
# =============================================================================

def convert_soh_to_percent_if_needed(
    t: np.ndarray,
    p: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert SOH to percentage only if it is in 0-1 scale.
    """
    t2 = t.astype(np.float64).copy()
    p2 = p.astype(np.float64).copy()

    max_true = np.nanmax(np.abs(t2))
    max_pred = np.nanmax(np.abs(p2))

    if max_true <= 2.0 and max_pred <= 2.0:
        t2 *= 100.0
        p2 *= 100.0

    return t2, p2


def prepare_plot_values(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    target: str,
) -> Tuple[np.ndarray, np.ndarray, str, str, List[float]]:
    target = target.lower()

    t = true_values.astype(np.float64).copy()
    p = pred_values.astype(np.float64).copy()

    if target == "soh":
        t, p = convert_soh_to_percent_if_needed(t, p)
        xlabel = "Measured SOH (%)"
        ylabel = "Predicted SOH (%)"

        vmin = min(float(np.nanmin(t)), float(np.nanmin(p)))
        vmax = max(float(np.nanmax(t)), float(np.nanmax(p)))
        margin = (vmax - vmin) * 0.05 if vmax > vmin else 1.0
        lims = [vmin - margin, vmax + margin]

    elif target == "soc":
        xlabel = "Measured SOC (%)"
        ylabel = "Predicted SOC (%)"
        lims = [0, 100]

    else:
        raise ValueError("target must be 'soc' or 'soh'.")

    return t, p, xlabel, ylabel, lims


def compute_metrics(t: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    ape = np.abs((t - p) / np.maximum(np.abs(t), 1e-8)) * 100.0
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    medape = float(np.median(ape))
    return {"rmse": rmse, "medape": medape}


def get_plot_metrics(
    predictions: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    metrics = {}

    for target in ["soh", "soc"]:
        for split in ["train", "test"]:
            true_values, pred_values = predictions[(target, split)]
            t, p, _, _, _ = prepare_plot_values(true_values, pred_values, target)
            metrics[(target, split)] = compute_metrics(t, p)

    return metrics


def print_metric_summary(
    metrics: Dict[Tuple[str, str], Dict[str, float]],
) -> None:
    print("\n[METRICS] Figure 3a")
    print("Target  Split   Median APE (%)   RMSE")
    print("----------------------------------------")

    for target in ["soh", "soc"]:
        for split in ["train", "test"]:
            m = metrics[(target, split)]
            print(
                f"{target.upper():<7} {split:<7} "
                f"{m['medape']:>14.4f}   {m['rmse']:>8.4f}"
            )


def kde_scatter_on_ax(
    ax: plt.Axes,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    target: str,
    split: str,
    cfg: dict,
    metrics: Optional[Dict[str, float]] = None,
    show_title: bool = True,
    show_labels: bool = True,
):
    t, p, xlabel, ylabel, lims = prepare_plot_values(
        true_values=true_values,
        pred_values=pred_values,
        target=target,
    )

    if metrics is None:
        metrics = compute_metrics(t, p)

    xy = np.vstack([t, p])

    if xy.shape[1] < 3:
        raise RuntimeError(f"Not enough points for KDE: {xy.shape[1]}")

    z = gaussian_kde(xy, bw_method=cfg["bw_method"])(xy)

    idx = z.argsort()
    t_sorted = t[idx]
    p_sorted = p[idx]
    z_sorted = z[idx]

    z_min = float(z_sorted.min())
    z_max = float(z_sorted.max())
    z_norm = (z_sorted - z_min) / (z_max - z_min + 1e-10)

    cmap_custom = mcolors.LinearSegmentedColormap.from_list(
        f"{target}_{split}_cmap",
        cfg["hex_colors"],
    )

    colors_rgba = cmap_custom(z_norm)
    colors_rgba[:, 3] = 0.7 * z_norm + 0.2

    ax.scatter(
        t_sorted,
        p_sorted,
        c=colors_rgba,
        s=cfg["dot_size"],
        edgecolors="none",
    )

    ax.plot(
        lims,
        lims,
        color="#333333",
        linestyle=":",
        alpha=0.7,
        linewidth=1.5,
        zorder=0,
    )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, linestyle=":", alpha=0.25)
    ax.tick_params(axis="both", which="major", labelsize=11)

    if show_labels:
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

    if show_title:
        ax.set_title(
            f"{target.upper()} {split.capitalize()}\nMedian APE = {metrics['medape']:.3f}%",
            fontsize=12,
            pad=10,
        )

    return cmap_custom, z_min, z_max


def plot_single_full_and_pure(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    target: str,
    split: str,
    fig_out_dir: Path,
    cfg: dict,
    dpi: int,
    metrics: Dict[str, float],
):
    plt.rcParams["font.sans-serif"] = [cfg["font"]]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7.8, 6.2))

    cmap_custom, z_min, z_max = kde_scatter_on_ax(
        ax=ax,
        true_values=true_values,
        pred_values=pred_values,
        target=target,
        split=split,
        cfg=cfg,
        metrics=metrics,
        show_title=True,
        show_labels=True,
    )

    sm = plt.cm.ScalarMappable(
        cmap=cmap_custom,
        norm=plt.Normalize(vmin=z_min, vmax=z_max),
    )
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Point density", rotation=270, labelpad=15, fontsize=11)

    full_base = fig_out_dir / f"fig3a_{target}_{split}_full"
    save_figure(fig, full_base, dpi=dpi, transparent=False)
    plt.close(fig)

    fig_pure, ax_pure = plt.subplots(figsize=(6, 6))

    kde_scatter_on_ax(
        ax=ax_pure,
        true_values=true_values,
        pred_values=pred_values,
        target=target,
        split=split,
        cfg=cfg,
        metrics=metrics,
        show_title=False,
        show_labels=False,
    )

    ax_pure.set_axis_off()

    pure_base = fig_out_dir / f"fig3a_{target}_{split}_pure"
    save_figure(fig_pure, pure_base, dpi=dpi, transparent=True)
    plt.close(fig_pure)


def plot_combined_2x2(
    predictions: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
    fig_out_dir: Path,
    cfg: dict,
    dpi: int,
    all_metrics: Dict[Tuple[str, str], Dict[str, float]],
):
    plt.rcParams["font.sans-serif"] = [cfg["font"]]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    layout = [
        ("soh", "train", axes[0, 0]),
        ("soh", "test", axes[0, 1]),
        ("soc", "train", axes[1, 0]),
        ("soc", "test", axes[1, 1]),
    ]

    for target, split, ax in layout:
        true_values, pred_values = predictions[(target, split)]
        kde_scatter_on_ax(
            ax=ax,
            true_values=true_values,
            pred_values=pred_values,
            target=target,
            split=split,
            cfg=cfg,
            metrics=all_metrics[(target, split)],
            show_title=True,
            show_labels=True,
        )

    fig.tight_layout()

    out_base = fig_out_dir / "fig3a_prediction_scatter_2x2"
    save_figure(fig, out_base, dpi=dpi, transparent=False)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Figure 3a prediction scatter plots."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root folder of raw battery data.",
    )

    parser.add_argument(
        "--exp_dir",
        type=str,
        default="results/proposed_framework",
        help=(
            "Experiment directory containing cache/splits/checkpoints/"
            "u41_norm_train_only.npz/label_mapping.json. "
            "For old results, set this to results/i10_normalization_flow."
        ),
    )

    parser.add_argument(
        "--pred_cache_dir",
        type=str,
        default="results/figures/cache/fig3a_prediction_scatter",
        help="Directory to save or load prediction npz files.",
    )

    parser.add_argument(
        "--fig_out_dir",
        type=str,
        default="results/figures/main/fig3a",
        help="Directory to save figures.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure dpi.",
    )

    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        help="Regenerate prediction npz files even if they already exist.",
    )

    parser.add_argument(
        "--force_zscore",
        action="store_true",
        help="Force zscore_normalize=True, useful for old i10 checkpoints.",
    )

    parser.add_argument(
        "--n_mc_test",
        type=int,
        default=None,
        help="Override Monte Carlo sample number for inference.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override inference batch size.",
    )

    parser.add_argument(
        "--pt_norm_mode",
        type=str,
        default="train",
        choices=["train", "none"],
        help=(
            "Pulse-width normalization mode. "
            "'train' uses train-only pt_norm; "
            "'none' matches old temporary figure scripts where pt_norm was not passed."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed data-loading and inference logs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    exp_dir = Path(args.exp_dir)
    pred_cache_dir = ensure_dir(args.pred_cache_dir)
    fig_out_dir = ensure_dir(args.fig_out_dir)

    cfg = resolve_runtime_config(exp_dir, verbose=args.verbose)

    if args.force_zscore:
        cfg["zscore_normalize"] = True

    if args.n_mc_test is not None:
        cfg["n_mc_test"] = int(args.n_mc_test)

    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)

    set_random_seed(int(cfg["seed"]))

    print("[FIGURE] Figure 3a prediction scatter")
    print(f"[EXP_DIR] {exp_dir}")
    print(f"[PRED_CACHE] {pred_cache_dir}")
    print(f"[FIGURES] {fig_out_dir}")

    predictions = generate_prediction_npz_files(
        data_root=args.data_root,
        exp_dir=exp_dir,
        pred_cache_dir=pred_cache_dir,
        cfg=cfg,
        force_regenerate=bool(args.force_regenerate),
        pt_norm_mode=args.pt_norm_mode,
        verbose=bool(args.verbose),
    )

    all_metrics = get_plot_metrics(predictions)
    print_metric_summary(all_metrics)

    for target in ["soh", "soc"]:
        for split in ["train", "test"]:
            true_values, pred_values = predictions[(target, split)]

            plot_single_full_and_pure(
                true_values=true_values,
                pred_values=pred_values,
                target=target,
                split=split,
                fig_out_dir=fig_out_dir,
                cfg=cfg,
                dpi=int(args.dpi),
                metrics=all_metrics[(target, split)],
            )

    plot_combined_2x2(
        predictions=predictions,
        fig_out_dir=fig_out_dir,
        cfg=cfg,
        dpi=int(args.dpi),
        all_metrics=all_metrics,
    )

    print("\n[DONE] Figure 3a saved successfully.")
    print(f"[PRED CACHE] {pred_cache_dir}")
    print(f"[FIGURES] {fig_out_dir}")


if __name__ == "__main__":
    main()