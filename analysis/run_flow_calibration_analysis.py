# -*- coding: utf-8 -*-
"""
run_probabilistic_evaluation.py

Unified probabilistic evaluation for:
    1. Conditional normalizing-flow model
    2. Gaussian probabilistic baseline

Metrics:
    - Negative log-likelihood (NLL)
    - Continuous Ranked Probability Score (CRPS)
    - Empirical coverage at 50%, 80%, 90%, and 95%
    - Mean prediction-interval width at 50%, 80%, 90%, and 95%
    - Absolute coverage error at 50%, 80%, 90%, and 95%

Outputs:
    results/analysis/probabilistic_evaluation/
        probabilistic_metrics_long.csv
        supplementary_table14_probabilistic.csv
        per_sample_probabilistic_metrics.csv
        run_config.json

Run:
    python analysis/run_probabilistic_evaluation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.cache import load_or_build_cache, drop_nan_inf_rows
from proposed_framework.data.build_dataset import (
    build_train_mix_soc_mix_pt,
    build_test_random_mix_pt,
    pick_test_ids,
    apply_id_split,
)
from proposed_framework.data.pulse_dataset import HierPulseDataset
from proposed_framework.models.hierarchical_model import Hier3HeadModel
from analysis.train_calibration_baseline import (
    CalibrationBaselineDataset,
    GaussianCalibrationBaseline,
)

# =============================================================================
# Paths
# =============================================================================
DATA_ROOT = PROJECT_ROOT / "data"
PROPOSED_DIR = PROJECT_ROOT / "results" / "proposed_framework"
FLOW_CKPT = PROPOSED_DIR / "checkpoints" / "finetune" / "best.pt"
GAUSSIAN_DIR = PROJECT_ROOT / "results" / "calibration_baseline"
GAUSSIAN_CKPT = GAUSSIAN_DIR / "checkpoints" / "best.pt"
SAVE_DIR = PROJECT_ROOT / "results" / "analysis" / "probabilistic_evaluation"

# =============================================================================
# Data configuration
# =============================================================================
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

# =============================================================================
# Model defaults
# =============================================================================
FLOW_WIDTH = 32
FLOW_BLOCKS = 4
FLOW_DROP2D = 0.0
FLOW_HEAD_DROPOUT = 0.2
FLOW_LAYERS = 6
FLOW_BINS = 8
FLOW_TAIL_BOUND = 3.0

GAUSSIAN_WIDTH = 32
GAUSSIAN_BLOCKS = 4
GAUSSIAN_DROP2D = 0.0
GAUSSIAN_HEAD_DROPOUT = 0.2
GAUSSIAN_SOC_HIDDEN = 64
GAUSSIAN_SOH_HIDDEN = 64

# =============================================================================
# Evaluation configuration
# =============================================================================
FLOW_NUM_SAMPLES = 1000
INTERVAL_LEVELS = [0.50, 0.80, 0.90, 0.95]
SIGMA_MIN = 1e-6


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def torch_load_compatible(path: str | Path, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(ckpt) -> dict:
    if not isinstance(ckpt, dict):
        raise RuntimeError("Unsupported checkpoint format.")
    for key in ["model", "model_state_dict", "state_dict"]:
        if key in ckpt:
            return ckpt[key]
    return ckpt


def extract_run_config(ckpt) -> dict:
    if isinstance(ckpt, dict) and isinstance(ckpt.get("run_config", {}), dict):
        return ckpt.get("run_config", {})
    return {}


def cfg_int(config: dict, key: str, default: int) -> int:
    return int(config.get(key, default))


def cfg_float(config: dict, key: str, default: float) -> float:
    return float(config.get(key, default))


def cfg_bool(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def inverse_single_target(
    values_z: np.ndarray,
    target: str,
    soc_norm: Tuple[float, float],
    soh_norm: Tuple[float, float],
) -> np.ndarray:
    values = np.asarray(values_z, dtype=np.float64)
    if target == "SOC":
        raw = values * float(soc_norm[1]) + float(soc_norm[0])
        return raw * 100.0 if NORMALIZE_SOC else raw
    if target == "SOH":
        return values * float(soh_norm[1]) + float(soh_norm[0])
    raise ValueError(f"Unknown target: {target}")


def target_scale_factor(
    target: str,
    soc_norm: Tuple[float, float],
    soh_norm: Tuple[float, float],
) -> float:
    if target == "SOC":
        scale = float(soc_norm[1])
        return scale * 100.0 if NORMALIZE_SOC else scale
    if target == "SOH":
        return float(soh_norm[1])
    raise ValueError(f"Unknown target: {target}")


def normalize_flow_samples(samples: torch.Tensor, batch_size: int) -> torch.Tensor:
    if not isinstance(samples, torch.Tensor):
        samples = torch.as_tensor(samples)
    if samples.ndim == 3:
        if samples.shape[-1] == 1:
            samples = samples.squeeze(-1)
        elif samples.shape[1] == 1:
            samples = samples.squeeze(1)
        else:
            raise RuntimeError(f"Unexpected 3D flow sample shape: {tuple(samples.shape)}")
    if samples.ndim != 2:
        raise RuntimeError(f"Expected 2D flow samples, got {tuple(samples.shape)}")
    if samples.shape[0] == batch_size:
        return samples
    if samples.shape[1] == batch_size:
        return samples.transpose(0, 1)
    raise RuntimeError(
        f"Cannot identify batch dimension in {tuple(samples.shape)} for batch={batch_size}"
    )


def sample_from_flow(flow_module, context: torch.Tensor, num_samples: int) -> torch.Tensor:
    try:
        samples = flow_module.sample(num_samples=int(num_samples), context=context)
    except TypeError:
        samples = flow_module.sample(int(num_samples), context=context)
    return normalize_flow_samples(samples, int(context.shape[0]))


def flow_log_prob(flow_module, target: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    target = target.view(-1, 1)
    if hasattr(flow_module, "log_prob"):
        try:
            return flow_module.log_prob(target, context=context).view(-1)
        except TypeError:
            return flow_module.log_prob(inputs=target, context=context).view(-1)
    if hasattr(flow_module, "flow"):
        try:
            return flow_module.flow.log_prob(target, context=context).view(-1)
        except TypeError:
            return flow_module.flow.log_prob(inputs=target, context=context).view(-1)
    raise AttributeError("Flow module does not expose log_prob().")


def gaussian_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), SIGMA_MIN, None)
    z = (y - mu) / sigma
    z_t = torch.tensor(z, dtype=torch.float64)
    normal = torch.distributions.Normal(0.0, 1.0)
    cdf = normal.cdf(z_t).numpy()
    pdf = torch.exp(normal.log_prob(z_t)).numpy()
    crps = sigma * (
        z * (2.0 * cdf - 1.0)
        + 2.0 * pdf
        - 1.0 / np.sqrt(np.pi)
    )
    return np.maximum(crps, 0.0)


def sample_crps(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] != len(y):
        raise ValueError(f"Expected samples [N,S], got {samples.shape}, N={len(y)}")
    s = samples.shape[1]
    if s < 2:
        raise ValueError("At least two samples are required for CRPS.")
    term_obs = np.mean(np.abs(samples - y[:, None]), axis=1)
    x = np.sort(samples, axis=1)
    weights = 2.0 * np.arange(1, s + 1) - s - 1.0
    pairwise_expectation = 2.0 * np.sum(x * weights[None, :], axis=1) / float(s ** 2)
    return np.maximum(term_obs - 0.5 * pairwise_expectation, 0.0)


def interval_metrics_from_samples(
    y: np.ndarray,
    samples: np.ndarray,
    levels: list[float],
) -> Dict[str, np.ndarray]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    samples = np.asarray(samples, dtype=np.float64)
    out: Dict[str, np.ndarray] = {}
    for level in levels:
        alpha = 1.0 - level
        key = int(round(level * 100))
        lower = np.quantile(samples, alpha / 2.0, axis=1)
        upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=1)
        out[f"lower_{key}"] = lower
        out[f"upper_{key}"] = upper
        out[f"covered_{key}"] = ((y >= lower) & (y <= upper)).astype(float)
        out[f"width_{key}"] = upper - lower
    return out


def interval_metrics_gaussian(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    levels: list[float],
) -> Dict[str, np.ndarray]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    sigma = np.clip(np.asarray(sigma, dtype=np.float64).reshape(-1), SIGMA_MIN, None)
    normal = torch.distributions.Normal(0.0, 1.0)
    out: Dict[str, np.ndarray] = {}
    for level in levels:
        alpha = 1.0 - level
        key = int(round(level * 100))
        z = float(normal.icdf(torch.tensor(1.0 - alpha / 2.0, dtype=torch.float64)))
        lower = mu - z * sigma
        upper = mu + z * sigma
        out[f"lower_{key}"] = lower
        out[f"upper_{key}"] = upper
        out[f"covered_{key}"] = ((y >= lower) & (y <= upper)).astype(float)
        out[f"width_{key}"] = upper - lower
    return out


def build_flow_model(num_classes: int, device: str):
    ckpt = torch_load_compatible(FLOW_CKPT, device)
    config = extract_run_config(ckpt)
    base_kwargs = dict(
        num_classes=num_classes,
        width=cfg_int(config, "width", FLOW_WIDTH),
        blocks=cfg_int(config, "blocks", FLOW_BLOCKS),
        drop2d=cfg_float(config, "drop2d", FLOW_DROP2D),
        use_pt_as_feature=cfg_bool(config, "use_pt_as_feature", USE_PT_AS_FEATURE),
        head_dropout=cfg_float(config, "head_dropout", FLOW_HEAD_DROPOUT),
    )
    try:
        model = Hier3HeadModel(
            **base_kwargs,
            flow_layers=cfg_int(config, "flow_layers", FLOW_LAYERS),
            flow_bins=cfg_int(config, "flow_bins", FLOW_BINS),
            flow_tail_bound=cfg_float(config, "flow_tail_bound", FLOW_TAIL_BOUND),
        ).to(device)
    except TypeError:
        model = Hier3HeadModel(**base_kwargs).to(device)
    model.load_state_dict(extract_state_dict(ckpt), strict=True)
    model.eval()
    print(f"[MODEL] Loaded conditional flow: {FLOW_CKPT}")
    return model, config


def build_gaussian_model(num_classes: int, device: str):
    ckpt = torch_load_compatible(GAUSSIAN_CKPT, device)
    config = extract_run_config(ckpt)
    model = GaussianCalibrationBaseline(
        num_classes=num_classes,
        width=cfg_int(config, "width", GAUSSIAN_WIDTH),
        blocks=cfg_int(config, "blocks", GAUSSIAN_BLOCKS),
        drop2d=cfg_float(config, "drop2d", GAUSSIAN_DROP2D),
        use_pt_as_feature=cfg_bool(config, "use_pt_as_feature", USE_PT_AS_FEATURE),
        soc_hidden=cfg_int(config, "soc_hidden", GAUSSIAN_SOC_HIDDEN),
        soh_hidden=cfg_int(config, "soh_hidden", GAUSSIAN_SOH_HIDDEN),
        head_dropout=cfg_float(config, "head_dropout", GAUSSIAN_HEAD_DROPOUT),
    ).to(device)
    model.load_state_dict(extract_state_dict(ckpt), strict=True)
    model.eval()
    print(f"[MODEL] Loaded Gaussian baseline: {GAUSSIAN_CKPT}")
    return model, config


def build_shared_test_loaders():
    cache_dir = PROPOSED_DIR / "cache"
    split_dir = PROPOSED_DIR / "splits"
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs = dict(
        data_root=str(DATA_ROOT),
        soc_list=list(range(5, 90, 5)),
        pulse_list=list(map(int, PULSE_LIST)),
        u_start=U_START,
        u_end=U_END,
        drop_first_class=DROP_FIRST_CLASS,
    )
    Xtr_raw, ytr_raw, mtr_raw, tag_tr, hit_tr = load_or_build_cache(
        str(cache_dir), "raw_train", build_train_mix_soc_mix_pt, train_kwargs
    )

    test_kwargs = dict(
        data_root=str(DATA_ROOT),
        pulse_list=list(map(int, PULSE_LIST)),
        u_start=U_START,
        u_end=U_END,
        drop_first_class=DROP_FIRST_CLASS,
    )
    Xte_raw, yte_raw, mte_raw, tag_te, hit_te = load_or_build_cache(
        str(cache_dir), "raw_test", build_test_random_mix_pt, test_kwargs
    )
    print(f"[CACHE] Train tag: {tag_tr} | hit={hit_tr}")
    print(f"[CACHE] Test  tag: {tag_te} | hit={hit_te}")

    Xtr_raw, ytr_raw, mtr_raw = drop_nan_inf_rows(Xtr_raw, ytr_raw, mtr_raw, "RAW_TRAIN")
    Xte_raw, yte_raw, mte_raw = drop_nan_inf_rows(Xte_raw, yte_raw, mte_raw, "RAW_TEST")

    all_ids = pd.concat([mtr_raw["ID"], mte_raw["ID"]], axis=0).astype(str).to_numpy()
    if TEST_ID_COUNT > 0:
        split_name = f"testIDs_seed{SEED}_n{TEST_ID_COUNT}"
    else:
        split_name = f"testIDs_seed{SEED}_frac{TEST_ID_FRAC}"
    split_path = split_dir / f"{split_name}.txt"

    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            test_ids = np.array([line.strip() for line in f if line.strip()])
        print(f"[SPLIT] Loaded existing split: {split_path}")
    else:
        test_ids = pick_test_ids(
            all_ids=all_ids,
            test_id_frac=TEST_ID_FRAC,
            test_id_count=TEST_ID_COUNT,
            seed=SEED,
        )
        with open(split_path, "w", encoding="utf-8") as f:
            for test_id in test_ids:
                f.write(str(test_id) + "\n")
        print(f"[SPLIT] Saved new split: {split_path}")

    Xtr, ytr_str, mtr, Xte, yte_str, mte = apply_id_split(
        Xtr=Xtr_raw,
        ytr_str=ytr_raw,
        mtr=mtr_raw,
        Xte=Xte_raw,
        yte_str=yte_raw,
        mte=mte_raw,
        test_ids=test_ids,
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(ytr_str)
    train_classes = set(label_encoder.classes_.tolist())
    mask_known = np.array([label in train_classes for label in yte_str], dtype=bool)
    if not mask_known.all():
        print(f"[WARN] Removing {(~mask_known).sum()} unseen-label test samples.")
        Xte = Xte[mask_known]
        yte_str = yte_str[mask_known]
        mte = mte.loc[mask_known].reset_index(drop=True)
    yte_cls = label_encoder.transform(yte_str)
    num_classes = len(label_encoder.classes_)

    soc_train = mtr[SOC_COL].astype(float).to_numpy(dtype=np.float64)
    if NORMALIZE_SOC:
        soc_train = soc_train / 100.0
    soc_norm = (float(soc_train.mean()), float(soc_train.std() + 1e-8))

    soh_train = mtr[SOH_COL].astype(float).to_numpy(dtype=np.float64)
    soh_norm = (float(soh_train.mean()), float(soh_train.std() + 1e-8))

    if "pulse_ms" in mtr.columns:
        pt_col = "pulse_ms"
    elif "pulse_width_ms" in mtr.columns:
        pt_col = "pulse_width_ms"
    else:
        raise RuntimeError("No pulse-width column found.")
    pt_log = np.log1p(mtr[pt_col].astype(float).to_numpy(dtype=np.float64))
    pt_norm = (float(pt_log.mean()), float(pt_log.std() + 1e-8))

    flow_norm_path = PROPOSED_DIR / "u41_norm_train_only.npz"
    if flow_norm_path.exists():
        norm_obj = np.load(flow_norm_path)
        u_mean = norm_obj["u_mean"]
        u_std = norm_obj["u_std"]
        print(f"[NORM] Loaded flow U normalization: {flow_norm_path}")
    else:
        u_mean = Xtr.mean(axis=0, keepdims=True)
        u_std = Xtr.std(axis=0, keepdims=True) + 1e-8
        print("[WARN] Flow U normalization missing; computed train-only stats.")

    ds_flow = HierPulseDataset(
        X_u=(Xte - u_mean) / u_std,
        y_cls=yte_cls,
        meta=mte,
        soc_col=SOC_COL,
        soh_col=SOH_COL,
        pt_col=pt_col,
        use_pt_as_feature=USE_PT_AS_FEATURE,
        pt_norm=pt_norm,
        normalize_soc=NORMALIZE_SOC,
        zscore_normalize=ZSCORE_NORMALIZE,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
    )

    ds_gaussian = CalibrationBaselineDataset(
        X_u=Xte.copy(),
        y_cls=yte_cls,
        meta=mte,
        soc_col=SOC_COL,
        soh_col=SOH_COL,
        pt_col=pt_col,
        use_pt_as_feature=USE_PT_AS_FEATURE,
        pt_norm=pt_norm,
        normalize_soc=NORMALIZE_SOC,
        zscore_normalize=ZSCORE_NORMALIZE,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
    )

    dl_flow = DataLoader(ds_flow, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    dl_gaussian = DataLoader(ds_gaussian, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    info = dict(
        num_classes=int(num_classes),
        n_test=int(len(ds_flow)),
        split_path=str(split_path),
        soc_norm=soc_norm,
        soh_norm=soh_norm,
        pt_norm=pt_norm,
    )
    print(f"[DATA] Test samples: {info['n_test']}")
    print(f"[DATA] Num classes: {info['num_classes']}")
    return dl_flow, dl_gaussian, info


@torch.no_grad()
def evaluate_flow(model, loader, device: str, soc_norm, soh_norm):
    model.eval()
    store = {
        "SOC": {"true_z": [], "log_prob": [], "samples_z": []},
        "SOH": {"true_z": [], "log_prob": [], "samples_z": []},
    }
    for batch_idx, (x3, pt, _, soc, soh) in enumerate(loader):
        x3 = x3.to(device)
        pt = pt.to(device)
        soc = soc.to(device).view(-1, 1)
        soh = soh.to(device).view(-1, 1)

        _, _, _, cond_soc, _, cond_soh = model(
            x3,
            pt,
            soc_tf=None,
            n_mc=FLOW_NUM_SAMPLES,
        )

        soc_lp = flow_log_prob(model.soc_flow, soc, cond_soc)
        soh_lp = flow_log_prob(model.soh_flow, soh, cond_soh)
        soc_samples = sample_from_flow(model.soc_flow, cond_soc, FLOW_NUM_SAMPLES)
        soh_samples = sample_from_flow(model.soh_flow, cond_soh, FLOW_NUM_SAMPLES)

        store["SOC"]["true_z"].append(soc.cpu().numpy().reshape(-1))
        store["SOC"]["log_prob"].append(soc_lp.cpu().numpy().reshape(-1))
        store["SOC"]["samples_z"].append(soc_samples.cpu().numpy())
        store["SOH"]["true_z"].append(soh.cpu().numpy().reshape(-1))
        store["SOH"]["log_prob"].append(soh_lp.cpu().numpy().reshape(-1))
        store["SOH"]["samples_z"].append(soh_samples.cpu().numpy())

        if batch_idx == 0 or (batch_idx + 1) % 20 == 0:
            print(f"[FLOW] Processed batch {batch_idx + 1}/{len(loader)}")

    result = {}
    for target in ["SOC", "SOH"]:
        true_z = np.concatenate(store[target]["true_z"])
        log_prob = np.concatenate(store[target]["log_prob"])
        samples_z = np.concatenate(store[target]["samples_z"], axis=0)
        result[target] = dict(
            true_z=true_z,
            true_raw=inverse_single_target(true_z, target, soc_norm, soh_norm),
            log_prob=log_prob,
            samples_z=samples_z,
            samples_raw=inverse_single_target(samples_z, target, soc_norm, soh_norm),
        )
    return result


@torch.no_grad()
def evaluate_gaussian(model, loader, device: str, soc_norm, soh_norm):
    model.eval()
    store = {
        "SOC": {"true_z": [], "mu_z": [], "sigma_z": [], "log_prob": []},
        "SOH": {"true_z": [], "mu_z": [], "sigma_z": [], "log_prob": []},
    }
    for batch_idx, (x3, pt, _, soc, soh) in enumerate(loader):
        x3 = x3.to(device)
        pt = pt.to(device)
        soc = soc.to(device).view(-1)
        soh = soh.to(device).view(-1)

        _, soc_mu, _, soc_sigma, soh_mu, _, soh_sigma = model(x3, pt)
        soc_sigma = torch.clamp(soc_sigma, min=SIGMA_MIN)
        soh_sigma = torch.clamp(soh_sigma, min=SIGMA_MIN)
        soc_dist = torch.distributions.Normal(soc_mu, soc_sigma)
        soh_dist = torch.distributions.Normal(soh_mu, soh_sigma)

        for target, y, mu, sigma, dist in [
            ("SOC", soc, soc_mu, soc_sigma, soc_dist),
            ("SOH", soh, soh_mu, soh_sigma, soh_dist),
        ]:
            store[target]["true_z"].append(y.cpu().numpy())
            store[target]["mu_z"].append(mu.cpu().numpy())
            store[target]["sigma_z"].append(sigma.cpu().numpy())
            store[target]["log_prob"].append(dist.log_prob(y).cpu().numpy())

        if batch_idx == 0 or (batch_idx + 1) % 20 == 0:
            print(f"[GAUSSIAN] Processed batch {batch_idx + 1}/{len(loader)}")

    result = {}
    for target in ["SOC", "SOH"]:
        true_z = np.concatenate(store[target]["true_z"])
        mu_z = np.concatenate(store[target]["mu_z"])
        sigma_z = np.concatenate(store[target]["sigma_z"])
        log_prob = np.concatenate(store[target]["log_prob"])
        scale = target_scale_factor(target, soc_norm, soh_norm)
        result[target] = dict(
            true_z=true_z,
            true_raw=inverse_single_target(true_z, target, soc_norm, soh_norm),
            mu_z=mu_z,
            sigma_z=sigma_z,
            mu_raw=inverse_single_target(mu_z, target, soc_norm, soh_norm),
            sigma_raw=sigma_z * scale,
            log_prob=log_prob,
        )
    return result


def summarize_flow_target(target: str, data: Dict[str, np.ndarray]):
    y = data["true_raw"]
    samples = data["samples_raw"]
    log_prob = data["log_prob"]
    crps = sample_crps(y, samples)
    intervals = interval_metrics_from_samples(y, samples, INTERVAL_LEVELS)

    row = dict(
        model="Conditional flow",
        target=target,
        n_test=len(y),
        nll_normalized=float(-np.mean(log_prob)),
        crps_raw=float(np.mean(crps)),
    )
    per_sample = pd.DataFrame({
        "model": "Conditional flow",
        "target": target,
        "sample_index": np.arange(len(y)),
        "true_raw": y,
        "predictive_mean_raw": np.mean(samples, axis=1),
        "predictive_std_raw": np.std(samples, axis=1),
        "log_prob_normalized": log_prob,
        "nll_normalized": -log_prob,
        "crps_raw": crps,
    })
    for level in INTERVAL_LEVELS:
        key = int(round(level * 100))
        coverage = float(np.mean(intervals[f"covered_{key}"]) * 100.0)
        row[f"coverage_{key}_pct"] = coverage
        row[f"coverage_error_{key}_pp"] = abs(coverage - level * 100.0)
        row[f"mean_interval_width_{key}_raw"] = float(np.mean(intervals[f"width_{key}"]))
        for name in ["lower", "upper", "covered", "width"]:
            per_sample[f"{name}_{key}_raw" if name != "covered" else f"covered_{key}"] = intervals[f"{name}_{key}"]
    return row, per_sample


def summarize_gaussian_target(target: str, data: Dict[str, np.ndarray]):
    y = data["true_raw"]
    mu = data["mu_raw"]
    sigma = data["sigma_raw"]
    log_prob = data["log_prob"]
    crps = gaussian_crps(y, mu, sigma)
    intervals = interval_metrics_gaussian(y, mu, sigma, INTERVAL_LEVELS)

    row = dict(
        model="Gaussian baseline",
        target=target,
        n_test=len(y),
        nll_normalized=float(-np.mean(log_prob)),
        crps_raw=float(np.mean(crps)),
    )
    per_sample = pd.DataFrame({
        "model": "Gaussian baseline",
        "target": target,
        "sample_index": np.arange(len(y)),
        "true_raw": y,
        "predictive_mean_raw": mu,
        "predictive_std_raw": sigma,
        "log_prob_normalized": log_prob,
        "nll_normalized": -log_prob,
        "crps_raw": crps,
    })
    for level in INTERVAL_LEVELS:
        key = int(round(level * 100))
        coverage = float(np.mean(intervals[f"covered_{key}"]) * 100.0)
        row[f"coverage_{key}_pct"] = coverage
        row[f"coverage_error_{key}_pp"] = abs(coverage - level * 100.0)
        row[f"mean_interval_width_{key}_raw"] = float(np.mean(intervals[f"width_{key}"]))
        for name in ["lower", "upper", "covered", "width"]:
            per_sample[f"{name}_{key}_raw" if name != "covered" else f"covered_{key}"] = intervals[f"{name}_{key}"]
    return row, per_sample


def build_table14(metrics_df: pd.DataFrame) -> pd.DataFrame:
    table = metrics_df[[
        "model",
        "target",
        "nll_normalized",
        "crps_raw",
        "coverage_90_pct",
        "coverage_error_90_pp",
        "mean_interval_width_90_raw",
    ]].copy()
    table = table.rename(columns={
        "model": "Model",
        "target": "Target",
        "nll_normalized": "Negative log-likelihood (normalized target space)",
        "crps_raw": "CRPS (original target scale)",
        "coverage_90_pct": "Empirical 90% coverage (%)",
        "coverage_error_90_pp": "Absolute 90% coverage error (percentage points)",
        "mean_interval_width_90_raw": "Mean 90% prediction-interval width (original target scale)",
    })
    order = {
        ("Gaussian baseline", "SOC"): 0,
        ("Conditional flow", "SOC"): 1,
        ("Gaussian baseline", "SOH"): 2,
        ("Conditional flow", "SOH"): 3,
    }
    table["_order"] = [order.get((m, t), 999) for m, t in zip(table["Model"], table["Target"])]
    return table.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if not FLOW_CKPT.exists():
        raise FileNotFoundError(f"Flow checkpoint not found: {FLOW_CKPT}")
    if not GAUSSIAN_CKPT.exists():
        raise FileNotFoundError(f"Gaussian checkpoint not found: {GAUSSIAN_CKPT}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    dl_flow, dl_gaussian, info = build_shared_test_loaders()
    flow_model, flow_config = build_flow_model(info["num_classes"], device)
    gaussian_model, gaussian_config = build_gaussian_model(info["num_classes"], device)

    flow_result = evaluate_flow(
        flow_model, dl_flow, device, info["soc_norm"], info["soh_norm"]
    )
    gaussian_result = evaluate_gaussian(
        gaussian_model, dl_gaussian, device, info["soc_norm"], info["soh_norm"]
    )

    rows = []
    per_sample_frames = []
    for target in ["SOC", "SOH"]:
        row, frame = summarize_gaussian_target(target, gaussian_result[target])
        rows.append(row)
        per_sample_frames.append(frame)
        row, frame = summarize_flow_target(target, flow_result[target])
        rows.append(row)
        per_sample_frames.append(frame)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(
        SAVE_DIR / "probabilistic_metrics_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_sample_df = pd.concat(per_sample_frames, ignore_index=True)
    per_sample_df.to_csv(
        SAVE_DIR / "per_sample_probabilistic_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    table14_df = build_table14(metrics_df)
    table14_df.to_csv(
        SAVE_DIR / "supplementary_table14_probabilistic.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_json(
        SAVE_DIR / "run_config.json",
        {
            "flow_checkpoint": str(FLOW_CKPT),
            "gaussian_checkpoint": str(GAUSSIAN_CKPT),
            "split_path": info["split_path"],
            "n_test": info["n_test"],
            "num_classes": info["num_classes"],
            "flow_num_samples": FLOW_NUM_SAMPLES,
            "interval_levels": INTERVAL_LEVELS,
            "nll_space": "normalized target space",
            "crps_space": "original target scale",
            "soc_interval_width_unit": "SOC percentage points",
            "soh_interval_width_unit": "original SOH unit",
            "flow_run_config": flow_config,
            "gaussian_run_config": gaussian_config,
        },
    )

    print("\n[PROBABILISTIC METRICS]")
    print(metrics_df.to_string(index=False))
    print("\n[SUPPLEMENTARY TABLE 14]")
    print(table14_df.to_string(index=False))
    print(f"\n[OK] Saved outputs under: {SAVE_DIR}")


if __name__ == "__main__":
    main()
