# -*- coding: utf-8 -*-
"""
plot_fig5c_calibration.py

Figure 5c:
Calibration deviation curve comparing the conditional-flow model and
the Gaussian calibration baseline.

This script follows the old Figure 5c logic:
    1. Rebuild the test set.
    2. Load the proposed conditional-flow checkpoint.
    3. Load the Gaussian calibration-baseline checkpoint.
    4. Run inference.
    5. Compute calibrated probability residuals.
    6. Plot observed-minus-expected calibration deviation.

Inputs:
    results/proposed_framework/checkpoints/finetune/best.pt
    results/calibration_baseline/checkpoints/best.pt

Output:
    results/figures/main/fig5c/fig5c_calibration.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder


# =============================================================================
# Project path
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Project imports
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

from analysis.train_calibration_baseline import (
    GaussianCalibrationBaseline,
    CalibrationBaselineDataset,
)


# =============================================================================
# Paths
# =============================================================================
DATA_ROOT = PROJECT_ROOT / "data"

PROPOSED_DIR = PROJECT_ROOT / "results" / "proposed_framework"
BASELINE_DIR = PROJECT_ROOT / "results" / "calibration_baseline"

PROPOSED_CKPT = PROPOSED_DIR / "checkpoints" / "finetune" / "best.pt"
BASELINE_CKPT = BASELINE_DIR / "checkpoints" / "best.pt"

SAVE_DIR = PROJECT_ROOT / "results" / "figures" / "main" / "fig5c"
SAVE_NAME = "fig5c_calibration.png"

os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# Configuration
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

WIDTH = 32
BLOCKS = 4
DROP2D = 0.0
HEAD_DROPOUT = 0.2

SOC_HIDDEN = 64
SOH_HIDDEN = 64

FLOW_LAYERS = 6
FLOW_BINS = 8
FLOW_TAIL_BOUND = 3.0

Y_LIMIT = 0.15


# =============================================================================
# Style
# =============================================================================
STYLE = {
    "flow_color": "#0072B2",
    "gauss_color": "#D55E00",
    "inner_band": "#CCCCCC",
    "outer_band": "#E6E6E6",
    "line_width": 2.2,
    "dpi": 600,
}


# =============================================================================
# Utilities
# =============================================================================
def torch_load_compatible(path: str | Path, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(ckpt) -> dict:
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        return ckpt
    raise RuntimeError("Unsupported checkpoint format.")


def inverse_targets_np(
    soc_z: np.ndarray,
    soh_z: np.ndarray,
    soc_norm: Optional[Tuple[float, float]],
    soh_norm: Optional[Tuple[float, float]],
    normalize_soc: bool,
    zscore_normalize: bool,
):
    soc = np.asarray(soc_z, dtype=np.float64).copy()
    soh = np.asarray(soh_z, dtype=np.float64).copy()

    if zscore_normalize:
        if soc_norm is None or soh_norm is None:
            raise RuntimeError("soc_norm and soh_norm are required.")
        soc = soc * float(soc_norm[1]) + float(soc_norm[0])
        soh = soh * float(soh_norm[1]) + float(soh_norm[0])

    if normalize_soc:
        soc = soc * 100.0

    return soc, soh


def first_linear_in_features(module: torch.nn.Module) -> Optional[int]:
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            return int(m.in_features)
    return None


def model_uses_pt(model: torch.nn.Module) -> bool:
    if hasattr(model, "use_pt"):
        return bool(getattr(model, "use_pt"))
    if hasattr(model, "use_pt_as_feature"):
        return bool(getattr(model, "use_pt_as_feature"))
    return False


def build_flow_model(num_classes: int, device: str) -> torch.nn.Module:
    try:
        model = Hier3HeadModel(
            num_classes=int(num_classes),
            width=WIDTH,
            blocks=BLOCKS,
            drop2d=DROP2D,
            use_pt_as_feature=USE_PT_AS_FEATURE,
            head_dropout=HEAD_DROPOUT,
            flow_layers=FLOW_LAYERS,
            flow_bins=FLOW_BINS,
            flow_tail_bound=FLOW_TAIL_BOUND,
        ).to(device)
    except TypeError:
        model = Hier3HeadModel(
            num_classes=int(num_classes),
            width=WIDTH,
            blocks=BLOCKS,
            drop2d=DROP2D,
            use_pt_as_feature=USE_PT_AS_FEATURE,
            head_dropout=HEAD_DROPOUT,
        ).to(device)

    return model


def load_flow_model(num_classes: int, device: str) -> torch.nn.Module:
    if not PROPOSED_CKPT.exists():
        raise FileNotFoundError(
            f"Proposed-framework checkpoint not found: {PROPOSED_CKPT}"
        )

    model = build_flow_model(num_classes=num_classes, device=device)
    ckpt = torch_load_compatible(PROPOSED_CKPT, map_location=device)
    state = extract_state_dict(ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"[MODEL] Loaded flow checkpoint: {PROPOSED_CKPT}")
    return model


def load_gaussian_baseline(num_classes: int, device: str) -> torch.nn.Module:
    if not BASELINE_CKPT.exists():
        raise FileNotFoundError(
            f"Calibration-baseline checkpoint not found: {BASELINE_CKPT}\n"
            "Please run analysis/train_calibration_baseline.py first."
        )

    model = GaussianCalibrationBaseline(
        num_classes=int(num_classes),
        width=WIDTH,
        blocks=BLOCKS,
        drop2d=DROP2D,
        use_pt_as_feature=USE_PT_AS_FEATURE,
        soc_hidden=SOC_HIDDEN,
        soh_hidden=SOH_HIDDEN,
        head_dropout=HEAD_DROPOUT,
    ).to(device)

    ckpt = torch_load_compatible(BASELINE_CKPT, map_location=device)
    state = extract_state_dict(ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"[MODEL] Loaded Gaussian baseline checkpoint: {BASELINE_CKPT}")
    return model


# =============================================================================
# Data
# =============================================================================
def build_test_loaders():
    cache_dir = PROPOSED_DIR / "cache"
    split_dir = PROPOSED_DIR / "splits"

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    soc_list = list(range(5, 90, 5))

    train_kwargs = {
        "data_root": str(DATA_ROOT),
        "soc_list": soc_list,
        "pulse_list": list(map(int, PULSE_LIST)),
        "u_start": U_START,
        "u_end": U_END,
        "drop_first_class": DROP_FIRST_CLASS,
    }

    Xtr_raw, ytr_raw, mtr_raw, tag_tr, hit_tr = load_or_build_cache(
        str(cache_dir),
        "raw_train",
        build_train_mix_soc_mix_pt,
        train_kwargs,
    )

    test_kwargs = {
        "data_root": str(DATA_ROOT),
        "pulse_list": list(map(int, PULSE_LIST)),
        "u_start": U_START,
        "u_end": U_END,
        "drop_first_class": DROP_FIRST_CLASS,
    }

    Xte_raw, yte_raw, mte_raw, tag_te, hit_te = load_or_build_cache(
        str(cache_dir),
        "raw_test",
        build_test_random_mix_pt,
        test_kwargs,
    )

    print(f"[CACHE] Train tag: {tag_tr} | hit={hit_tr}")
    print(f"[CACHE] Test  tag: {tag_te} | hit={hit_te}")

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

    split_name = (
        f"testIDs_seed{SEED}_n{TEST_ID_COUNT}"
        if TEST_ID_COUNT and TEST_ID_COUNT > 0
        else f"testIDs_seed{SEED}_frac{TEST_ID_FRAC}"
    )
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

    if len(ytr_str) == 0 or len(yte_str) == 0:
        raise RuntimeError("Empty train or test set after ID split.")

    label_encoder = LabelEncoder()
    ytr_cls = label_encoder.fit_transform(ytr_str)

    train_classes = set(label_encoder.classes_.tolist())
    mask_known = np.array([lbl in train_classes for lbl in yte_str], dtype=bool)

    if not mask_known.all():
        n_removed = int((~mask_known).sum())
        print(f"[WARN] Removing {n_removed} test samples with unseen labels.")
        Xte = Xte[mask_known]
        yte_str = yte_str[mask_known]
        mte = mte.loc[mask_known].reset_index(drop=True)

    yte_cls = label_encoder.transform(yte_str)
    num_classes = len(label_encoder.classes_)

    soc_train = mtr[SOC_COL].astype(float).to_numpy(dtype=np.float64)
    if NORMALIZE_SOC:
        soc_train = soc_train / 100.0

    soc_norm = (
        float(soc_train.mean()),
        float(soc_train.std() + 1e-8),
    )

    soh_train = mtr[SOH_COL].astype(float).to_numpy(dtype=np.float64)
    soh_norm = (
        float(soh_train.mean()),
        float(soh_train.std() + 1e-8),
    )

    if USE_PT_AS_FEATURE and "pulse_ms" in mtr.columns:
        pt_train = np.log1p(mtr["pulse_ms"].astype(float).to_numpy(dtype=np.float64))
        pt_norm = (
            float(pt_train.mean()),
            float(pt_train.std() + 1e-8),
        )
    elif USE_PT_AS_FEATURE and "pulse_width_ms" in mtr.columns:
        pt_train = np.log1p(
            mtr["pulse_width_ms"].astype(float).to_numpy(dtype=np.float64)
        )
        pt_norm = (
            float(pt_train.mean()),
            float(pt_train.std() + 1e-8),
        )
    else:
        pt_norm = (0.0, 1.0)

    pt_col = "pulse_ms" if "pulse_ms" in mte.columns else "pulse_width_ms"

    # Proposed flow uses train-only U normalization.
    norm_path = PROPOSED_DIR / "u41_norm_train_only.npz"
    if norm_path.exists():
        norm_obj = np.load(norm_path)
        u_mean = norm_obj["u_mean"]
        u_std = norm_obj["u_std"]
        print(f"[NORM] Loaded U normalization: {norm_path}")
    else:
        u_mean = Xtr.mean(axis=0, keepdims=True)
        u_std = Xtr.std(axis=0, keepdims=True) + 1e-8
        print("[NORM] Computed U normalization from train split.")

    Xte_flow = (Xte - u_mean) / u_std

    # Gaussian baseline follows train_calibration_baseline.py:
    # NORMALIZE_U_WITH_TRAIN_STATS = False by default.
    Xte_gauss = Xte.copy()

    ds_flow = HierPulseDataset(
        X_u=Xte_flow,
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

    ds_gauss = CalibrationBaselineDataset(
        X_u=Xte_gauss,
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

    dl_flow = DataLoader(
        ds_flow,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    dl_gauss = DataLoader(
        ds_gauss,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    info = {
        "num_classes": int(num_classes),
        "soc_norm": soc_norm,
        "soh_norm": soh_norm,
        "pt_norm": pt_norm,
        "n_test": int(len(ds_flow)),
    }

    print(f"[DATA] Test samples: {info['n_test']}")
    print(f"[DATA] Num classes: {info['num_classes']}")

    return dl_flow, dl_gauss, info


# =============================================================================
# Calibration computations
# =============================================================================
@torch.no_grad()
def compute_flow_soc_noise(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> np.ndarray:
    model.eval()
    noise_list = []

    for x3, pt, _, soc, _ in loader:
        x3 = x3.to(device)
        pt = pt.to(device)
        soc = soc.to(device).view(-1, 1)

        z = model.encoder(x3)

        use_pt = model_uses_pt(model)
        in_features = first_linear_in_features(model.head_mat)
        z_dim = int(z.shape[1])
        pt_dim = int(pt.shape[1])

        if use_pt and in_features == z_dim + pt_dim:
            logits = model.head_mat(torch.cat([z, pt], dim=1))
        else:
            logits = model.head_mat(z)

        p = torch.softmax(logits, dim=1)

        if use_pt:
            cond_soc = torch.cat([z, p, pt], dim=1)
        else:
            cond_soc = torch.cat([z, p], dim=1)

        if hasattr(model.soc_flow, "flow"):
            noise, _ = model.soc_flow.flow._transform(soc, context=cond_soc)
        else:
            noise, _ = model.soc_flow._transform(soc, context=cond_soc)

        noise_list.append(noise.detach().cpu().numpy())

    return np.concatenate(noise_list, axis=0).reshape(-1)


@torch.no_grad()
def compute_gaussian_soc_z(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> np.ndarray:
    model.eval()
    z_list = []

    for x3, pt, _, soc, _ in loader:
        x3 = x3.to(device)
        pt = pt.to(device)
        soc = soc.to(device).view(-1)

        _, soc_mu, _, soc_sigma, _ = model(x3, pt)

        soc_sigma = torch.clamp(soc_sigma, min=1e-6)
        z = (soc - soc_mu) / soc_sigma

        z_list.append(z.detach().cpu().numpy())

    return np.concatenate(z_list, axis=0).reshape(-1)


def search_best_temperature(noise_array: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    best_t = 1.0
    best_ece = float("inf")
    best_u = None

    expected_p = np.linspace(0, 1, 100)
    core_mask = (expected_p >= 0.10) & (expected_p <= 0.90)

    noise_array = np.asarray(noise_array, dtype=np.float64)
    noise_array = noise_array[np.isfinite(noise_array)]

    normal = torch.distributions.Normal(0, 1)

    for t in np.linspace(0.3, 3.0, 300):
        u = normal.cdf(torch.tensor(noise_array / t, dtype=torch.float32)).numpy()
        u = u[np.isfinite(u)]

        observed_p = np.array([np.mean(u <= p) for p in expected_p])
        ece = np.mean(np.abs(observed_p[core_mask] - expected_p[core_mask]))

        if ece < best_ece:
            best_ece = float(ece)
            best_t = float(t)
            best_u = u

    print(f"[TEMP] {name}: best T = {best_t:.3f}, core ECE = {best_ece:.4f}")
    return best_u, best_t


# =============================================================================
# Plot
# =============================================================================
def plot_calibration_deviation(results: dict, save_path: str | Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    x_range = np.linspace(0, 1, 300)

    ax.fill_between(
        x_range,
        -0.10,
        0.10,
        color=STYLE["outer_band"],
        label="±10% tolerance",
        zorder=1,
    )
    ax.fill_between(
        x_range,
        -0.05,
        0.05,
        color=STYLE["inner_band"],
        label="±5% tolerance",
        zorder=2,
    )

    ax.axhline(
        0,
        color="black",
        lw=1.2,
        linestyle="--",
        zorder=3,
    )

    for name, item in results.items():
        u = np.asarray(item["u"], dtype=np.float64)
        u = u[np.isfinite(u)]

        observed_p = np.array([np.mean(u <= p) for p in x_range])
        deviation = observed_p - x_range

        ax.plot(
            x_range,
            deviation,
            label=name,
            color=item["color"],
            lw=STYLE["line_width"],
            zorder=4,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-Y_LIMIT, Y_LIMIT)

    ax.set_xlabel("Expected confidence level", fontsize=11, fontweight="bold")
    ax.set_ylabel("Calibration error (Obs-Exp)", fontsize=11, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_yticks([-0.10, -0.05, 0, 0.05, 0.10])
    ax.set_yticklabels(["-10%", "-5%", "0", "+5%", "+10%"])

    ax.legend(
        loc="upper left",
        frameon=True,
        edgecolor="black",
        fancybox=False,
        fontsize=9,
    )

    ax.grid(
        True,
        axis="y",
        color="#F0F0F0",
        lw=0.8,
        zorder=0,
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=STYLE["dpi"],
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    dl_flow, dl_gauss, info = build_test_loaders()

    flow_model = load_flow_model(
        num_classes=info["num_classes"],
        device=device,
    )

    gauss_model = load_gaussian_baseline(
        num_classes=info["num_classes"],
        device=device,
    )

    print("[INFER] Computing conditional-flow SOC calibration residuals...")
    flow_noise = compute_flow_soc_noise(
        model=flow_model,
        loader=dl_flow,
        device=device,
    )

    print("[INFER] Computing Gaussian-baseline SOC calibration residuals...")
    gauss_z = compute_gaussian_soc_z(
        model=gauss_model,
        loader=dl_gauss,
        device=device,
    )

    u_flow, t_flow = search_best_temperature(
        flow_noise,
        "Conditional flow",
    )

    u_gauss, t_gauss = search_best_temperature(
        gauss_z,
        "Gaussian baseline",
    )

    results = {
        f"Conditional flow (T={t_flow:.3f})": {
            "u": u_flow,
            "color": STYLE["flow_color"],
        },
        f"Gaussian baseline (T={t_gauss:.3f})": {
            "u": u_gauss,
            "color": STYLE["gauss_color"],
        },
    }

    save_path = SAVE_DIR / SAVE_NAME
    plot_calibration_deviation(
        results=results,
        save_path=save_path,
    )

    print(f"[OK] Saved: {save_path}")


if __name__ == "__main__":
    main()