# measurement_sensitivity/input_quality_sensitivity.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import sys
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from utils.cache import ensure_dir, load_or_build_cache, drop_nan_inf_rows
from utils.metrics import medape, mape, rmse, mae
from utils.seed import set_random_seed

from proposed_framework.run_proposed_framework import run_experiment
from proposed_framework.data.build_dataset import (
    build_train_mix_soc_mix_pt,
    build_test_random_mix_pt,
    pick_test_ids,
    apply_id_split,
)
from proposed_framework.data.feature_builder import build_three_channel_representation
from proposed_framework.models.hierarchical_model import Hier3HeadModel


# =============================================================================
# Corruption functions
# =============================================================================

def random_drop_and_interpolate(
    X_raw: np.ndarray,
    drop_count: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Randomly drop U features and interpolate them.

    Parameters
    ----------
    X_raw:
        Raw U1-U41 input, shape (N, 41), before z-score normalization.

    drop_count:
        Number of dropped points per sample. Dropping is applied only to U2-U41.
        U1 is preserved as the OCV/baseline reference.

    seed:
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Interpolated raw U1-U41 input, shape (N, 41).
    """
    if drop_count <= 0:
        return X_raw.copy()

    rng = np.random.RandomState(seed)
    X_corrupt = X_raw.copy()
    x_full = np.arange(X_raw.shape[1], dtype=float)

    for i in range(X_corrupt.shape[0]):
        row = X_corrupt[i].copy()

        drop_idx = rng.choice(
            np.arange(1, 41),
            size=min(int(drop_count), 40),
            replace=False,
        )

        row[drop_idx] = np.nan

        mask = ~np.isnan(row)
        x_keep = x_full[mask]
        y_keep = row[mask]

        try:
            if len(x_keep) >= 3:
                from scipy.interpolate import PchipInterpolator

                f = PchipInterpolator(
                    x_keep,
                    y_keep,
                    extrapolate=True,
                )
                row_interp = f(x_full)
            else:
                s = pd.Series(row)
                row_interp = (
                    s.interpolate(method="linear", limit_direction="both")
                    .to_numpy()
                )

        except Exception:
            s = pd.Series(row)
            s_interp = s.interpolate(method="linear", limit_direction="both")

            if s_interp.isna().any():
                s_interp = s_interp.bfill().ffill()

            row_interp = s_interp.to_numpy()

        X_corrupt[i] = row_interp

    return X_corrupt


def add_structured_noise(
    x3: np.ndarray,
    alpha: float,
    noise_scale: float,
    mode: str,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Add Gaussian noise to the structured 3-channel input.

    Parameters
    ----------
    x3:
        Structured input, shape (3, 5, 8).

    alpha:
        Noise level. For example, 0.003 means sigma = 0.003 * noise_scale.

    noise_scale:
        Global noise scale estimated from training data.

    mode:
        - "none": no noise
        - "ocv_preserved": perturb ch1 and ch2 only; keep ch3 clean
        - "all_perturbed": perturb all three channels, including ch3

    rng:
        NumPy random generator.

    Returns
    -------
    np.ndarray
        Noisy structured input.
    """
    if alpha <= 0 or mode == "none":
        return x3.astype(np.float32)

    sigma = float(alpha) * float(noise_scale)

    if sigma <= 0:
        return x3.astype(np.float32)

    x_noisy = x3.astype(np.float32).copy()

    if mode == "ocv_preserved":
        noise = rng.normal(
            loc=0.0,
            scale=sigma,
            size=x_noisy[:2].shape,
        ).astype(np.float32)

        x_noisy[:2] += noise

    elif mode == "all_perturbed":
        noise = rng.normal(
            loc=0.0,
            scale=sigma,
            size=x_noisy.shape,
        ).astype(np.float32)

        x_noisy += noise

    else:
        raise ValueError(
            "mode must be one of {'none', 'ocv_preserved', 'all_perturbed'}."
        )

    return x_noisy


# =============================================================================
# Evaluation dataset
# =============================================================================

class DegradedEvalDataset(Dataset):
    """
    Test-only dataset for input-quality sensitivity.

    It uses already normalized U1-U41 input and applies optional structured
    Gaussian noise after the 3-channel representation is built.
    """

    def __init__(
        self,
        X_u_norm: np.ndarray,
        y_cls: np.ndarray,
        meta: pd.DataFrame,
        soc_col: str,
        soh_col: str,
        pt_norm: Tuple[float, float],
        normalize_soc: bool,
        zscore_normalize: bool,
        soc_norm: Optional[Tuple[float, float]],
        soh_norm: Optional[Tuple[float, float]],
        noise_alpha: float = 0.0,
        noise_scale: float = 1.0,
        noise_mode: str = "none",
        seed: int = 42,
    ):
        self.X_u_norm = X_u_norm
        self.y_cls = y_cls.astype(np.int64)
        self.meta = meta.reset_index(drop=True)

        self.soc_col = soc_col
        self.soh_col = soh_col

        self.normalize_soc = bool(normalize_soc)
        self.zscore_normalize = bool(zscore_normalize)
        self.soc_norm = soc_norm
        self.soh_norm = soh_norm

        self.pt_mean = float(pt_norm[0])
        self.pt_std = float(pt_norm[1])

        self.noise_alpha = float(noise_alpha)
        self.noise_scale = float(noise_scale)
        self.noise_mode = str(noise_mode)

        self.rng = np.random.RandomState(seed)

        if soc_col not in self.meta.columns or soh_col not in self.meta.columns:
            raise RuntimeError(
                f"Meta must contain soc_col='{soc_col}' and soh_col='{soh_col}'."
            )

        soc = self.meta[soc_col].astype(float).to_numpy(dtype=np.float32)
        soh = self.meta[soh_col].astype(float).to_numpy(dtype=np.float32)

        if self.normalize_soc:
            soc = soc / 100.0

        if self.zscore_normalize:
            if self.soc_norm is None or self.soh_norm is None:
                raise RuntimeError(
                    "zscore_normalize=True requires soc_norm and soh_norm."
                )

            soc_mean, soc_std = float(self.soc_norm[0]), float(self.soc_norm[1])
            soh_mean, soh_std = float(self.soh_norm[0]), float(self.soh_norm[1])

            soc = (soc - soc_mean) / (soc_std + 1e-8)
            soh = (soh - soh_mean) / (soh_std + 1e-8)

        self.soc = soc
        self.soh = soh

        if "pulse_ms" in self.meta.columns:
            self.pt_ms = self.meta["pulse_ms"].astype(float).to_numpy(dtype=np.float32)
        else:
            self.pt_ms = np.zeros(len(self.meta), dtype=np.float32)

    def __len__(self) -> int:
        return int(self.X_u_norm.shape[0])

    def __getitem__(self, idx: int):
        x3 = build_three_channel_representation(self.X_u_norm[idx])

        x3 = add_structured_noise(
            x3=x3,
            alpha=self.noise_alpha,
            noise_scale=self.noise_scale,
            mode=self.noise_mode,
            rng=self.rng,
        )

        x3 = torch.from_numpy(x3)

        p = (np.log1p(float(self.pt_ms[idx])) - self.pt_mean) / self.pt_std
        pt = torch.tensor([p], dtype=torch.float32)

        y_cls = torch.tensor(int(self.y_cls[idx]), dtype=torch.long)
        soc = torch.tensor(float(self.soc[idx]), dtype=torch.float32)
        soh = torch.tensor(float(self.soh[idx]), dtype=torch.float32)

        return x3, pt, y_cls, soc, soh


# =============================================================================
# Utilities
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


def _torch_load(path: str | Path, map_location: str):
    """
    Compatibility wrapper for torch.load across PyTorch versions.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def find_best_checkpoint(clean_exp_dir: str | Path) -> Path:
    """
    Find the final-stage best checkpoint from a clean proposed-framework run.
    """
    clean_exp_dir = Path(clean_exp_dir)

    final_metrics_path = clean_exp_dir / "metrics" / "final_metrics.csv"

    candidate_stages = ["finetune", "stage2_soh", "stage1_soc", "single"]

    if final_metrics_path.exists():
        df = pd.read_csv(final_metrics_path)

        if len(df) > 0 and "final_stage" in df.columns:
            final_stage = str(df.iloc[0]["final_stage"])
            candidate_stages = [final_stage] + [
                s for s in candidate_stages if s != final_stage
            ]

    for stage in candidate_stages:
        path = clean_exp_dir / "checkpoints" / stage / "best.pt"

        if path.exists():
            return path

    raise FileNotFoundError(
        f"No best checkpoint found under: {clean_exp_dir / 'checkpoints'}"
    )


def build_clean_eval_context(
    data_root: str | Path,
    clean_exp_dir: str | Path,
    pulse_list: List[int],
    batch_size: int,
    seed: int,
    width: int,
    blocks: int,
    drop2d: float,
    head_dropout: float,
    soc_col: str,
    soh_col: str,
    normalize_soc: bool,
    zscore_normalize: bool,
    test_id_frac: float,
    test_id_count: int,
    u_start: int = 1,
    u_end: int = 41,
    drop_first_class: bool = True,
) -> Dict:
    """
    Rebuild the same train/test split, normalization statistics, label encoder
    and model object used by the clean proposed-framework experiment.
    """
    data_root = Path(data_root)
    clean_exp_dir = Path(clean_exp_dir)

    cache_dir = clean_exp_dir / "cache"
    ensure_dir(str(cache_dir))

    soc_list = list(range(5, 90, 5))

    train_kwargs = {
        "data_root": str(data_root),
        "soc_list": soc_list,
        "pulse_list": list(map(int, pulse_list)),
        "u_start": u_start,
        "u_end": u_end,
        "drop_first_class": drop_first_class,
    }

    Xtr_raw, ytr_raw, mtr_raw, _, _ = load_or_build_cache(
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

    Xte_raw, yte_raw, mte_raw, _, _ = load_or_build_cache(
        str(cache_dir),
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
        test_id_frac=test_id_frac,
        test_id_count=test_id_count,
        seed=seed,
    )

    Xtr, ytr_str, mtr, Xte_raw_original, yte_str, mte = apply_id_split(
        Xtr=Xtr_raw,
        ytr_str=ytr_raw,
        mtr=mtr_raw,
        Xte=Xte_raw,
        yte_str=yte_raw,
        mte=mte_raw,
        test_ids=test_ids,
    )

    u_mean = Xtr.mean(axis=0, keepdims=True)
    u_std = Xtr.std(axis=0, keepdims=True) + 1e-8

    Xte_clean_norm = (Xte_raw_original - u_mean) / u_std

    noise_global_scale = float(np.mean(np.std(Xtr[:, 1:], axis=0)))

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

    label_encoder = LabelEncoder()
    ytr_cls = label_encoder.fit_transform(ytr_str)

    train_classes = set(label_encoder.classes_.tolist())

    mask_known = np.array(
        [label in train_classes for label in yte_str],
        dtype=bool,
    )

    if not mask_known.all():
        Xte_raw_original = Xte_raw_original[mask_known]
        Xte_clean_norm = Xte_clean_norm[mask_known]
        yte_str = yte_str[mask_known]
        mte = mte.loc[mask_known].reset_index(drop=True)

    yte_cls = label_encoder.transform(yte_str)

    if "pulse_ms" in mtr.columns:
        pt_train_ms = mtr["pulse_ms"].astype(float).to_numpy(dtype=np.float32)
        pt_log = np.log1p(pt_train_ms)
        pt_norm = (
            float(pt_log.mean()),
            float(pt_log.std() + 1e-8),
        )
    else:
        pt_norm = (0.0, 1.0)

    model = Hier3HeadModel(
        num_classes=len(label_encoder.classes_),
        width=width,
        blocks=blocks,
        drop2d=drop2d,
        use_pt_as_feature=True,
        head_dropout=head_dropout,
    )

    ckpt_path = find_best_checkpoint(clean_exp_dir)
    ckpt = _torch_load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    return {
        "model": model,
        "Xte_raw_original": Xte_raw_original,
        "Xte_clean_norm": Xte_clean_norm,
        "yte_cls": yte_cls,
        "mte": mte,
        "u_mean": u_mean,
        "u_std": u_std,
        "pt_norm": pt_norm,
        "soc_norm": soc_norm,
        "soh_norm": soh_norm,
        "noise_global_scale": noise_global_scale,
        "checkpoint_path": str(ckpt_path),
        "num_classes": len(label_encoder.classes_),
        "batch_size": int(batch_size),
    }


@torch.no_grad()
def evaluate_degraded_input(
    model,
    Xte_norm: np.ndarray,
    yte_cls: np.ndarray,
    mte: pd.DataFrame,
    pt_norm: Tuple[float, float],
    soc_norm: Optional[Tuple[float, float]],
    soh_norm: Optional[Tuple[float, float]],
    normalize_soc: bool,
    zscore_normalize: bool,
    noise_alpha: float,
    noise_scale: float,
    noise_mode: str,
    batch_size: int,
    seed: int,
    device: str,
) -> Dict[str, float]:
    """
    Evaluate model on corrupted test input.
    """
    model.eval()
    model.to(device)

    ds = DegradedEvalDataset(
        X_u_norm=Xte_norm,
        y_cls=yte_cls,
        meta=mte,
        soc_col="SOC",
        soh_col="SOH",
        pt_norm=pt_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
        soc_norm=soc_norm if zscore_normalize else None,
        soh_norm=soh_norm if zscore_normalize else None,
        noise_alpha=noise_alpha,
        noise_scale=noise_scale,
        noise_mode=noise_mode,
        seed=seed,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    y_cls_true = []
    y_cls_pred = []

    soc_true = []
    soc_pred = []

    soh_true = []
    soh_pred = []

    for x3, pt, y_cls, soc, soh in dl:
        x3 = x3.to(device)
        pt = pt.to(device)

        logits, soc_p, _, _, soh_p, _ = model(
            x_img=x3,
            x_pt=pt,
            soc_tf=None,
            n_mc=32,
        )

        y_cls_true.append(y_cls.cpu().numpy())
        y_cls_pred.append(logits.argmax(1).detach().cpu().numpy())

        soc_true.append(soc.cpu().numpy())
        soc_pred.append(soc_p.detach().cpu().numpy())

        soh_true.append(soh.cpu().numpy())
        soh_pred.append(soh_p.detach().cpu().numpy())

    y_cls_true = np.concatenate(y_cls_true)
    y_cls_pred = np.concatenate(y_cls_pred)

    soc_true = np.concatenate(soc_true)
    soc_pred = np.concatenate(soc_pred)

    soh_true = np.concatenate(soh_true)
    soh_pred = np.concatenate(soh_pred)

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
        "cls_acc": float(accuracy_score(y_cls_true, y_cls_pred)),
        "soc_medape_raw": medape(soc_true_raw, soc_pred_raw),
        "soh_medape_raw": medape(soh_true_raw, soh_pred_raw),
        "soc_mape_raw": mape(soc_true_raw, soc_pred_raw),
        "soh_mape_raw": mape(soh_true_raw, soh_pred_raw),
        "soc_rmse_raw": rmse(soc_true_raw, soc_pred_raw),
        "soh_rmse_raw": rmse(soh_true_raw, soh_pred_raw),
        "soc_mae_raw": mae(soc_true_raw, soc_pred_raw),
        "soh_mae_raw": mae(soh_true_raw, soh_pred_raw),
        "n_test": int(len(y_cls_true)),
    }


# =============================================================================
# Main sensitivity runner
# =============================================================================

def run_input_quality_sensitivity(
    data_root: str | Path,
    output_root: str | Path,
    clean_exp_dir: str | Path,
    smoke: bool = False,
    train_clean_if_needed: bool = True,
    resume_clean: bool = True,
) -> pd.DataFrame:
    """
    Run drop/noise input-quality sensitivity analysis without modifying
    proposed_framework.

    Workflow:
    1. Train or load a clean proposed-framework model.
    2. Rebuild the same test split.
    3. Evaluate Gaussian-noise and drop-interpolation corruptions.
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    clean_exp_dir = Path(clean_exp_dir)

    ensure_dir(str(output_root), str(clean_exp_dir))

    set_random_seed(42)

    if smoke:
        pulse_list = [5000]

        clean_train_kwargs = {
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

        noise_levels = [0.0, 0.003]
        drop_levels = [0, 1]
        drop_repeats = 1

        width = 16
        blocks = 1
        head_dropout = 0.1
        batch_size = 32

    else:
        pulse_list = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

        clean_train_kwargs = {
            "batch_size": 128,
            "max_epochs": 400,
            "early_stopping": False,
            "patience": 20,
            "resume": resume_clean,
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

        noise_levels=[0, 0.001,0.003, 0.005, 0.01, 0.02, 0.05]
        drop_levels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
        drop_repeats = 5

        width = 32
        blocks = 4
        head_dropout = 0.2
        batch_size = 128

    checkpoint_exists = False

    try:
        _ = find_best_checkpoint(clean_exp_dir)
        checkpoint_exists = True
    except FileNotFoundError:
        checkpoint_exists = False

    if train_clean_if_needed and not checkpoint_exists:
        print("[CLEAN] No clean checkpoint found. Running clean proposed-framework training.")

        run_experiment(
            data_root=str(data_root),
            pulse_list=pulse_list,
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
            exp_dir=clean_exp_dir,
            **clean_train_kwargs,
        )

    context = build_clean_eval_context(
        data_root=data_root,
        clean_exp_dir=clean_exp_dir,
        pulse_list=pulse_list,
        batch_size=batch_size,
        seed=42,
        width=width,
        blocks=blocks,
        drop2d=0.0,
        head_dropout=head_dropout,
        soc_col="SOC",
        soh_col="SOH",
        normalize_soc=True,
        zscore_normalize=True,
        test_id_frac=0.2,
        test_id_count=0,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Evaluation device: {device}")
    print(f"[INFO] Loaded checkpoint: {context['checkpoint_path']}")
    print(f"[INFO] Noise global scale: {context['noise_global_scale']:.8f}")

    rows = []

    # -------------------------------------------------------------------------
    # A. Gaussian noise on structured input
    # -------------------------------------------------------------------------
    for alpha in noise_levels:
        for noise_mode in ["ocv_preserved", "all_perturbed"]:
            if alpha == 0.0 and noise_mode == "all_perturbed":
                continue

            print("\n" + "=" * 90)
            print(f"[RUN] Noise sensitivity | alpha={alpha} | mode={noise_mode}")
            print("=" * 90)

            out = evaluate_degraded_input(
                model=context["model"],
                Xte_norm=context["Xte_clean_norm"],
                yte_cls=context["yte_cls"],
                mte=context["mte"],
                pt_norm=context["pt_norm"],
                soc_norm=context["soc_norm"],
                soh_norm=context["soh_norm"],
                normalize_soc=True,
                zscore_normalize=True,
                noise_alpha=float(alpha),
                noise_scale=float(context["noise_global_scale"]),
                noise_mode=noise_mode if alpha > 0 else "none",
                batch_size=batch_size,
                seed=42 + int(alpha * 100000),
                device=device,
            )

            row = {
                "type": "noise",
                "level": float(alpha),
                "noise_mode": noise_mode if alpha > 0 else "none",
                "drop_count": 0,
                "repeat_id": 0,
                **out,
            }

            rows.append(row)

            pd.DataFrame(rows).to_csv(
                output_root / "input_quality_sensitivity_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

            print(row)

    # -------------------------------------------------------------------------
    # B. Random drop + interpolation on raw U1-U41
    # -------------------------------------------------------------------------
    for drop_count in drop_levels:
        repeat_range = range(1) if drop_count == 0 else range(drop_repeats)

        for repeat_id in repeat_range:
            print("\n" + "=" * 90)
            print(
                f"[RUN] Drop sensitivity | drop_count={drop_count} | repeat={repeat_id}"
            )
            print("=" * 90)

            if drop_count == 0:
                X_drop_norm = context["Xte_clean_norm"]
            else:
                X_drop_raw = random_drop_and_interpolate(
                    X_raw=context["Xte_raw_original"],
                    drop_count=int(drop_count),
                    seed=42 + 1000 * int(drop_count) + int(repeat_id),
                )

                X_drop_norm = (X_drop_raw - context["u_mean"]) / context["u_std"]

            out = evaluate_degraded_input(
                model=context["model"],
                Xte_norm=X_drop_norm,
                yte_cls=context["yte_cls"],
                mte=context["mte"],
                pt_norm=context["pt_norm"],
                soc_norm=context["soc_norm"],
                soh_norm=context["soh_norm"],
                normalize_soc=True,
                zscore_normalize=True,
                noise_alpha=0.0,
                noise_scale=float(context["noise_global_scale"]),
                noise_mode="none",
                batch_size=batch_size,
                seed=42 + repeat_id,
                device=device,
            )

            row = {
                "type": "drop",
                "level": int(drop_count),
                "noise_mode": "none",
                "drop_count": int(drop_count),
                "repeat_id": int(repeat_id),
                **out,
            }

            rows.append(row)

            pd.DataFrame(rows).to_csv(
                output_root / "input_quality_sensitivity_partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

            print(row)

    summary = pd.DataFrame(rows)

    summary.to_csv(
        output_root / "input_quality_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Aggregated drop summary
    drop_summary = (
        summary[summary["type"] == "drop"]
        .groupby("drop_count", as_index=False)
        .agg(
            cls_acc_mean=("cls_acc", "mean"),
            cls_acc_std=("cls_acc", "std"),
            soc_medape_raw_mean=("soc_medape_raw", "mean"),
            soc_medape_raw_std=("soc_medape_raw", "std"),
            soh_medape_raw_mean=("soh_medape_raw", "mean"),
            soh_medape_raw_std=("soh_medape_raw", "std"),
            n_test=("n_test", "first"),
        )
    )

    drop_summary = drop_summary.fillna(0.0)

    drop_summary.to_csv(
        output_root / "drop_sensitivity_aggregated.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Noise summary
    noise_summary = summary[summary["type"] == "noise"].copy()

    noise_summary.to_csv(
        output_root / "noise_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        output_root / "input_quality_sensitivity_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    data_root = PROJECT_ROOT / "data"

    output_root = (
        PROJECT_ROOT
        / "results"
        / "measurement_sensitivity"
        / "input_quality"
    )

    clean_exp_dir = (
        PROJECT_ROOT
        / "results"
        / "measurement_sensitivity"
        / "input_quality"
        / "clean_model"
    )

    summary = run_input_quality_sensitivity(
        data_root=data_root,
        output_root=output_root,
        clean_exp_dir=clean_exp_dir,
        smoke=False,
        train_clean_if_needed=True,
        resume_clean=True,
    )

    print("\n[SUMMARY]")
    print(summary)


if __name__ == "__main__":
    main()