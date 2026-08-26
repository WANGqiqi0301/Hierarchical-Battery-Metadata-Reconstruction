# ablation/material_conditioning_ablation.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import sys
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import proposed_framework.run_proposed_framework as M

from utils.cache import ensure_dir, load_or_build_cache, drop_nan_inf_rows
from utils.metrics import rmse, mae, mape, medape
from utils.seed import set_random_seed
from proposed_framework.data.build_dataset import (
    build_train_mix_soc_mix_pt,
    build_test_random_mix_pt,
    pick_test_ids,
)
from proposed_framework.data.pulse_dataset import HierPulseDataset


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

DEFAULT_PROPOSED_SUMMARY = Path(
    "/scratch/lt28/qq002/Hierachical_battery_metadata_reconstruction/"
    "results/proposed_framework/further_analysis/tables/proposed_method_summary.csv"
)


OriginalHier3HeadModel = M.Hier3HeadModel


# =============================================================================
# Hard / soft material-conditioning model wrapper
# =============================================================================

class MaterialConditioningHier3HeadModel(OriginalHier3HeadModel):
    """
    Ablation-only wrapper around the public proposed model.

    soft:
        Downstream SOC/SOH heads receive the full material softmax vector.

    hard:
        Downstream SOC/SOH heads receive a one-hot argmax material vector.
    """

    def __init__(
        self,
        *args,
        material_condition_mode: str = "soft",
        default_n_mc: int = 16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.material_condition_mode = str(material_condition_mode).lower()
        if self.material_condition_mode not in {"soft", "hard"}:
            raise ValueError("material_condition_mode must be 'soft' or 'hard'.")

        self.default_n_mc = int(default_n_mc)

    @staticmethod
    def _sample_mean_1d(samples: torch.Tensor, batch_size: int, n_mc: int, name: str):
        if samples.ndim == 3:
            if samples.shape[0] == int(n_mc) and samples.shape[1] == batch_size:
                return samples.mean(dim=0).squeeze(-1)

            if samples.shape[0] == batch_size and samples.shape[1] == int(n_mc):
                return samples.mean(dim=1).squeeze(-1)

            samples = samples.reshape(int(n_mc), batch_size, 1)
            return samples.mean(dim=0).squeeze(-1)

        if samples.ndim == 2:
            samples = samples.view(int(n_mc), batch_size, 1)
            return samples.mean(dim=0).squeeze(-1)

        raise RuntimeError(f"Unexpected {name} sample shape: {tuple(samples.shape)}")

    def _material_logits(self, z: torch.Tensor, x_pt: torch.Tensor) -> torch.Tensor:
        """
        Compatible with both public-model styles:
        1. material head uses z only;
        2. material head uses concat([z, pt]).
        """
        first_linear = None

        if hasattr(self.head_mat, "__iter__"):
            for layer in self.head_mat:
                if isinstance(layer, torch.nn.Linear):
                    first_linear = layer
                    break

        if first_linear is not None:
            in_features = int(first_linear.in_features)

            if in_features == z.shape[1]:
                return self.head_mat(z)

            if in_features == z.shape[1] + x_pt.shape[1]:
                return self.head_mat(torch.cat([z, x_pt], dim=1))

        return self.head_mat(z)

    def forward(
        self,
        x_img: torch.Tensor,
        x_pt: torch.Tensor,
        soc_tf: Optional[torch.Tensor] = None,
        n_mc: Optional[int] = None,
    ):
        if n_mc is None:
            n_mc = int(self.default_n_mc)
        else:
            n_mc = int(n_mc)

        z = self.encoder(x_img)
        batch_size = z.size(0)

        logits_mat = self._material_logits(z, x_pt)
        p_soft = torch.softmax(logits_mat, dim=1)

        if self.material_condition_mode == "soft":
            p_mat = p_soft
        elif self.material_condition_mode == "hard":
            hard_idx = torch.argmax(p_soft, dim=1)
            p_mat = torch.zeros_like(p_soft).scatter_(1, hard_idx.unsqueeze(1), 1.0)
        else:
            raise RuntimeError(
                f"Unknown material_condition_mode={self.material_condition_mode}"
            )

        if self.use_pt:
            cond_soc = torch.cat([z, p_mat, x_pt], dim=1)
        else:
            cond_soc = torch.cat([z, p_mat], dim=1)

        soc_logp = None

        if soc_tf is not None:
            soc_tf = soc_tf.view(-1)
            soc_logp = self.soc_flow.log_prob(soc_tf, cond_soc)

        with torch.no_grad():
            soc_samples = self.soc_flow.sample(cond_soc, num_samples=n_mc)
            soc_pred = self._sample_mean_1d(
                samples=soc_samples,
                batch_size=batch_size,
                n_mc=n_mc,
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
            soh_samples = self.soh_flow.sample(cond_soh, num_samples=n_mc)
            soh_pred = self._sample_mean_1d(
                samples=soh_samples,
                batch_size=batch_size,
                n_mc=n_mc,
                name="SOH",
            )

        soh_pred = soh_pred.view(-1)

        return logits_mat, soc_pred, soc_logp, cond_soc, soh_pred, cond_soh


# =============================================================================
# Small utilities
# =============================================================================

def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _torch_load(path: str | Path, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)



def _medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.median(np.abs(y_pred - y_true)))


def _split_material(label) -> str:
    return str(label).split("_")[0]


def _material_accuracy_from_labels(true_labels, pred_labels) -> float:
    true_material = np.asarray([_split_material(x) for x in true_labels], dtype=object)
    pred_material = np.asarray([_split_material(x) for x in pred_labels], dtype=object)
    if len(true_material) == 0:
        return float("nan")
    return float(np.mean(true_material == pred_material))


def _read_metric_from_row(row: pd.Series, names: List[str], default=np.nan) -> float:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            value = float(row[name])
            if name.endswith("_pct") and "acc" in name:
                value = value / 100.0
            return value
    return float(default)


def _scale_soh_error_to_percent(value: float) -> float:
    value = float(value)
    if np.isfinite(value) and abs(value) <= 2.0:
        return value * 100.0
    return value


def _maybe_soh_array_to_percent(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    if np.nanmedian(np.abs(values)) <= 2.0:
        return values * 100.0
    return values


def _find_prediction_file_from_summary(proposed_summary: Path) -> Optional[Path]:
    proposed_summary = Path(proposed_summary)
    search_roots = []

    for root in [
        proposed_summary.parent,
        proposed_summary.parent.parent,
        PROJECT_ROOT / "results" / "proposed_framework" / "further_analysis",
    ]:
        if root.exists() and root not in search_roots:
            search_roots.append(root)

    names = [
        "test_predictions_per_sample.csv",
        "predictions_per_sample.csv",
        "proposed_predictions_per_sample.csv",
        "proposed_method_predictions_per_sample.csv",
    ]

    for root in search_roots:
        for name in names:
            path = root / name
            if path.exists():
                return path

    for root in search_roots:
        matches = sorted(root.rglob("*prediction*sample*.csv"))
        if matches:
            return matches[0]

    return None


def _metrics_from_prediction_df(df: pd.DataFrame) -> Dict[str, float]:
    required = {"true_label", "pred_label", "soc_true", "soc_pred", "soh_true", "soh_pred"}
    if not required.issubset(df.columns):
        return {}

    true_labels = df["true_label"].astype(str).to_numpy()
    pred_labels = df["pred_label"].astype(str).to_numpy()

    soc_true = df["soc_true"].astype(float).to_numpy(dtype=np.float64)
    soc_pred = df["soc_pred"].astype(float).to_numpy(dtype=np.float64)

    soh_true = _maybe_soh_array_to_percent(
        df["soh_true"].astype(float).to_numpy(dtype=np.float64)
    )
    soh_pred = _maybe_soh_array_to_percent(
        df["soh_pred"].astype(float).to_numpy(dtype=np.float64)
    )

    return {
        "test_cls_acc": float(np.mean(true_labels == pred_labels)),
        "test_material_acc": _material_accuracy_from_labels(true_labels, pred_labels),
        "test_soc_medae_raw": _medae(soc_true, soc_pred),
        "test_soh_medae_raw": _medae(soh_true, soh_pred),
    }

def _inverse_targets(
    soc_z: np.ndarray,
    soh_z: np.ndarray,
    soc_norm: Optional[Tuple[float, float]],
    soh_norm: Optional[Tuple[float, float]],
    normalize_soc: bool,
    zscore_normalize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    soc = np.asarray(soc_z, dtype=np.float64)
    soh = np.asarray(soh_z, dtype=np.float64)

    if zscore_normalize:
        if soc_norm is None or soh_norm is None:
            raise RuntimeError("soc_norm/soh_norm are required for inverse transform.")
        soc = soc * float(soc_norm[1]) + float(soc_norm[0])
        soh = soh * float(soh_norm[1]) + float(soh_norm[0])

    if normalize_soc:
        soc = soc * 100.0

    # Report SOH on the same percentage scale as SOC.
    # Example: 0.95 becomes 95.0, so SOH MAE/MedAE/RMSE are percentage points.
    soh = soh * 100.0

    return soc, soh


def _add_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    summary = summary.copy()

    if "test_material_acc" not in summary.columns:
        summary["test_material_acc"] = np.nan

    summary["cls_acc_pct"] = summary["test_cls_acc"].astype(float) * 100.0
    summary["material_acc_pct"] = summary["test_material_acc"].astype(float) * 100.0
    summary["mat_acc_pct"] = summary["material_acc_pct"]

    summary["soc_medae"] = summary["test_soc_medae_raw"].astype(float)
    summary["soh_medae"] = summary["test_soh_medae_raw"].astype(float)
    summary["soc_medape_pct"] = summary["test_soc_medape_raw"].astype(float)
    summary["soh_medape_pct"] = summary["test_soh_medape_raw"].astype(float)

    if "soft" in set(summary["material_condition_mode"]):
        ref = summary.loc[summary["material_condition_mode"] == "soft"].iloc[0]

        ref_cls = float(ref["test_cls_acc"])
        ref_mat = float(ref["test_material_acc"])
        ref_soc_medae = float(ref["test_soc_medae_raw"])
        ref_soh_medae = float(ref["test_soh_medae_raw"])
        ref_soc_medape = float(ref["test_soc_medape_raw"])
        ref_soh_medape = float(ref["test_soh_medape_raw"])

        summary["cls_acc_change_pp_vs_soft"] = (
            summary["test_cls_acc"].astype(float) - ref_cls
        ) * 100.0
        summary["material_acc_change_pp_vs_soft"] = (
            summary["test_material_acc"].astype(float) - ref_mat
        ) * 100.0
        summary["soc_medae_change_vs_soft"] = (
            summary["test_soc_medae_raw"].astype(float) - ref_soc_medae
        )
        summary["soh_medae_change_vs_soft"] = (
            summary["test_soh_medae_raw"].astype(float) - ref_soh_medae
        )
        summary["soc_medape_change_pp_vs_soft"] = (
            summary["test_soc_medape_raw"].astype(float) - ref_soc_medape
        )
        summary["soh_medape_change_pp_vs_soft"] = (
            summary["test_soh_medape_raw"].astype(float) - ref_soh_medape
        )

    return summary


# =============================================================================
# Read proposed further-analysis summary as soft result
# =============================================================================

def read_soft_from_proposed_further_summary(
    proposed_summary: str | Path = DEFAULT_PROPOSED_SUMMARY,
) -> dict:
    proposed_summary = Path(proposed_summary)

    if not proposed_summary.exists():
        raise FileNotFoundError(
            f"Proposed further-analysis summary not found: {proposed_summary}"
        )

    df = pd.read_csv(proposed_summary)
    if "split" not in df.columns:
        raise RuntimeError(
            f"Expected a 'split' column in proposed summary: {proposed_summary}"
        )

    test_df = df.loc[df["split"].astype(str).str.lower() == "test"].copy()
    if test_df.empty:
        raise RuntimeError(
            f"No split == 'test' row found in proposed summary: {proposed_summary}"
        )

    row = test_df.iloc[0]

    pred_metrics = {}
    pred_path = _find_prediction_file_from_summary(proposed_summary)
    if pred_path is not None:
        try:
            pred_metrics = _metrics_from_prediction_df(pd.read_csv(pred_path))
            if pred_metrics:
                print(f"[SOFT] Loaded per-sample predictions for material_acc/MedAE: {pred_path}")
        except Exception as exc:
            print(f"[WARN] Could not compute soft per-sample metrics from {pred_path}: {exc}")

    soft_material_acc = _read_metric_from_row(
        row,
        ["test_material_acc", "material_acc", "mat_acc", "material_acc_pct", "mat_acc_pct"],
        default=pred_metrics.get("test_material_acc", np.nan),
    )

    soft_soc_medae = _read_metric_from_row(
        row,
        ["test_soc_medae_raw", "soc_medae_raw", "test_soc_medae", "soc_medae"],
        default=pred_metrics.get("test_soc_medae_raw", np.nan),
    )
    soft_soh_medae = _read_metric_from_row(
        row,
        ["test_soh_medae_raw", "soh_medae_raw", "test_soh_medae", "soh_medae"],
        default=pred_metrics.get("test_soh_medae_raw", np.nan),
    )
    soft_soh_medae = _scale_soh_error_to_percent(soft_soh_medae)

    out = {
        "config": "material_soft",
        "material_condition_mode": "soft",
        "source": str(proposed_summary),
        "soft_prediction_source": str(pred_path) if pred_path is not None else "not_found",
        "final_stage": "proposed_further_analysis",
        "test_cls_acc": _read_metric_from_row(row, ["test_cls_acc", "cls_acc"]),
        "test_material_acc": soft_material_acc,

        "test_soc_rmse": _read_metric_from_row(row, ["test_soc_rmse", "soc_rmse"]),
        "test_soc_mae": _read_metric_from_row(row, ["test_soc_mae", "soc_mae"]),
        "test_soc_medae": soft_soc_medae,
        "test_soc_mape": _read_metric_from_row(row, ["test_soc_mape", "soc_mape"]),
        "test_soc_medape": _read_metric_from_row(row, ["test_soc_medape", "soc_medape"]),

        "test_soh_rmse": _scale_soh_error_to_percent(
            _read_metric_from_row(row, ["test_soh_rmse", "soh_rmse"])
        ),
        "test_soh_mae": _scale_soh_error_to_percent(
            _read_metric_from_row(row, ["test_soh_mae", "soh_mae"])
        ),
        "test_soh_medae": soft_soh_medae,
        "test_soh_mape": _read_metric_from_row(row, ["test_soh_mape", "soh_mape"]),
        "test_soh_medape": _read_metric_from_row(row, ["test_soh_medape", "soh_medape"]),

        "test_soc_rmse_raw": _read_metric_from_row(row, ["test_soc_rmse_raw", "soc_rmse_raw", "test_soc_rmse", "soc_rmse"]),
        "test_soc_mae_raw": _read_metric_from_row(row, ["test_soc_mae_raw", "soc_mae_raw", "test_soc_mae", "soc_mae"]),
        "test_soc_medae_raw": soft_soc_medae,
        "test_soc_mape_raw": _read_metric_from_row(row, ["test_soc_mape_raw", "soc_mape_raw", "test_soc_mape", "soc_mape"]),
        "test_soc_medape_raw": _read_metric_from_row(row, ["test_soc_medape_raw", "soc_medape_raw", "test_soc_medape", "soc_medape"]),

        "test_soh_rmse_raw": _scale_soh_error_to_percent(
            _read_metric_from_row(row, ["test_soh_rmse_raw", "soh_rmse_raw", "test_soh_rmse", "soh_rmse"])
        ),
        "test_soh_mae_raw": _scale_soh_error_to_percent(
            _read_metric_from_row(row, ["test_soh_mae_raw", "soh_mae_raw", "test_soh_mae", "soh_mae"])
        ),
        "test_soh_medae_raw": soft_soh_medae,
        "test_soh_mape_raw": _read_metric_from_row(row, ["test_soh_mape_raw", "soh_mape_raw", "test_soh_mape", "soh_mape"]),
        "test_soh_medape_raw": _read_metric_from_row(row, ["test_soh_medape_raw", "soh_medape_raw", "test_soh_medape", "soh_medape"]),

        "n_train": np.nan,
        "n_test": int(_read_metric_from_row(row, ["n_test", "n"], default=0)),
        "num_classes": np.nan,
        "device": "from_proposed_further_analysis",
        "elapsed_sec": 0.0,
    }

    if np.isnan(out["test_material_acc"]):
        print("[WARN] soft material_acc is NaN because no compatible per-sample prediction file or summary column was found.")
    if np.isnan(out["test_soc_medae_raw"]):
        print("[WARN] soft SOC MedAE is NaN because no compatible per-sample prediction file or summary column was found.")
    if np.isnan(out["test_soh_medae_raw"]):
        print("[WARN] soft SOH MedAE is NaN because no compatible per-sample prediction file or summary column was found.")

    return out


# =============================================================================
# Hard-model stable test evaluation inside this script
# =============================================================================

def _find_best_checkpoint(exp_dir: str | Path, final_stage: str = "finetune") -> Path:
    exp_dir = Path(exp_dir)
    candidates = []

    preferred = [
        exp_dir / "checkpoints" / str(final_stage) / "best.pt",
        exp_dir / str(final_stage) / "best.pt",
        exp_dir / "checkpoints" / "stage2_soh" / "best.pt",
        exp_dir / "stage2_soh" / "best.pt",
        exp_dir / "checkpoints" / "stage1_soc" / "best.pt",
        exp_dir / "stage1_soc" / "best.pt",
    ]

    for p in preferred:
        if p.exists():
            return p

    candidates = list(exp_dir.rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError(f"No best.pt checkpoint found under: {exp_dir}")

    # Prefer finetune if present, otherwise newest best.pt.
    for p in candidates:
        if str(final_stage) in p.parts:
            return p

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model", "model_state_dict", "state_dict"]:
            if key in ckpt:
                return ckpt[key]
    return ckpt


def _prepare_hard_test_loader(
    data_root: str | Path,
    exp_dir: str | Path,
    pulse_list: List[int],
    u_start: int,
    u_end: int,
    drop_first_class: bool,
    soc_col: str,
    soh_col: str,
    use_pt_as_feature: bool,
    batch_size: int,
    num_workers: int,
    seed: int,
    test_id_frac: float,
    test_id_count: int,
    val_id_frac: float,
    val_id_count: int,
    normalize_soc: bool,
    zscore_normalize: bool,
) -> dict:
    data_root = Path(data_root)
    exp_dir = Path(exp_dir)
    cache_dir = exp_dir / "cache_stable_eval"
    splits_dir = exp_dir / "splits_stable_eval"
    ensure_dir(str(cache_dir), str(splits_dir))

    soc_list = list(range(5, 90, 5))

    Xtr_raw, ytr_raw, mtr_raw, tag_tr, hit_tr = load_or_build_cache(
        str(cache_dir),
        "raw_train",
        build_train_mix_soc_mix_pt,
        {
            "data_root": str(data_root),
            "soc_list": soc_list,
            "pulse_list": list(map(int, pulse_list)),
            "u_start": u_start,
            "u_end": u_end,
            "drop_first_class": drop_first_class,
        },
    )

    Xte_raw, yte_raw, mte_raw, tag_te, hit_te = load_or_build_cache(
        str(cache_dir),
        "raw_test",
        build_test_random_mix_pt,
        {
            "data_root": str(data_root),
            "pulse_list": list(map(int, pulse_list)),
            "u_start": u_start,
            "u_end": u_end,
            "drop_first_class": drop_first_class,
        },
    )

    Xtr_raw, ytr_raw, mtr_raw = drop_nan_inf_rows(
        Xtr_raw,
        ytr_raw,
        mtr_raw,
        name="RAW_TRAIN_STABLE_EVAL",
    )

    Xte_raw, yte_raw, mte_raw = drop_nan_inf_rows(
        Xte_raw,
        yte_raw,
        mte_raw,
        name="RAW_TEST_STABLE_EVAL",
    )

    all_ids = pd.concat([mtr_raw["ID"], mte_raw["ID"]], axis=0).astype(str).to_numpy()
    test_ids = np.asarray(
        pick_test_ids(
            all_ids=all_ids,
            test_id_frac=test_id_frac,
            test_id_count=test_id_count,
            seed=seed,
        )
    ).astype(str)

    train_candidate_ids = (
        pd.Series(mtr_raw["ID"].astype(str).unique())
        .loc[lambda s: ~s.isin(set(test_ids))]
        .to_numpy()
    )

    val_ids = np.asarray(
        pick_test_ids(
            all_ids=train_candidate_ids,
            test_id_frac=float(val_id_frac),
            test_id_count=val_id_count,
            seed=seed + 1,
        )
    ).astype(str)

    test_id_set = set(map(str, test_ids))
    val_id_set = set(map(str, val_ids))

    train_mask = (
        ~mtr_raw["ID"].astype(str).isin(test_id_set)
        & ~mtr_raw["ID"].astype(str).isin(val_id_set)
    ).to_numpy()
    test_mask = mte_raw["ID"].astype(str).isin(test_id_set).to_numpy()

    Xtr = Xtr_raw[train_mask]
    ytr_str = ytr_raw[train_mask]
    mtr = mtr_raw.loc[train_mask].reset_index(drop=True)

    Xte = Xte_raw[test_mask]
    yte_str = yte_raw[test_mask]
    mte = mte_raw.loc[test_mask].reset_index(drop=True)

    if len(ytr_str) == 0 or len(yte_str) == 0:
        raise RuntimeError(
            f"Empty train/test after stable-eval split: n_train={len(ytr_str)}, n_test={len(yte_str)}"
        )

    u_mean = Xtr.mean(axis=0, keepdims=True)
    u_std = Xtr.std(axis=0, keepdims=True) + 1e-8
    Xte = (Xte - u_mean) / u_std

    soc_train = mtr[soc_col].astype(float).to_numpy(dtype=np.float64)
    if normalize_soc:
        soc_train = soc_train / 100.0
    soc_norm = (float(soc_train.mean()), float(soc_train.std() + 1e-8))

    soh_train = mtr[soh_col].astype(float).to_numpy(dtype=np.float64)
    soh_norm = (float(soh_train.mean()), float(soh_train.std() + 1e-8))

    label_encoder = LabelEncoder()
    label_encoder.fit(ytr_str)
    train_classes = set(label_encoder.classes_.tolist())

    mask_test_known = np.array([label in train_classes for label in yte_str], dtype=bool)
    if not mask_test_known.all():
        n_removed = int((~mask_test_known).sum())
        print(f"[WARN] Removing {n_removed} stable-eval test samples with unseen labels.")
        Xte = Xte[mask_test_known]
        yte_str = yte_str[mask_test_known]
        mte = mte.loc[mask_test_known].reset_index(drop=True)

    yte_cls = label_encoder.transform(yte_str)

    if use_pt_as_feature and "pulse_ms" in mtr.columns:
        pt_log = np.log1p(mtr["pulse_ms"].astype(float).to_numpy(dtype=np.float32))
        pt_norm = (float(pt_log.mean()), float(pt_log.std() + 1e-8))
    else:
        pt_norm = (0.0, 1.0)

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
    )

    dl_te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return {
        "dl_te": dl_te,
        "label_encoder": label_encoder,
        "meta_test": mte,
        "true_label_strings": yte_str,
        "num_classes": int(len(label_encoder.classes_)),
        "soc_norm": soc_norm,
        "soh_norm": soh_norm,
        "n_train": int(len(ytr_str)),
        "n_test": int(len(ds_te)),
    }


@torch.no_grad()
def _stable_eval_hard_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    label_encoder: LabelEncoder,
    meta_test: pd.DataFrame,
    soc_norm: Optional[Tuple[float, float]],
    soh_norm: Optional[Tuple[float, float]],
    normalize_soc: bool,
    zscore_normalize: bool,
    n_mc: int,
) -> dict:
    model.eval()

    y_true_all = []
    y_pred_all = []
    soc_true_all = []
    soc_pred_all = []
    soh_true_all = []
    soh_pred_all = []

    for x_img, x_pt, y_cls, soc, soh in loader:
        x_img = x_img.to(device)
        x_pt = x_pt.to(device)
        y_cls = y_cls.to(device)
        soc = soc.to(device).view(-1)
        soh = soh.to(device).view(-1)

        logits, soc_pred, soc_logp, cond_soc, soh_pred, cond_soh = model(
            x_img=x_img,
            x_pt=x_pt,
            soc_tf=None,
            n_mc=int(n_mc),
        )

        y_true_all.append(y_cls.detach().cpu().numpy())
        y_pred_all.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        soc_true_all.append(soc.detach().cpu().numpy())
        soc_pred_all.append(soc_pred.detach().cpu().numpy())
        soh_true_all.append(soh.detach().cpu().numpy())
        soh_pred_all.append(soh_pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all).astype(int)
    y_pred = np.concatenate(y_pred_all).astype(int)
    soc_true = np.concatenate(soc_true_all)
    soc_pred = np.concatenate(soc_pred_all)
    soh_true = np.concatenate(soh_true_all)
    soh_pred = np.concatenate(soh_pred_all)

    true_labels = label_encoder.inverse_transform(y_true)
    pred_labels = label_encoder.inverse_transform(y_pred)

    soc_true_raw, soh_true_raw = _inverse_targets(
        soc_true,
        soh_true,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
    )
    soc_pred_raw, soh_pred_raw = _inverse_targets(
        soc_pred,
        soh_pred,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
    )

    pred_df = pd.DataFrame(
        {
            "true_label": true_labels.astype(str),
            "pred_label": pred_labels.astype(str),
            "true_material": [_split_material(x) for x in true_labels],
            "pred_material": [_split_material(x) for x in pred_labels],
            "cls_correct": true_labels.astype(str) == pred_labels.astype(str),
            "material_correct": [
                _split_material(t) == _split_material(p)
                for t, p in zip(true_labels, pred_labels)
            ],
            "soc_true": soc_true_raw,
            "soc_pred": soc_pred_raw,
            "soc_ae": np.abs(soc_pred_raw - soc_true_raw),
            "soh_true": soh_true_raw,
            "soh_pred": soh_pred_raw,
            "soh_ae": np.abs(soh_pred_raw - soh_true_raw),
        }
    )

    if len(meta_test) == len(pred_df):
        for col in ["ID", "pulse_ms", "SOC", "SOH"]:
            if col in meta_test.columns and col not in pred_df.columns:
                pred_df[col] = meta_test[col].to_numpy()

    return {
        "cls_acc": float(accuracy_score(y_true, y_pred)),
        "material_acc": _material_accuracy_from_labels(true_labels, pred_labels),

        "soc_rmse": float(rmse(soc_true, soc_pred)),
        "soc_mae": float(mae(soc_true, soc_pred)),
        "soc_medae": _medae(soc_true, soc_pred),
        "soc_mape": float(mape(soc_true, soc_pred)),
        "soc_medape": float(medape(soc_true, soc_pred)),

        "soh_rmse": float(rmse(soh_true, soh_pred)),
        "soh_mae": float(mae(soh_true, soh_pred)),
        "soh_medae": _medae(soh_true, soh_pred),
        "soh_mape": float(mape(soh_true, soh_pred)),
        "soh_medape": float(medape(soh_true, soh_pred)),

        "soc_rmse_raw": float(rmse(soc_true_raw, soc_pred_raw)),
        "soc_mae_raw": float(mae(soc_true_raw, soc_pred_raw)),
        "soc_medae_raw": _medae(soc_true_raw, soc_pred_raw),
        "soc_mape_raw": float(mape(soc_true_raw, soc_pred_raw)),
        "soc_medape_raw": float(medape(soc_true_raw, soc_pred_raw)),

        "soh_rmse_raw": float(rmse(soh_true_raw, soh_pred_raw)),
        "soh_mae_raw": float(mae(soh_true_raw, soh_pred_raw)),
        "soh_medae_raw": _medae(soh_true_raw, soh_pred_raw),
        "soh_mape_raw": float(mape(soh_true_raw, soh_pred_raw)),
        "soh_medape_raw": float(medape(soh_true_raw, soh_pred_raw)),

        "predictions": pred_df,
    }


def run_hard_stable_final_eval(
    exp_dir: str | Path,
    data_root: str | Path,
    pulse_list: List[int],
    final_stage: str,
    model_kwargs: dict,
    split_kwargs: dict,
    final_eval_n_mc_soc: int = 500,
    final_eval_n_mc_soh: int = 500,
) -> dict:
    exp_dir = Path(exp_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_data = _prepare_hard_test_loader(
        data_root=data_root,
        exp_dir=exp_dir,
        pulse_list=pulse_list,
        **split_kwargs,
    )

    model = MaterialConditioningHier3HeadModel(
        num_classes=eval_data["num_classes"],
        material_condition_mode="hard",
        default_n_mc=max(int(final_eval_n_mc_soc), int(final_eval_n_mc_soh)),
        **model_kwargs,
    ).to(device)

    best_path = _find_best_checkpoint(exp_dir=exp_dir, final_stage=final_stage)
    ckpt = _torch_load(best_path, map_location=device)
    model.load_state_dict(_extract_state_dict(ckpt))

    final_n_mc = max(int(final_eval_n_mc_soc), int(final_eval_n_mc_soh))
    print(
        f"[FINAL] hard stable test evaluation from {best_path} with "
        f"n_mc_soc={int(final_eval_n_mc_soc)}, n_mc_soh={int(final_eval_n_mc_soh)}"
    )

    te = _stable_eval_hard_model(
        model=model,
        loader=eval_data["dl_te"],
        device=device,
        label_encoder=eval_data["label_encoder"],
        meta_test=eval_data["meta_test"],
        soc_norm=eval_data["soc_norm"],
        soh_norm=eval_data["soh_norm"],
        normalize_soc=split_kwargs["normalize_soc"],
        zscore_normalize=split_kwargs["zscore_normalize"],
        n_mc=final_n_mc,
    )

    pred_df = te.pop("predictions")

    out = {
        "config": "material_hard",
        "material_condition_mode": "hard",
        "source": str(best_path),
        "final_stage": final_stage,
        "test_cls_acc": float(te["cls_acc"]),
        "test_material_acc": float(te["material_acc"]),

        "test_soc_rmse": float(te["soc_rmse"]),
        "test_soc_mae": float(te["soc_mae"]),
        "test_soc_medae": float(te["soc_medae"]),
        "test_soc_mape": float(te["soc_mape"]),
        "test_soc_medape": float(te["soc_medape"]),

        "test_soh_rmse": float(te["soh_rmse"]),
        "test_soh_mae": float(te["soh_mae"]),
        "test_soh_medae": float(te["soh_medae"]),
        "test_soh_mape": float(te["soh_mape"]),
        "test_soh_medape": float(te["soh_medape"]),

        "test_soc_rmse_raw": float(te["soc_rmse_raw"]),
        "test_soc_mae_raw": float(te["soc_mae_raw"]),
        "test_soc_medae_raw": float(te["soc_medae_raw"]),
        "test_soc_mape_raw": float(te["soc_mape_raw"]),
        "test_soc_medape_raw": float(te["soc_medape_raw"]),

        "test_soh_rmse_raw": float(te["soh_rmse_raw"]),
        "test_soh_mae_raw": float(te["soh_mae_raw"]),
        "test_soh_medae_raw": float(te["soh_medae_raw"]),
        "test_soh_mape_raw": float(te["soh_mape_raw"]),
        "test_soh_medape_raw": float(te["soh_medape_raw"]),

        "n_train": int(eval_data["n_train"]),
        "n_test": int(eval_data["n_test"]),
        "num_classes": int(eval_data["num_classes"]),
        "device": device,
        "elapsed_sec": np.nan,
    }

    metrics_dir = exp_dir / "metrics_stable_eval"
    ensure_dir(str(metrics_dir))

    pred_path = metrics_dir / "hard_retrospective_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    pd.DataFrame([out]).to_csv(
        metrics_dir / "final_metrics_nmc500.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _save_json(metrics_dir / "final_metrics_nmc500.json", out)

    print(f"[SAVED] {pred_path}")

    return out


# =============================================================================
# Main ablation runner
# =============================================================================

def run_material_conditioning_ablation(
    data_root: str | Path,
    output_root: str | Path,
    proposed_summary: str | Path = DEFAULT_PROPOSED_SUMMARY,
    config: str = "both",
    smoke: bool = False,
    resume: bool = True,
    summary_only: bool = False,
) -> pd.DataFrame:
    """
    Material-conditioning ablation.

    summary_only=True:
        soft: read proposed further-analysis summary.
        hard: skip training, load existing material_hard checkpoint, and re-evaluate.

    summary_only=False:
        keep the original behavior for hard: train/resume first, then stable evaluation.
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []

    if config in {"soft", "both"}:
        print("\n" + "=" * 90)
        print("[LOAD] Material conditioning: soft")
        print(f"[LOAD] Source summary: {proposed_summary}")
        print("[LOAD] Soft mode is the proposed model; no retraining is performed.")
        print("=" * 90)
        rows.append(read_soft_from_proposed_further_summary(proposed_summary))

    if smoke:
        pulse_list = [5000]
        train_kwargs = dict(
            data_root=str(data_root),
            pulse_list=pulse_list,
            batch_size=32,
            max_epochs=1,
            early_stopping=False,
            patience=1,
            resume=False,
            num_workers=0,
            seed=42,
            width=16,
            blocks=1,
            drop2d=0.0,
            head_dropout=0.1,
            two_stage=False,
            stage1_epochs=1,
            stage2_epochs=1,
            finetune_epochs=0,
            final_best_stage="single",
            use_soc_prior_weighting=False,
            use_soh_prior_weighting=False,
            normalize_soc=True,
            zscore_normalize=True,
            test_id_frac=0.2,
            test_id_count=0,
            val_id_frac=0.1,
            val_id_count=0,
        )
    else:
        pulse_list = DEFAULT_PULSE_LIST
        train_kwargs = dict(
            data_root=str(data_root),
            pulse_list=pulse_list,
            batch_size=128,
            max_epochs=400,
            early_stopping=False,
            patience=20,
            resume=resume,
            num_workers=0,
            seed=42,
            width=32,
            blocks=4,
            drop2d=0.0,
            head_dropout=0.2,
            two_stage=True,
            stage1_epochs=200,
            stage2_epochs=200,
            finetune_epochs=30,
            final_best_stage="finetune",
            use_soc_prior_weighting=True,
            use_soh_prior_weighting=True,
            normalize_soc=True,
            zscore_normalize=True,
            test_id_frac=0.2,
            test_id_count=0,
            val_id_frac=0.1,
            val_id_count=0,
        )

    if config in {"hard", "both"}:
        hard_exp_dir = output_root / "material_hard"

        print("\n" + "=" * 90)
        print("[RUN] Material conditioning: hard")
        print(f"[RUN] Output directory: {hard_exp_dir}")
        if summary_only:
            print("[SUMMARY-ONLY] Skip M.run_experiment(); load existing checkpoint only.")
        print("=" * 90)

        train_out = {}

        if not summary_only:
            original_model_ref = M.Hier3HeadModel

            try:
                def _hard_model_factory(*args, **kwargs):
                    return MaterialConditioningHier3HeadModel(
                        *args,
                        material_condition_mode="hard",
                        default_n_mc=16,
                        **kwargs,
                    )

                M.Hier3HeadModel = _hard_model_factory

                train_out = M.run_experiment(
                    exp_dir=hard_exp_dir,
                    **train_kwargs,
                )

            finally:
                M.Hier3HeadModel = original_model_ref

        model_kwargs = dict(
            width=int(train_kwargs["width"]),
            blocks=int(train_kwargs["blocks"]),
            drop2d=float(train_kwargs["drop2d"]),
            use_pt_as_feature=True,
            head_dropout=float(train_kwargs["head_dropout"]),
        )

        split_kwargs = dict(
            u_start=1,
            u_end=41,
            drop_first_class=True,
            soc_col="SOC",
            soh_col="SOH",
            use_pt_as_feature=True,
            batch_size=int(train_kwargs["batch_size"]),
            num_workers=int(train_kwargs["num_workers"]),
            seed=int(train_kwargs["seed"]),
            test_id_frac=float(train_kwargs["test_id_frac"]),
            test_id_count=int(train_kwargs["test_id_count"]),
            val_id_frac=float(train_kwargs["val_id_frac"]),
            val_id_count=int(train_kwargs["val_id_count"]),
            normalize_soc=bool(train_kwargs["normalize_soc"]),
            zscore_normalize=bool(train_kwargs["zscore_normalize"]),
        )

        hard_out = run_hard_stable_final_eval(
            exp_dir=hard_exp_dir,
            data_root=data_root,
            pulse_list=pulse_list,
            final_stage=str(train_kwargs["final_best_stage"]),
            model_kwargs=model_kwargs,
            split_kwargs=split_kwargs,
            final_eval_n_mc_soc=500,
            final_eval_n_mc_soh=500,
        )

        hard_out["elapsed_sec"] = (
            float(train_out.get("elapsed_sec", np.nan))
            if isinstance(train_out, dict)
            else np.nan
        )
        rows.append(hard_out)

    summary = pd.DataFrame(rows)
    summary = _add_summary_columns(summary)

    summary_csv = output_root / "material_conditioning_ablation_summary.csv"
    summary_json = output_root / "material_conditioning_ablation_summary.json"
    partial_csv = output_root / "material_conditioning_ablation_partial.csv"
    partial_json = output_root / "material_conditioning_ablation_partial.json"

    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(partial_csv, index=False, encoding="utf-8-sig")
    _save_json(summary_json, summary.to_dict(orient="records"))
    _save_json(partial_json, summary.to_dict(orient="records"))

    print("\n[SUMMARY]")
    print(summary)
    print(f"\n[SAVED] {summary_csv}")
    print(f"[SAVED] {summary_json}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="both",
        choices=["soft", "hard", "both"],
        help="soft: read proposed further summary only; hard: evaluate hard only; both: combine both.",
    )
    parser.add_argument(
        "--proposed-summary",
        type=str,
        default=str(DEFAULT_PROPOSED_SUMMARY),
        help="Path to proposed_framework/further_analysis/tables/proposed_method_summary.csv.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root. If omitted, uses PROJECT_ROOT/data.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Do not train. Load existing hard checkpoint and recalculate material accuracy and MedAE.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root is not None else PROJECT_ROOT / "data"
    output_root = PROJECT_ROOT / "results" / "ablation" / "material_conditioning_ablation"

    run_material_conditioning_ablation(
        data_root=data_root,
        output_root=output_root,
        proposed_summary=args.proposed_summary,
        config=args.config,
        smoke=bool(args.smoke),
        resume=not bool(args.no_resume),
        summary_only=bool(args.summary_only),
    )


if __name__ == "__main__":
    main()
