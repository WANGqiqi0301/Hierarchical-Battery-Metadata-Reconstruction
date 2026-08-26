# measurement_sensitivity/pulse_width_sensitivity.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from proposed_framework.run_proposed_framework import run_experiment
from utils.cache import load_or_build_cache, drop_nan_inf_rows
from proposed_framework.data.build_dataset import (
    build_train_mix_soc_mix_pt,
    build_test_random_mix_pt,
    pick_test_ids,
)
from proposed_framework.data.pulse_dataset import HierPulseDataset
from proposed_framework.models.hierarchical_model import Hier3HeadModel


PULSE_WIDTH_CONFIGS: Dict[str, List[int]] = {
    "P1_70": [70],
    "P2_3000": [3000],
    "P3_30_50_70_100": [30, 50, 70, 100],
    "P4_300_500_700": [300, 500, 700],
    "P5_1000_3000_5000": [1000, 3000, 5000],
    "P6_30_50_300_500": [30, 50, 300, 500],
    "P7_30_50_3000_5000": [30, 50, 3000, 5000],
    "P8_300_500_3000_5000": [300, 500, 3000, 5000],
}

REF_NAME = "P9_All"

N_MC_SOC = 500
N_MC_SOH = 500

SOC_COL = "SOC"
SOH_COL = "SOH"
ID_COL = "ID"
PT_COL = "pulse_ms"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _torch_load(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _medae(a, b):
    return float(np.median(np.abs(np.asarray(a) - np.asarray(b))))


def _mape(a, b, eps=1e-8):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs((b - a) / np.maximum(np.abs(a), eps))) * 100.0)


def _medape(a, b, eps=1e-8):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.median(np.abs((b - a) / np.maximum(np.abs(a), eps))) * 100.0)


def _inverse_targets(soc_z, soh_z, soc_norm, soh_norm, normalize_soc=True, zscore_normalize=True):
    soc = np.asarray(soc_z, dtype=np.float64)
    soh = np.asarray(soh_z, dtype=np.float64)

    if zscore_normalize:
        soc = soc * soc_norm[1] + soc_norm[0]
        soh = soh * soh_norm[1] + soh_norm[0]

    if normalize_soc:
        soc = soc * 100.0
    soh = soh * 100.0

    return soc, soh


def _resolve_checkpoint(exp_dir: Path) -> Path:
    run_cfg = _load_json(exp_dir / "run_config.json")
    final_stage = run_cfg.get("final_best_stage", "finetune")

    candidates = [
        exp_dir / "checkpoints" / str(final_stage) / "best.pt",
        exp_dir / "checkpoints" / "finetune" / "best.pt",
        exp_dir / "checkpoints" / "stage2_soh" / "best.pt",
        exp_dir / "checkpoints" / "stage1_soc" / "best.pt",
        exp_dir / "checkpoints" / "single" / "best.pt",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(f"No checkpoint found under {exp_dir / 'checkpoints'}")


def _load_p9_from_further():
    tables_dir = PROJECT_ROOT / "results" / "proposed_framework" / "further_analysis" / "tables"
    pred_path = tables_dir / "test_predictions_per_sample.csv"
    summary_path = tables_dir / "proposed_method_summary.csv"

    if pred_path.exists():
        df = pd.read_csv(pred_path)
        required = {"true_label", "pred_label", "soc_true", "soc_pred", "soh_true", "soh_pred"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"Missing columns in {pred_path}: {sorted(missing)}")

        true_labels = df["true_label"].astype(str).to_numpy()
        pred_labels = df["pred_label"].astype(str).to_numpy()
        true_material = np.array([x.split("_")[0] for x in true_labels])
        pred_material = np.array([x.split("_")[0] for x in pred_labels])

        soc_true = df["soc_true"].to_numpy(dtype=np.float64)
        soc_pred = df["soc_pred"].to_numpy(dtype=np.float64)
        soh_true = df["soh_true"].to_numpy(dtype=np.float64)
        soh_pred = df["soh_pred"].to_numpy(dtype=np.float64)

        if np.nanmedian(np.abs(soh_true)) <= 2.0:
            soh_true *= 100.0
            soh_pred *= 100.0

        return {
            "config": REF_NAME,
            "pulse_widths": "30,50,70,100,300,500,700,1000,3000,5000",
            "num_pulse_widths": 10,
            "pulse_width_sum_ms": 10750,
            "n_test": int(len(df)),
            "test_cls_acc": float(np.mean(true_labels == pred_labels)),
            "test_material_acc": float(np.mean(true_material == pred_material)),
            "test_soc_mae_raw": _mae(soc_true, soc_pred),
            "test_soc_medae_raw": _medae(soc_true, soc_pred),
            "test_soc_rmse_raw": _rmse(soc_true, soc_pred),
            "test_soc_mape_raw": _mape(soc_true, soc_pred),
            "test_soc_medape_raw": _medape(soc_true, soc_pred),
            "test_soh_mae_raw": _mae(soh_true, soh_pred),
            "test_soh_medae_raw": _medae(soh_true, soh_pred),
            "test_soh_rmse_raw": _rmse(soh_true, soh_pred),
            "test_soh_mape_raw": _mape(soh_true, soh_pred),
            "test_soh_medape_raw": _medape(soh_true, soh_pred),
            "n_mc_soc": N_MC_SOC,
            "n_mc_soh": N_MC_SOH,
            "checkpoint_path": "proposed_framework/further_analysis",
            "predictions_path": str(pred_path),
        }

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Cannot find either {pred_path} or {summary_path}. "
            "Run analysis/run_further_analysis_proposed.py first."
        )

    df = pd.read_csv(summary_path)
    row = df[df["split"].astype(str).str.lower() == "test"].iloc[0] if "split" in df.columns else df.iloc[0]

    def get_value(*names, default=np.nan):
        for name in names:
            if name in row.index and pd.notna(row[name]):
                return float(row[name])
        return float(default)

    return {
        "config": REF_NAME,
        "pulse_widths": "30,50,70,100,300,500,700,1000,3000,5000",
        "num_pulse_widths": 10,
        "pulse_width_sum_ms": 10750,
        "n_test": int(get_value("n", "n_test", default=0)),
        "test_cls_acc": get_value("cls_acc", "test_cls_acc"),
        "test_material_acc": get_value("material_acc", "test_material_acc"),
        "test_soc_mae_raw": get_value("soc_mae", "test_soc_mae_raw"),
        "test_soc_medae_raw": get_value("soc_medae", "test_soc_medae_raw"),
        "test_soc_rmse_raw": get_value("soc_rmse", "test_soc_rmse_raw"),
        "test_soc_mape_raw": get_value("soc_mape", "test_soc_mape_raw"),
        "test_soc_medape_raw": get_value("soc_medape", "test_soc_medape_raw"),
        "test_soh_mae_raw": get_value("soh_mae", "test_soh_mae_raw"),
        "test_soh_medae_raw": get_value("soh_medae", "test_soh_medae_raw"),
        "test_soh_rmse_raw": get_value("soh_rmse", "test_soh_rmse_raw"),
        "test_soh_mape_raw": get_value("soh_mape", "test_soh_mape_raw"),
        "test_soh_medape_raw": get_value("soh_medape", "test_soh_medape_raw"),
        "n_mc_soc": N_MC_SOC,
        "n_mc_soh": N_MC_SOH,
        "checkpoint_path": "proposed_framework/further_analysis",
        "predictions_path": "",
    }


def _add_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    summary = summary.copy()
    summary["Material-Capacity Accuracy"] = summary["test_cls_acc"].astype(float) * 100.0
    summary["Material Accuracy"] = summary["test_material_acc"].astype(float) * 100.0
    summary["SOC MedAE"] = summary["test_soc_medae_raw"].astype(float)
    summary["SOH MedAE"] = summary["test_soh_medae_raw"].astype(float)
    summary["SOC Median APE"] = summary["test_soc_medape_raw"].astype(float)
    summary["SOH Median APE"] = summary["test_soh_medape_raw"].astype(float)

    summary["cls_acc_pct"] = summary["test_cls_acc"].astype(float) * 100.0
    summary["material_acc_pct"] = summary["test_material_acc"].astype(float) * 100.0
    summary["mat_acc_pct"] = summary["cls_acc_pct"]
    summary["soc_medae_pp"] = summary["test_soc_medae_raw"].astype(float)
    summary["soh_medae_pp"] = summary["test_soh_medae_raw"].astype(float)
    summary["soc_medape_pct"] = summary["test_soc_medape_raw"].astype(float)
    summary["soh_medape_pct"] = summary["test_soh_medape_raw"].astype(float)

    if REF_NAME in set(summary["config"]):
        ref = summary.loc[summary["config"] == REF_NAME].iloc[0]
        summary["cls_acc_change_pp_vs_all"] = (
            summary["test_cls_acc"].astype(float) - float(ref["test_cls_acc"])
        ) * 100.0
        summary["material_acc_change_pp_vs_all"] = (
            summary["test_material_acc"].astype(float) - float(ref["test_material_acc"])
        ) * 100.0
        summary["soc_medae_change_pp_vs_all"] = (
            summary["test_soc_medae_raw"].astype(float) - float(ref["test_soc_medae_raw"])
        )
        summary["soh_medae_change_pp_vs_all"] = (
            summary["test_soh_medae_raw"].astype(float) - float(ref["test_soh_medae_raw"])
        )
        summary["soc_medape_change_pp_vs_all"] = (
            summary["test_soc_medape_raw"].astype(float) - float(ref["test_soc_medape_raw"])
        )
        summary["soh_medape_change_pp_vs_all"] = (
            summary["test_soh_medape_raw"].astype(float) - float(ref["test_soh_medape_raw"])
        )

    return summary


def _build_test_context(
    data_root: Path,
    output_root: Path,
    config_name: str,
    pulse_list: List[int],
):
    exp_dir = output_root / config_name
    run_cfg = _load_json(exp_dir / "run_config.json")

    data_root = Path(data_root)

    seed = int(run_cfg.get("seed", 42))
    test_id_frac = float(run_cfg.get("test_id_frac", 0.2))
    test_id_count = int(run_cfg.get("test_id_count", 0))
    batch_size = int(run_cfg.get("batch_size", 128))

    u_start = int(run_cfg.get("u_start", 1))
    u_end = int(run_cfg.get("u_end", 41))
    drop_first_class = bool(run_cfg.get("drop_first_class", True))

    normalize_soc = bool(run_cfg.get("normalize_soc", True))
    zscore_normalize = bool(run_cfg.get("zscore_normalize", True))
    use_pt_as_feature = bool(run_cfg.get("use_pt_as_feature", True))

    cache_dir = exp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    soc_list = list(range(5, 90, 5))

    Xtr_raw, ytr_raw, mtr_raw, _, _ = load_or_build_cache(
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

    Xte_raw, yte_raw, mte_raw, _, _ = load_or_build_cache(
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
        Xtr_raw, ytr_raw, mtr_raw, name="RAW_TRAIN"
    )
    Xte_raw, yte_raw, mte_raw = drop_nan_inf_rows(
        Xte_raw, yte_raw, mte_raw, name="RAW_TEST"
    )

    all_ids = pd.concat(
        [mtr_raw[ID_COL], mte_raw[ID_COL]],
        axis=0,
    ).astype(str).to_numpy()

    test_ids = pick_test_ids(
        all_ids=all_ids,
        test_id_frac=test_id_frac,
        test_id_count=test_id_count,
        seed=seed,
    )

    test_id_set = set(map(str, test_ids))

    train_mask = ~mtr_raw[ID_COL].astype(str).isin(test_id_set).to_numpy()
    test_mask = mte_raw[ID_COL].astype(str).isin(test_id_set).to_numpy()

    Xtr = Xtr_raw[train_mask]
    ytr_str = np.asarray(ytr_raw)[train_mask]
    mtr = mtr_raw.loc[train_mask].reset_index(drop=True)

    Xte = Xte_raw[test_mask]
    yte_str = np.asarray(yte_raw)[test_mask]
    mte = mte_raw.loc[test_mask].reset_index(drop=True)

    norm_path = exp_dir / "u41_norm_train_only.npz"
    if norm_path.exists():
        obj = np.load(norm_path)
        u_mean = obj["u_mean"]
        u_std = obj["u_std"]
    else:
        u_mean = Xtr.mean(axis=0, keepdims=True)
        u_std = Xtr.std(axis=0, keepdims=True) + 1e-8

    Xte = (Xte - u_mean) / (u_std + 1e-8)

    target_norm_path = exp_dir / "target_norm_train_only.npz"
    if target_norm_path.exists():
        obj = np.load(target_norm_path)
        soc_norm = (float(obj["soc_mean"][0]), float(obj["soc_std"][0]))
        soh_norm = (float(obj["soh_mean"][0]), float(obj["soh_std"][0]))
    else:
        soc_train = mtr[SOC_COL].astype(float).to_numpy(dtype=np.float64)
        if normalize_soc:
            soc_train = soc_train / 100.0
        soc_norm = (float(soc_train.mean()), float(soc_train.std() + 1e-8))

        soh_train = mtr[SOH_COL].astype(float).to_numpy(dtype=np.float64)
        soh_norm = (float(soh_train.mean()), float(soh_train.std() + 1e-8))

    label_encoder = LabelEncoder()
    label_encoder.fit(ytr_str)

    train_classes = set(label_encoder.classes_.tolist())
    known_mask = np.array(
        [label in train_classes for label in yte_str],
        dtype=bool,
    )

    if not known_mask.all():
        Xte = Xte[known_mask]
        yte_str = yte_str[known_mask]
        mte = mte.loc[known_mask].reset_index(drop=True)

    yte_cls = label_encoder.transform(yte_str)

    if use_pt_as_feature and PT_COL in mtr.columns:
        pt_train = mtr[PT_COL].astype(float).to_numpy(dtype=np.float64)
        pt_log = np.log1p(pt_train)
        pt_norm = (float(pt_log.mean()), float(pt_log.std() + 1e-8))
    else:
        pt_norm = (0.0, 1.0)

    ds_te = HierPulseDataset(
        X_u=Xte,
        y_cls=yte_cls,
        meta=mte,
        soc_col=SOC_COL,
        soh_col=SOH_COL,
        pt_col=PT_COL,
        use_pt_as_feature=use_pt_as_feature,
        pt_norm=pt_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
    )

    dl_te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return {
        "exp_dir": exp_dir,
        "run_cfg": run_cfg,
        "dl_te": dl_te,
        "soc_norm": soc_norm,
        "soh_norm": soh_norm,
        "normalize_soc": normalize_soc,
        "zscore_normalize": zscore_normalize,
        "num_classes": int(len(label_encoder.classes_)),
        "label_encoder": label_encoder,
        "mte": mte,
    }


@torch.no_grad()
def _evaluate_one_config_further(
    data_root: Path,
    output_root: Path,
    config_name: str,
    pulse_list: List[int],
) -> Dict:
    print(f"\n[FURTHER TEST] {config_name} -> {pulse_list}")

    ctx = _build_test_context(
        data_root=data_root,
        output_root=output_root,
        config_name=config_name,
        pulse_list=pulse_list,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = ctx["exp_dir"]
    run_cfg = ctx["run_cfg"]

    model = Hier3HeadModel(
        num_classes=ctx["num_classes"],
        width=int(run_cfg.get("width", 32)),
        blocks=int(run_cfg.get("blocks", 4)),
        drop2d=float(run_cfg.get("drop2d", 0.0)),
        use_pt_as_feature=bool(run_cfg.get("use_pt_as_feature", True)),
        head_dropout=float(run_cfg.get("head_dropout", 0.2)),
    ).to(device)

    ckpt_path = _resolve_checkpoint(exp_dir)
    ckpt = _torch_load(ckpt_path, device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    model.eval()

    y_true_all, y_pred_all = [], []
    soc_true_all, soc_pred_all = [], []
    soh_true_all, soh_pred_all = [], []

    for x3, pt, y_cls, soc, soh in ctx["dl_te"]:
        x3 = x3.to(device)
        pt = pt.to(device)

        logits, soc_pred, _, _, soh_pred, _ = model(
            x_img=x3,
            x_pt=pt,
            soc_tf=None,
            n_mc=N_MC_SOC,
        )

        y_true_all.append(y_cls.numpy())
        y_pred_all.append(logits.argmax(dim=1).detach().cpu().numpy())

        soc_true_raw, soh_true_raw = _inverse_targets(
            soc.detach().cpu().numpy(),
            soh.detach().cpu().numpy(),
            soc_norm=ctx["soc_norm"],
            soh_norm=ctx["soh_norm"],
            normalize_soc=ctx["normalize_soc"],
            zscore_normalize=ctx["zscore_normalize"],
        )
        soc_pred_raw, soh_pred_raw = _inverse_targets(
            soc_pred.detach().cpu().numpy(),
            soh_pred.detach().cpu().numpy(),
            soc_norm=ctx["soc_norm"],
            soh_norm=ctx["soh_norm"],
            normalize_soc=ctx["normalize_soc"],
            zscore_normalize=ctx["zscore_normalize"],
        )

        soc_true_all.append(soc_true_raw)
        soc_pred_all.append(soc_pred_raw)
        soh_true_all.append(soh_true_raw)
        soh_pred_all.append(soh_pred_raw)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    soc_true_all = np.concatenate(soc_true_all)
    soc_pred_all = np.concatenate(soc_pred_all)
    soh_true_all = np.concatenate(soh_true_all)
    soh_pred_all = np.concatenate(soh_pred_all)

    label_encoder = ctx["label_encoder"]
    true_labels = label_encoder.inverse_transform(y_true_all)
    pred_labels = label_encoder.inverse_transform(y_pred_all)
    true_material = np.array([x.split("_")[0] for x in true_labels])
    pred_material = np.array([x.split("_")[0] for x in pred_labels])

    mte = ctx["mte"].reset_index(drop=True)
    predictions = pd.DataFrame({
        "true_label": true_labels,
        "pred_label": pred_labels,
        "true_material": true_material,
        "pred_material": pred_material,
        "material_correct": true_material == pred_material,
        "soc_true": soc_true_all,
        "soc_pred": soc_pred_all,
        "soh_true": soh_true_all,
        "soh_pred": soh_pred_all,
    })
    if ID_COL in mte.columns:
        predictions.insert(0, ID_COL, mte[ID_COL].astype(str).to_numpy())
    if PT_COL in mte.columns:
        predictions["pulse_ms"] = mte[PT_COL].astype(float).to_numpy()

    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = metrics_dir / "retrospective_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    row = {
        "config": config_name,
        "pulse_widths": ",".join(map(str, pulse_list)),
        "num_pulse_widths": len(pulse_list),
        "pulse_width_sum_ms": int(sum(pulse_list)),
        "n_test": int(len(y_true_all)),
        "test_cls_acc": float(accuracy_score(y_true_all, y_pred_all)),
        "test_material_acc": float(accuracy_score(true_material, pred_material)),
        "test_soc_mae_raw": _mae(soc_true_all, soc_pred_all),
        "test_soc_medae_raw": _medae(soc_true_all, soc_pred_all),
        "test_soc_rmse_raw": _rmse(soc_true_all, soc_pred_all),
        "test_soc_mape_raw": _mape(soc_true_all, soc_pred_all),
        "test_soc_medape_raw": _medape(soc_true_all, soc_pred_all),
        "test_soh_mae_raw": _mae(soh_true_all, soh_pred_all),
        "test_soh_medae_raw": _medae(soh_true_all, soh_pred_all),
        "test_soh_rmse_raw": _rmse(soh_true_all, soh_pred_all),
        "test_soh_mape_raw": _mape(soh_true_all, soh_pred_all),
        "test_soh_medape_raw": _medape(soh_true_all, soh_pred_all),
        "n_mc_soc": N_MC_SOC,
        "n_mc_soh": N_MC_SOH,
        "checkpoint_path": str(ckpt_path),
        "predictions_path": str(predictions_path),
    }

    pd.DataFrame([row]).to_csv(
        metrics_dir / "retrospective_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with open(metrics_dir / "retrospective_metrics.json", "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)

    print(
        f"[RESULT] {config_name}: "
        f"fine={row['test_cls_acc']:.4f}, "
        f"material={row['test_material_acc']:.4f}, "
        f"SOC MedAE={row['test_soc_medae_raw']:.4f} pp, "
        f"SOH MedAE={row['test_soh_medae_raw']:.4f} pp"
    )

    return row


def run_summary_only_further(
    data_root: str | Path,
    output_root: str | Path,
    selected_config: str | None = None,
) -> pd.DataFrame:
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if selected_config is None:
        configs_to_run = PULSE_WIDTH_CONFIGS
    else:
        if selected_config not in PULSE_WIDTH_CONFIGS:
            valid = ", ".join(PULSE_WIDTH_CONFIGS.keys())
            raise ValueError(f"Unknown config: {selected_config}. Valid configs: {valid}")
        configs_to_run = {selected_config: PULSE_WIDTH_CONFIGS[selected_config]}

    rows = [
        _evaluate_one_config_further(
            data_root=data_root,
            output_root=output_root,
            config_name=config_name,
            pulse_list=pulse_list,
        )
        for config_name, pulse_list in configs_to_run.items()
    ]

    if selected_config is None:
        rows.append(_load_p9_from_further())

    summary = _add_summary(pd.DataFrame(rows))

    if selected_config is None:
        summary_path = output_root / "pulse_width_sensitivity_summary.csv"
        json_path = output_root / "pulse_width_sensitivity_summary.json"
    else:
        config_dir = output_root / selected_config
        config_dir.mkdir(parents=True, exist_ok=True)
        summary_path = config_dir / f"{selected_config}_retrospective_summary.csv"
        json_path = config_dir / f"{selected_config}_retrospective_summary.json"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] {summary_path}")
    print(f"[SAVED] {json_path}")
    print(
        summary[
            [
                "config",
                "Material-Capacity Accuracy",
                "Material Accuracy",
                "SOC MedAE",
                "SOH MedAE",
                "n_test",
            ]
        ]
    )

    return summary


def run_pulse_width_sensitivity(
    data_root: str | Path,
    output_root: str | Path,
    smoke: bool = False,
    resume: bool = True,
    selected_config: str | None = None,
) -> pd.DataFrame:

    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if selected_config is None:
        if smoke:
            configs_to_run = {
                "P2_3000": PULSE_WIDTH_CONFIGS["P2_3000"],
            }
            print("[MODE] Smoke test: running P2_3000 only")
        else:
            configs_to_run = PULSE_WIDTH_CONFIGS
            print("[MODE] Running all pulse-width configs: P1–P8")
    else:
        if selected_config not in PULSE_WIDTH_CONFIGS:
            valid = ", ".join(PULSE_WIDTH_CONFIGS.keys())
            raise ValueError(
                f"Unknown config: {selected_config}\n"
                f"Valid configs are: {valid}"
            )

        configs_to_run = {
            selected_config: PULSE_WIDTH_CONFIGS[selected_config]
        }
        print(f"[MODE] Running selected config only: {selected_config}")

    rows = []

    for config_name, pulse_list in configs_to_run.items():

        exp_dir = output_root / config_name

        print(f"\n[RUN] {config_name} -> {pulse_list}")
        print(f"[OUT] {exp_dir}")

        out = run_experiment(
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
            exp_dir=exp_dir,
            batch_size=128,
            max_epochs=1 if smoke else 400,
            early_stopping=False,
            patience=20,
            resume=resume,
            width=32,
            blocks=4,
            head_dropout=0.2,
            two_stage=False if smoke else True,
            stage1_epochs=1 if smoke else 200,
            stage2_epochs=1 if smoke else 200,
            finetune_epochs=1 if smoke else 30,
            use_soc_prior_weighting=True,
            use_soh_prior_weighting=True,
            final_best_stage="single" if smoke else "finetune",
        )

        rows.append({
            "config": config_name,
            "pulse_widths": ",".join(map(str, pulse_list)),
            "num_pulse_widths": len(pulse_list),
            "pulse_width_sum_ms": sum(pulse_list),
            **out,
        })

    if selected_config is None:
        p9 = _load_p9_from_further()
        rows.append(p9)

    summary = pd.DataFrame(rows)
    summary = _add_summary(summary)

    if selected_config is None:
        summary_path = output_root / "pulse_width_sensitivity_summary.csv"
    else:
        summary_path = output_root / selected_config / f"{selected_config}_summary.csv"

    summary.to_csv(summary_path, index=False)

    print(f"\n[SAVED] {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None, choices=list(PULSE_WIDTH_CONFIGS))
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Do not train. Reload P1-P8 checkpoints and run test-only further analysis with n_mc=500/500.",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else PROJECT_ROOT / "data"
    output_root = PROJECT_ROOT / "results" / "measurement_sensitivity" / "pulse_width"

    if args.summary_only:
        summary = run_summary_only_further(
            data_root=data_root,
            output_root=output_root,
            selected_config=args.config,
        )

    elif args.config is not None:
        summary = run_pulse_width_sensitivity(
            data_root=data_root,
            output_root=output_root,
            smoke=False,
            resume=not args.no_resume,
            selected_config=args.config,
        )

    else:
        run_pulse_width_sensitivity(
            data_root=data_root,
            output_root=output_root,
            smoke=False,
            resume=not args.no_resume,
            selected_config=None,
        )

        summary = run_summary_only_further(
            data_root=data_root,
            output_root=output_root,
        )

    print("\n[SUMMARY]")
    print(summary)


if __name__ == "__main__":
    main()