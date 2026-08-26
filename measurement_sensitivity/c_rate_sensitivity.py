# measurement_sensitivity/c_rate_sensitivity.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import sys
import argparse
import json
from typing import Dict, List, Optional

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


C_RATE_NAMES = {
    1: "0.5C",
    2: "1.0C",
    3: "1.5C",
    4: "2.0C",
    5: "2.5C",
}


C_RATE_CONFIGS: Dict[str, List[int]] = {
    "C1_0p5C": [1],
    "C2_1p5C": [3],
    "C3_2p5C": [5],

    "C4_0p5_1p0C": [1, 2],
    "C5_1p5_2p0C": [3, 4],
    "C6_2p0_2p5C": [4, 5],
    "C7_0p5_2p5C": [1, 5],

    "C8_0p5_1p0_1p5C": [1, 2, 3],
    "C9_0p5_1p5_2p5C": [1, 3, 5],
    "C10_1p5_2p0_2p5C": [3, 4, 5],

    # C11_All is the full-input reference.
    # In full mode, this script does NOT retrain it.
    # It is loaded from the main proposed further-analysis result instead.
    "C11_All": [1, 2, 3, 4, 5],
}


FULL_C_RATE_CONFIG = "C11_All"
FULL_C_RATE_COMBO = C_RATE_CONFIGS[FULL_C_RATE_CONFIG]


# Main proposed result used as the All-C-rate reference.
# The first existing file in this list will be used.
DEFAULT_FULL_REFERENCE_CANDIDATES = [
    PROJECT_ROOT
    / "results"
    / "proposed_framework"
    / "further_analysis"
    / "tables"
    / "proposed_method_summary.csv",
    PROJECT_ROOT
    / "results"
    / "proposed_framework"
    / "metrics"
    / "final_metrics.csv",
]


def _combo_to_label(combo: List[int]) -> str:
    return ",".join(C_RATE_NAMES[i] for i in combo)


def _read_csv_first_existing(paths: List[Path]) -> tuple[pd.DataFrame, Path]:
    for path in paths:
        path = Path(path)
        if path.exists():
            return pd.read_csv(path), path

    raise FileNotFoundError(
        "Cannot find full C-rate reference result. Tried:\n"
        + "\n".join(str(Path(p)) for p in paths)
    )


def _pick_first_available(row: pd.Series, names: List[str], default=None):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return default


def _load_full_reference_row(
    reference_csv: Optional[str | Path] = None,
) -> Dict:
    """
    Load C11_All from the main proposed result instead of retraining it.

    Preferred source:
        results/proposed_framework/further_analysis/tables/proposed_method_summary.csv

    Fallback:
        results/proposed_framework/metrics/final_metrics.csv

    The function is intentionally tolerant of slightly different column names,
    because different further-analysis scripts may save either direct final
    metric names or compact summary names.
    """
    if reference_csv is not None:
        candidates = [Path(reference_csv)]
    else:
        candidates = [Path(p) for p in DEFAULT_FULL_REFERENCE_CANDIDATES]

    df, used_path = _read_csv_first_existing(candidates)

    if df.empty:
        raise RuntimeError(f"Full C-rate reference file is empty: {used_path}")

    # If this is a train/val/test further-analysis summary, use TEST row.
    row_df = df.copy()
    lowered_cols = {c.lower(): c for c in row_df.columns}

    if "split" in lowered_cols:
        split_col = lowered_cols["split"]
        test_mask = row_df[split_col].astype(str).str.lower().eq("test")
        if test_mask.any():
            row_df = row_df.loc[test_mask].copy()

    # If multiple rows still exist, prefer proposed/full/final row when identifiable.
    for key in ["method", "config", "setting", "stage", "final_stage"]:
        if key in lowered_cols:
            col = lowered_cols[key]
            mask = row_df[col].astype(str).str.lower().str.contains(
                "proposed|full|all|finetune|stage2_soh",
                regex=True,
                na=False,
            )
            if mask.any():
                row_df = row_df.loc[mask]
                break

    row = row_df.iloc[0]

    test_cls_acc = _pick_first_available(
        row,
        ["test_cls_acc", "cls_acc", "material_acc", "material_accuracy", "mat_acc"],
    )
    test_soc_medape_raw = _pick_first_available(
        row,
        [
            "test_soc_medape_raw",
            "soc_medape_raw",
            "soc_medape_pct",
            "SOC_MedAPE",
            "soc_medape",
        ],
    )
    test_soh_medape_raw = _pick_first_available(
        row,
        [
            "test_soh_medape_raw",
            "soh_medape_raw",
            "soh_medape_pct",
            "SOH_MedAPE",
            "soh_medape",
        ],
    )

    required = {
        "test_cls_acc": test_cls_acc,
        "test_soc_medape_raw": test_soc_medape_raw,
        "test_soh_medape_raw": test_soh_medape_raw,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise RuntimeError(
            f"Full C-rate reference is missing required columns {missing}.\n"
            f"Used file: {used_path}\n"
            f"Available columns: {list(df.columns)}"
        )

    out = {
        "config": FULL_C_RATE_CONFIG,
        "c_rate_indices": ",".join(map(str, FULL_C_RATE_COMBO)),
        "c_rates": _combo_to_label(FULL_C_RATE_COMBO),
        "num_c_rates": int(len(FULL_C_RATE_COMBO)),
        "reference_source": str(used_path),
        "test_cls_acc": float(test_cls_acc),
        "test_soc_medape_raw": float(test_soc_medape_raw),
        "test_soh_medape_raw": float(test_soh_medape_raw),
    }

    # Preserve other useful metric columns if present.
    optional_columns = [
        "final_stage",
        "best_epoch",
        "best_score",
        "test_soc_rmse",
        "test_soc_mae",
        "test_soc_mape",
        "test_soc_medape",
        "test_soh_rmse",
        "test_soh_mae",
        "test_soh_mape",
        "test_soh_medape",
        "test_soc_rmse_raw",
        "test_soc_mae_raw",
        "test_soc_mape_raw",
        "test_soh_rmse_raw",
        "test_soh_mae_raw",
        "test_soh_mape_raw",
        "n_train",
        "n_val",
        "n_test",
        "n_train_ids",
        "n_val_ids",
        "n_test_ids",
        "num_classes",
        "device",
        "elapsed_sec",
    ]

    for col in optional_columns:
        if col in row.index and pd.notna(row[col]) and col not in out:
            value = row[col]
            try:
                if isinstance(value, str):
                    out[col] = value
                else:
                    out[col] = float(value)
            except Exception:
                out[col] = value

    print("[REFERENCE] Loaded C11_All from main proposed result:")
    print(f"[REFERENCE] {used_path}")
    print(
        "[REFERENCE] "
        f"cls={out['test_cls_acc']:.4f}, "
        f"SOC MedAPE={out['test_soc_medape_raw']:.4f}%, "
        f"SOH MedAPE={out['test_soh_medape_raw']:.4f}%"
    )

    return out


def _add_c_rate_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add relative change columns using C11_All as the reference.
    """
    if summary.empty:
        return summary

    if "C11_All" not in set(summary["config"]):
        return summary

    ref = summary.loc[summary["config"] == "C11_All"].iloc[0]

    ref_acc = float(ref["test_cls_acc"])
    ref_soc_medape = float(ref["test_soc_medape_raw"])
    ref_soh_medape = float(ref["test_soh_medape_raw"])
    ref_soc_medae = float(
        ref["test_soc_medae_raw"]
    )

    ref_soh_medae = float(
        ref["test_soh_medae_raw"]
    )
    summary = summary.copy()

    # material + capacity accuracy
    summary["mat_capacity_acc_pct"] = (
        summary["test_cls_acc"].astype(float)
        * 100.0
    )


    # material-only accuracy
    summary["material_acc_pct"] = (
        summary["test_material_acc"].astype(float)
        * 100.0
    )
    summary["soc_medape_pct"] = summary["test_soc_medape_raw"].astype(float)
    summary["soh_medape_pct"] = summary["test_soh_medape_raw"].astype(float)
    summary["soc_medae_pct"] = (
    summary["test_soc_medae_raw"]
    )

    summary["soh_medae_pct"] = (
        summary["test_soh_medae_raw"]
    )
    summary["mat_acc_change_pp_vs_all"] = (
        summary["test_cls_acc"].astype(float) - ref_acc
    ) * 100.0

    summary["soc_medape_change_pp_vs_all"] = (
        summary["test_soc_medape_raw"].astype(float) - ref_soc_medape
    )

    summary["soh_medape_change_pp_vs_all"] = (
        summary["test_soh_medape_raw"].astype(float) - ref_soh_medape
    )
    summary["soc_medae_change_pp_vs_all"] = (
        summary["test_soc_medae_raw"].astype(float)
        - ref_soc_medae
    )


    summary["soh_medae_change_pp_vs_all"] = (
        summary["test_soh_medae_raw"].astype(float)
        - ref_soh_medae
    )
    summary["relative_input_length_pct"] = (
        summary["num_c_rates"].astype(float) / 5.0 * 100.0
    )

    return summary


def _save_partial_outputs(output_root: Path, rows: List[Dict]) -> None:
    partial = pd.DataFrame(rows)
    partial = _add_c_rate_summary_columns(partial)

    partial.to_csv(
        output_root / "c_rate_sensitivity_partial.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        output_root / "c_rate_sensitivity_partial.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)



# =============================================================================
# Further-style test-only summary helpers
# =============================================================================

N_MC_SOC = 500
N_MC_SOH = 500

SOC_COL = "SOC"
SOH_COL = "SOH"
ID_COL = "ID"
PT_COL = "pulse_ms"


def _torch_load(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _medae(a,b):
    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    return float(np.median(np.abs(a-b)))

def _mae(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def _mape(a, b, eps=1e-8):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs((b - a) / np.maximum(np.abs(a), eps))) * 100.0)


def _medape(a, b, eps=1e-8):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.median(np.abs((b - a) / np.maximum(np.abs(a), eps))) * 100.0)


def _inverse_targets(
    soc_z,
    soh_z,
    soc_norm,
    soh_norm,
    normalize_soc: bool = True,
    zscore_normalize: bool = True,
):
    soc = np.asarray(soc_z, dtype=np.float64)
    soh = np.asarray(soh_z, dtype=np.float64)

    if zscore_normalize:
        soc = soc * float(soc_norm[1]) + float(soc_norm[0])
        soh = soh * float(soh_norm[1]) + float(soh_norm[0])

    if normalize_soc:
        soc = soc * 100.0
    soh = soh * 100.0
    return soc, soh


def _resolve_checkpoint(exp_dir: Path) -> Path:
    run_cfg_path = exp_dir / "run_config.json"
    run_cfg = {}
    if run_cfg_path.exists():
        with open(run_cfg_path, "r", encoding="utf-8") as f:
            run_cfg = json.load(f)

    final_stage = run_cfg.get("final_best_stage", "finetune")

    candidates = [
        exp_dir / "checkpoints" / str(final_stage) / "best.pt",
        exp_dir / "checkpoints" / "finetune" / "best.pt",
        exp_dir / "checkpoints" / "stage2_soh" / "best.pt",
        exp_dir / "checkpoints" / "stage1_soc" / "best.pt",
        exp_dir / "checkpoints" / "single" / "best.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"No checkpoint found under: {exp_dir / 'checkpoints'}")


def _load_run_config(exp_dir: Path) -> dict:
    path = exp_dir / "run_config.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_c_rate_test_context(
    data_root: Path,
    output_root: Path,
    config_name: str,
    c_rate_combo: List[int],
):
    exp_dir = output_root / config_name
    run_cfg = _load_run_config(exp_dir)

    data_root = Path(data_root)

    seed = int(run_cfg.get("seed", 42))
    test_id_frac = float(run_cfg.get("test_id_frac", 0.2))
    test_id_count = int(run_cfg.get("test_id_count", 0))
    batch_size = int(run_cfg.get("batch_size", 128))

    pulse_list = list(map(
        int,
        run_cfg.get(
            "pulse_list",
            [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000],
        ),
    ))

    u_start = int(run_cfg.get("u_start", 1))
    u_end = int(run_cfg.get("u_end", 41))
    drop_first_class = bool(run_cfg.get("drop_first_class", True))

    normalize_soc = bool(run_cfg.get("normalize_soc", True))
    zscore_normalize = bool(run_cfg.get("zscore_normalize", True))
    use_pt_as_feature = bool(run_cfg.get("use_pt_as_feature", True))

    cache_dir = exp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    soc_list = list(range(5, 90, 5))

    common_kwargs = {
        "data_root": str(data_root),
        "pulse_list": pulse_list,
        "u_start": u_start,
        "u_end": u_end,
        "drop_first_class": drop_first_class,
    }

    train_kwargs = {
        **common_kwargs,
        "soc_list": soc_list,
    }

    Xtr_raw, ytr_raw, mtr_raw, _, _ = load_or_build_cache(
        str(cache_dir),
        "raw_train",
        build_train_mix_soc_mix_pt,
        train_kwargs,
    )

    Xte_raw, yte_raw, mte_raw, _, _ = load_or_build_cache(
        str(cache_dir),
        "raw_test",
        build_test_random_mix_pt,
        common_kwargs,
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
        print(f"[NORM] WARNING: recomputed train-only U stats for {config_name}")

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
        print(f"[TARGET NORM] WARNING: recomputed target norm for {config_name}")

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
        c_rate_combo=list(map(int, c_rate_combo)),
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

        # keep label encoder for hierarchical accuracy calculation
        "label_encoder": label_encoder,

        "soc_norm": soc_norm,
        "soh_norm": soh_norm,
        "normalize_soc": normalize_soc,
        "zscore_normalize": zscore_normalize,
        "num_classes": int(len(label_encoder.classes_)),
        "pulse_list": pulse_list,
    }


@torch.no_grad()
def _evaluate_c_rate_config_further(
    data_root: Path,
    output_root: Path,
    config_name: str,
    c_rate_combo: List[int],
) -> Dict:
    print("\n" + "=" * 90)
    print(f"[FURTHER TEST] {config_name}: {c_rate_combo} -> {_combo_to_label(c_rate_combo)}")
    print("=" * 90)

    ctx = _build_c_rate_test_context(
        data_root=data_root,
        output_root=output_root,
        config_name=config_name,
        c_rate_combo=c_rate_combo,
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
    # ==================================================
    # Hierarchical material-only classification accuracy
    # ==================================================

    label_encoder = ctx["label_encoder"]

    true_labels = label_encoder.inverse_transform(
        y_true_all
    )

    pred_labels = label_encoder.inverse_transform(
        y_pred_all
    )


    true_material = np.array(
        [
            str(x).split("_")[0]
            for x in true_labels
        ]
    )

    pred_material = np.array(
        [
            str(x).split("_")[0]
            for x in pred_labels
        ]
    )


    material_only_acc = accuracy_score(
        true_material,
        pred_material
    )
    row = {
        "config": config_name,
        "c_rate_indices": ",".join(map(str, c_rate_combo)),
        "c_rates": _combo_to_label(c_rate_combo),
        "num_c_rates": int(len(c_rate_combo)),
        "n_test": int(len(y_true_all)),
        "test_cls_acc": float(accuracy_score(y_true_all, y_pred_all)),
        "test_material_acc": float(
            material_only_acc
        ),
        "test_soc_mae_raw": _mae(soc_true_all, soc_pred_all),
        "test_soc_rmse_raw": _rmse(soc_true_all, soc_pred_all),
        "test_soc_mape_raw": _mape(soc_true_all, soc_pred_all),
        "test_soc_medape_raw": _medape(soc_true_all, soc_pred_all),
        "test_soh_mae_raw": _mae(soh_true_all, soh_pred_all),
        "test_soh_rmse_raw": _rmse(soh_true_all, soh_pred_all),
        "test_soh_mape_raw": _mape(soh_true_all, soh_pred_all),
        "test_soh_medape_raw": _medape(soh_true_all, soh_pred_all),
        "test_soc_medae_raw": _medae(
            soc_true_all,
            soc_pred_all
        ),
        "test_soh_medae_raw": _medae(
            soh_true_all,
            soh_pred_all
        ),
        "n_mc_soc": N_MC_SOC,
        "n_mc_soh": N_MC_SOH,
        "checkpoint_path": str(ckpt_path),
    }

    print(
        f"[RESULT] {config_name}: "
        f"Material={row['test_material_acc']*100:.2f}%, "
        f"Material+Capacity={row['test_cls_acc']*100:.2f}%, "
        f"SOC MedAE={row['test_soc_medae_raw']:.4f}%, "
        f"SOH MedAE={row['test_soh_medae_raw']:.4f}%"
    )

    return row


def run_summary_only_further(
    data_root: str | Path,
    output_root: str | Path,
    full_reference_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []

    for config_name, c_rate_combo in C_RATE_CONFIGS.items():
        if config_name == FULL_C_RATE_CONFIG:
            continue

        rows.append(
            _evaluate_c_rate_config_further(
                data_root=data_root,
                output_root=output_root,
                config_name=config_name,
                c_rate_combo=c_rate_combo,
            )
        )

    rows.append(_load_full_reference_row(reference_csv=full_reference_csv))

    summary = pd.DataFrame(rows)
    summary = _add_c_rate_summary_columns(summary)

    summary_path = output_root / "c_rate_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    with open(
        output_root / "c_rate_sensitivity_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] {summary_path}")
    print(summary[["config", "material_acc_pct", "mat_capacity_acc_pct", "soc_medae_pct", "soh_medae_pct", "n_test"]])

    return summary


def run_c_rate_sensitivity(
    data_root: str | Path,
    output_root: str | Path,
    smoke: bool = False,
    resume: bool = True,
    full_reference_csv: Optional[str | Path] = None,
    config: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run C-rate sensitivity experiments.

    In full mode:
        - If config is None, C1-C10 are trained and C11_All is loaded as reference.
        - If config is C1-C10, only that one configuration is trained.
        - If config is C11_All, only the existing full-input reference is loaded.
        - C11_All is NOT retrained by this script.

    In smoke mode:
        - Only one lightweight smoke configuration is trained unless config is given.
        - No C11_All reference is appended unless config == C11_All.

    c_rate_combo uses 1-based row indices:
        1 -> 0.5C
        2 -> 1.0C
        3 -> 1.5C
        4 -> 2.0C
        5 -> 2.5C
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if config is not None:
        config = str(config).strip()
        if config not in C_RATE_CONFIGS:
            valid = ", ".join(C_RATE_CONFIGS.keys())
            raise ValueError(f"Unknown config: {config}. Valid configs are: {valid}")

    if smoke:
        if config is None:
            configs = {
                "SMOKE_C3_1p5C": [3],
            }
        elif config == FULL_C_RATE_CONFIG:
            configs = {}
        else:
            configs = {config: C_RATE_CONFIGS[config]}

        run_kwargs = {
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

    else:
        # Do not retrain C11_All here. It will be loaded from main proposed
        # further-analysis results below.
        if config is None:
            configs = {
                key: value
                for key, value in C_RATE_CONFIGS.items()
                if key != FULL_C_RATE_CONFIG
            }
        elif config == FULL_C_RATE_CONFIG:
            configs = {}
        else:
            configs = {config: C_RATE_CONFIGS[config]}

        run_kwargs = {
            "batch_size": 128,
            "max_epochs": 400,
            "early_stopping": False,
            "patience": 20,
            "resume": resume,
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

    rows = []

    for config_name, c_rate_combo in configs.items():
        exp_dir = output_root / config_name

        print("\n" + "=" * 90)
        print(f"[RUN] C-rate configuration: {config_name}")
        print(f"[RUN] C-rate combo: {c_rate_combo} -> {_combo_to_label(c_rate_combo)}")
        print(f"[RUN] Output directory: {exp_dir}")
        print("=" * 90)

        out = run_experiment(
            data_root=str(data_root),
            pulse_list=[1000]
            if smoke
            else [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000],
            c_rate_combo=c_rate_combo,

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

            **run_kwargs,
        )

        row = {
            "config": config_name,
            "c_rate_indices": ",".join(map(str, c_rate_combo)),
            "c_rates": _combo_to_label(c_rate_combo),
            "num_c_rates": int(len(c_rate_combo)),
            **out,
        }

        rows.append(row)
        _save_partial_outputs(output_root, rows)

    if (not smoke and config is None) or config == FULL_C_RATE_CONFIG:
        full_ref_row = _load_full_reference_row(reference_csv=full_reference_csv)
        rows.append(full_ref_row)
        _save_partial_outputs(output_root, rows)

    summary = pd.DataFrame(rows)
    summary = _add_c_rate_summary_columns(summary)

    if config is None:
        summary_csv = output_root / "c_rate_sensitivity_summary.csv"
        summary_json = output_root / "c_rate_sensitivity_summary.json"
    elif config == FULL_C_RATE_CONFIG:
        summary_csv = output_root / config / f"{config}_summary.csv"
        summary_json = output_root / config / f"{config}_summary.json"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        summary_csv = output_root / config / f"{config}_summary.csv"
        summary_json = output_root / config / f"{config}_summary.json"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] {summary_csv}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run C-rate sensitivity experiments one-by-one or all together."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=list(C_RATE_CONFIGS.keys()),
        help=(
            "Run only one C-rate configuration, e.g. C1_0p5C. "
            "If omitted, run all trainable C-rate configurations and append C11_All reference."
        ),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(PROJECT_ROOT / "results" / "measurement_sensitivity" / "c_rate"),
        help="Directory for saving C-rate sensitivity results.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run lightweight smoke-test settings.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resume for full runs.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Do not train. Reload C1-C10 checkpoints and run test-only "
            "further analysis with n_mc=500/500, then save the final summary."
        ),
    )
    parser.add_argument(
        "--full_reference_csv",
        type=str,
        default=None,
        help="Optional CSV file used as the C11_All full-input reference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.summary_only:
        summary = run_summary_only_further(
            data_root=args.data_root,
            output_root=args.output_root,
            full_reference_csv=args.full_reference_csv,
        )

    elif args.config is not None:
        summary = run_c_rate_sensitivity(
            data_root=args.data_root,
            output_root=args.output_root,
            smoke=args.smoke,
            resume=not args.no_resume,
            full_reference_csv=args.full_reference_csv,
            config=args.config,
        )

    else:
        run_c_rate_sensitivity(
            data_root=args.data_root,
            output_root=args.output_root,
            smoke=args.smoke,
            resume=not args.no_resume,
            full_reference_csv=args.full_reference_csv,
            config=None,
        )

        if args.smoke:
            summary = pd.DataFrame()
        else:
            summary = run_summary_only_further(
                data_root=args.data_root,
                output_root=args.output_root,
                full_reference_csv=args.full_reference_csv,
            )

    print("\n[SUMMARY]")
    print(summary)


if __name__ == "__main__":
    main()
