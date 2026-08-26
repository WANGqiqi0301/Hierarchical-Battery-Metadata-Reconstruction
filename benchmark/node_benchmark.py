# -*- coding: utf-8 -*-
"""
benchmark/node_benchmark.py

NODE benchmark.

Dependencies:
    - original NODE lib folder/module must be importable as `import lib`
    - qhoptim

Training:
    python benchmark/node_benchmark.py --setting fair
    python benchmark/node_benchmark.py --setting enhanced
    python benchmark/node_benchmark.py --setting both --quick

Retrospective evaluation without retraining:
    python benchmark/node_benchmark.py --setting fair --summary-only
    python benchmark/node_benchmark.py --setting enhanced --summary-only
    python benchmark/node_benchmark.py --setting both --summary-only
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import lib
from qhoptim.pyt import QHAdam

from benchmark.common import (
    prepare_benchmark_data,
    save_predictions_and_summary,
    ensure_dir,
    save_json,
)
from benchmark.enhanced_inputs import build_enhanced_inputs


DEFAULT_DATA_ROOT = r"data"
BASE_DIR = os.path.join("results", "benchmark")
MODEL_NAME = "node"

PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_MATERIAL_ACC = 0.91875
TARGET_SOC_RMSE_RAW = 7.864401414996508


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def build_node_model(input_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        lib.DenseBlock(
            input_dim=int(input_dim),
            layer_dim=128,
            num_layers=3,
            tree_dim=int(out_dim),
            depth=6,
            flatten_output=False,
        ),
        Lambda(lambda x: x.mean(dim=1)),
    ).to(DEVICE)


def _torch_load(path: str | Path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def _medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64))))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64))))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs((y_pred - y_true) / np.maximum(np.abs(y_true), 1e-8))) * 100.0)


def _medape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.median(np.abs((y_pred - y_true) / np.maximum(np.abs(y_true), 1e-8))) * 100.0)


def _class_names(data) -> np.ndarray:
    for attr in ("label_encoder", "le"):
        encoder = getattr(data, attr, None)
        if encoder is not None and hasattr(encoder, "inverse_transform"):
            return np.asarray(encoder.inverse_transform(np.arange(int(data.num_classes))), dtype=object)

    for attr in ("class_names", "classes"):
        values = getattr(data, attr, None)
        if values is not None and len(values) == int(data.num_classes):
            return np.asarray(values, dtype=object)

    label_cols = (
        "class_label",
        "label",
        "target",
        "material_capacity",
        "material_capacity_label",
        "battery_type",
        "class_name",
    )
    ytr = np.asarray(data.ytr_cls, dtype=int)

    for col in label_cols:
        if col not in data.mtr.columns:
            continue

        names = []
        valid = True

        for cls_idx in range(int(data.num_classes)):
            values = data.mtr.loc[ytr == cls_idx, col].astype(str)
            if values.empty:
                valid = False
                break
            names.append(values.mode().iloc[0])

        if valid:
            return np.asarray(names, dtype=object)

    return np.asarray([str(i) for i in range(int(data.num_classes))], dtype=object)


def _material_names(data, class_names: np.ndarray) -> np.ndarray:
    material_cols = ("material", "Material", "cathode_material", "chemistry", "Chemistry")
    ytr = np.asarray(data.ytr_cls, dtype=int)

    for col in material_cols:
        if col not in data.mtr.columns:
            continue

        materials = []
        valid = True

        for cls_idx in range(int(data.num_classes)):
            values = data.mtr.loc[ytr == cls_idx, col].astype(str)
            if values.empty:
                valid = False
                break
            materials.append(values.mode().iloc[0])

        if valid:
            return np.asarray(materials, dtype=object)

    return np.asarray([str(name).split("_")[0] for name in class_names], dtype=object)


def _build_extended_outputs(
    out_dir: str | Path,
    setting: str,
    data,
    pred_cls_idx: np.ndarray,
    pred_soc: np.ndarray,
    pred_soh: np.ndarray,
    extra_report: Optional[Dict] = None,
) -> Dict:
    out_dir = Path(out_dir)
    ensure_dir(str(out_dir))

    true_cls = np.asarray(data.yte_cls, dtype=int).reshape(-1)
    pred_cls = np.asarray(pred_cls_idx, dtype=int).reshape(-1)

    soc_true = data.mte["SOC"].to_numpy(dtype=np.float64)
    soc_pred = np.asarray(pred_soc, dtype=np.float64).reshape(-1)

    soh_true = data.mte["SOH"].to_numpy(dtype=np.float64) * 100.0
    soh_pred = np.asarray(pred_soh, dtype=np.float64).reshape(-1) * 100.0

    n = len(true_cls)
    if not all(len(x) == n for x in (pred_cls, soc_true, soc_pred, soh_true, soh_pred)):
        raise RuntimeError(
            "Prediction length mismatch: "
            f"n_cls={n}, pred_cls={len(pred_cls)}, soc_true={len(soc_true)}, "
            f"soc_pred={len(soc_pred)}, soh_true={len(soh_true)}, soh_pred={len(soh_pred)}"
        )

    class_names = _class_names(data)
    material_names = _material_names(data, class_names)

    true_labels = class_names[true_cls]
    pred_labels = class_names[pred_cls]
    true_material = material_names[true_cls]
    pred_material = material_names[pred_cls]

    soc_ae = np.abs(soc_pred - soc_true)
    soh_ae = np.abs(soh_pred - soh_true)
    soc_ape = soc_ae / np.maximum(np.abs(soc_true), 1e-8) * 100.0
    soh_ape = soh_ae / np.maximum(np.abs(soh_true), 1e-8) * 100.0

    pred_df = pd.DataFrame({
        "true_cls_idx": true_cls,
        "pred_cls_idx": pred_cls,
        "true_label": true_labels.astype(str),
        "pred_label": pred_labels.astype(str),
        "true_material": true_material.astype(str),
        "pred_material": pred_material.astype(str),
        "class_correct": true_cls == pred_cls,
        "material_correct": true_material.astype(str) == pred_material.astype(str),
        "soc_true": soc_true,
        "soc_pred": soc_pred,
        "soc_ae": soc_ae,
        "soc_ape": soc_ape,
        "soh_true": soh_true,
        "soh_pred": soh_pred,
        "soh_ae": soh_ae,
        "soh_ape": soh_ape,
    })

    for col in ("ID", "pulse_ms"):
        if col in data.mte.columns:
            pred_df[col] = data.mte[col].reset_index(drop=True).to_numpy()

    pred_df.to_csv(out_dir / "test_predictions_per_sample.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "model": MODEL_NAME,
        "setting": setting,
        "test_cls_acc": float(accuracy_score(true_cls, pred_cls)),
        "test_material_acc": float(accuracy_score(true_material.astype(str), pred_material.astype(str))),
        "test_soc_rmse_raw": _rmse(soc_true, soc_pred),
        "test_soc_mae_raw": _mae(soc_true, soc_pred),
        "test_soc_medae_raw": _medae(soc_true, soc_pred),
        "test_soc_mape_raw": _mape(soc_true, soc_pred),
        "test_soc_medape_raw": _medape(soc_true, soc_pred),
        "test_soh_rmse_raw": _rmse(soh_true, soh_pred),
        "test_soh_mae_raw": _mae(soh_true, soh_pred),
        "test_soh_medae_raw": _medae(soh_true, soh_pred),
        "test_soh_mape_raw": _mape(soh_true, soh_pred),
        "test_soh_medape_raw": _medape(soh_true, soh_pred),
        "n_test": int(n),
        "device": str(DEVICE),
    }

    if extra_report:
        for key, value in extra_report.items():
            metrics[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value

    pd.DataFrame([metrics]).to_csv(out_dir / "final_metrics.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

    print("\n[EXTENDED TEST METRICS]")
    print(f"setting: {setting}")
    print(f"test_cls_acc: {metrics['test_cls_acc']:.6f}")
    print(f"test_material_acc: {metrics['test_material_acc']:.6f}")
    print(f"test_soc_mae_raw: {metrics['test_soc_mae_raw']:.6f}")
    print(f"test_soc_medae_raw: {metrics['test_soc_medae_raw']:.6f}")
    print(f"test_soc_medape_raw: {metrics['test_soc_medape_raw']:.6f}")
    print(f"test_soh_mae_raw: {metrics['test_soh_mae_raw']:.6f}")
    print(f"test_soh_medae_raw: {metrics['test_soh_medae_raw']:.6f}")
    print(f"test_soh_medape_raw: {metrics['test_soh_medape_raw']:.6f}")
    print(f"[SAVED] {out_dir / 'test_predictions_per_sample.csv'}")
    print(f"[SAVED] {out_dir / 'final_metrics.csv'}")
    print(f"[SAVED] {out_dir / 'final_metrics.json'}")

    return metrics


def _merge_summary(summary, extended: Dict) -> Dict:
    if isinstance(summary, dict):
        merged = dict(summary)
        merged.update(extended)
        return merged
    return extended


def _save_and_print_master_summary(rows) -> pd.DataFrame:
    rows = list(rows)
    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)

    preferred_cols = [
        "model",
        "setting",
        "test_cls_acc",
        "test_material_acc",
        "test_soc_rmse_raw",
        "test_soc_mae_raw",
        "test_soc_medae_raw",
        "test_soc_mape_raw",
        "test_soc_medape_raw",
        "test_soh_rmse_raw",
        "test_soh_mae_raw",
        "test_soh_medae_raw",
        "test_soh_mape_raw",
        "test_soh_medape_raw",
        "n_test",
        "device",
        "summary_only",
    ]
    cols = [c for c in preferred_cols if c in summary.columns]
    cols += [c for c in summary.columns if c not in cols]
    summary = summary[cols]

    summary_dir = Path(BASE_DIR) / MODEL_NAME
    ensure_dir(str(summary_dir))

    summary_csv = summary_dir / "node_benchmark_summary.csv"
    summary_json = summary_dir / "node_benchmark_summary.json"

    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, ensure_ascii=False, indent=2, default=str)

    print("\n[NODE BENCHMARK SUMMARY]")
    show_cols = [
        "setting",
        "test_cls_acc",
        "test_material_acc",
        "test_soc_medae_raw",
        "test_soh_medae_raw",
        "test_soc_medape_raw",
        "test_soh_medape_raw",
        "n_test",
    ]
    show_cols = [c for c in show_cols if c in summary.columns]
    print(summary[show_cols].to_string(index=False))
    print(f"\n[SAVED] {summary_csv}")
    print(f"[SAVED] {summary_json}")

    return summary


def _load_node_model(
    out_dir: str | Path,
    model_name: str,
    final_filename: str,
    input_dim: int,
    out_dim: int,
) -> Tuple[nn.Module, str]:
    out_dir = Path(out_dir)
    candidates = [
        out_dir / final_filename,
        out_dir / f"{model_name}_checkpoint.pt",
    ]

    for path in candidates:
        if not path.exists():
            continue

        obj = _torch_load(path)

        if isinstance(obj, dict) and "model_state_dict" in obj:
            state = obj["model_state_dict"]
        elif isinstance(obj, dict) and "state_dict" in obj:
            state = obj["state_dict"]
        else:
            state = obj

        model = build_node_model(input_dim=input_dim, out_dim=out_dim)
        model.load_state_dict(state, strict=True)
        model.eval()

        print(f"[NODE] Loaded existing model: {path}")
        return model, str(path)

    raise FileNotFoundError(
        f"No trained model found for '{model_name}'. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


@torch.no_grad()
def _predict_node(model: nn.Module, X: np.ndarray, task: str, batch_size: int = 2048) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=int(batch_size),
        shuffle=False,
    )

    predictions = []
    model.eval()

    for (batch_x,) in loader:
        output = model(batch_x.to(DEVICE))

        if task == "clf":
            predictions.append(torch.argmax(output, dim=1).cpu().numpy())
        elif task == "reg":
            predictions.append(output.reshape(-1).cpu().numpy())
        else:
            raise ValueError(f"Unknown task: {task}")

    return np.concatenate(predictions)


def train_node(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    task: str = "reg",
    n_classes: int = 1,
    model_name: str = "model",
    out_dir: str = ".",
    num_epochs: int = 200,
    batch_size: int = 256,
) -> Tuple[np.ndarray, nn.Module]:
    ensure_dir(out_dir)

    X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    X_va_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)

    if task == "clf":
        y_tr_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
        out_dim = int(n_classes)
    elif task == "reg":
        y_tr_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE).view(-1, 1)
        out_dim = 1
    else:
        raise ValueError(f"Unknown task: {task}")

    model = build_node_model(input_dim=X_train.shape[1], out_dim=out_dim)

    optimizer = QHAdam(
        model.parameters(),
        lr=1e-3,
        nus=(0.7, 1.0),
        betas=(0.95, 0.998),
    )
    criterion = nn.CrossEntropyLoss() if task == "clf" else nn.MSELoss()

    checkpoint_path = os.path.join(out_dir, f"{model_name}_checkpoint.pt")
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"[NODE] Resuming from checkpoint: {checkpoint_path}")
        ckpt = _torch_load(checkpoint_path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"])

    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=int(batch_size),
        shuffle=True,
    )

    for epoch in range(start_epoch, int(num_epochs)):
        model.train()
        pbar = tqdm(loader, desc=f"[{model_name}] Epoch {epoch + 1:03d}/{num_epochs}")

        for batch_x, batch_y in pbar:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "task": task,
                    "n_classes": int(n_classes),
                    "input_dim": int(X_train.shape[1]),
                },
                checkpoint_path,
            )

    model.eval()

    with torch.no_grad():
        output = model(X_va_t)
        pred = torch.argmax(output, dim=1).cpu().numpy() if task == "clf" else output.cpu().numpy().reshape(-1)

    return pred, model


def _prepare_data(data_root: str, use_cache: bool):
    return prepare_benchmark_data(
        data_root=data_root,
        pulse_list=PULSE_LIST,
        base_dir=BASE_DIR,
        seed=RANDOM_SEED,
        use_cache=use_cache,
    )


def run_node_fair(
    data_root: str = DEFAULT_DATA_ROOT,
    quick: bool = False,
    use_cache: bool = True,
) -> Dict:
    out_dir = os.path.join(BASE_DIR, MODEL_NAME, "fair")
    ensure_dir(out_dir)

    data = _prepare_data(data_root, use_cache)
    num_epochs = 3 if quick else 100

    print("[NODE fair] Training material classifier.")
    pred_cls_idx, model_clf = train_node(
        data.Xtr,
        data.ytr_cls,
        data.Xte,
        "clf",
        data.num_classes,
        "node_clf",
        out_dir,
        num_epochs,
    )

    print("[NODE fair] Training SOC regressor.")
    pred_soc, model_soc = train_node(
        data.Xtr,
        data.mtr["SOC"].to_numpy(dtype=np.float32),
        data.Xte,
        "reg",
        model_name="node_soc",
        out_dir=out_dir,
        num_epochs=num_epochs,
    )

    print("[NODE fair] Training SOH regressor.")
    pred_soh, model_soh = train_node(
        data.Xtr,
        data.mtr["SOH"].to_numpy(dtype=np.float32),
        data.Xte,
        "reg",
        model_name="node_soh",
        out_dir=out_dir,
        num_epochs=num_epochs,
    )

    torch.save(model_clf.state_dict(), os.path.join(out_dir, "node_clf_final.pt"))
    torch.save(model_soc.state_dict(), os.path.join(out_dir, "node_soc_final.pt"))
    torch.save(model_soh.state_dict(), os.path.join(out_dir, "node_soh_final.pt"))

    extra = {"num_epochs": num_epochs, "summary_only": False}

    summary = save_predictions_and_summary(
        out_dir=out_dir,
        model_name=MODEL_NAME,
        setting="fair",
        data=data,
        pred_cls_idx=pred_cls_idx,
        pred_soc=pred_soc,
        pred_soh=pred_soh,
        extra_report=extra,
    )

    extended = _build_extended_outputs(
        out_dir,
        "fair",
        data,
        pred_cls_idx,
        pred_soc,
        pred_soh,
        extra,
    )

    return _merge_summary(summary, extended)


def run_node_fair_summary_only(
    data_root: str = DEFAULT_DATA_ROOT,
    use_cache: bool = True,
) -> Dict:
    out_dir = os.path.join(BASE_DIR, MODEL_NAME, "fair")
    data = _prepare_data(data_root, use_cache)

    model_clf, clf_path = _load_node_model(
        out_dir,
        "node_clf",
        "node_clf_final.pt",
        data.Xtr.shape[1],
        data.num_classes,
    )
    model_soc, soc_path = _load_node_model(
        out_dir,
        "node_soc",
        "node_soc_final.pt",
        data.Xtr.shape[1],
        1,
    )
    model_soh, soh_path = _load_node_model(
        out_dir,
        "node_soh",
        "node_soh_final.pt",
        data.Xtr.shape[1],
        1,
    )

    pred_cls_idx = _predict_node(model_clf, data.Xte, "clf")
    pred_soc = _predict_node(model_soc, data.Xte, "reg")
    pred_soh = _predict_node(model_soh, data.Xte, "reg")

    extra = {
        "summary_only": True,
        "classifier_checkpoint": clf_path,
        "soc_checkpoint": soc_path,
        "soh_checkpoint": soh_path,
    }

    summary = save_predictions_and_summary(
        out_dir=out_dir,
        model_name=MODEL_NAME,
        setting="fair",
        data=data,
        pred_cls_idx=pred_cls_idx,
        pred_soc=pred_soc,
        pred_soh=pred_soh,
        extra_report=extra,
    )

    extended = _build_extended_outputs(
        out_dir,
        "fair",
        data,
        pred_cls_idx,
        pred_soc,
        pred_soh,
        extra,
    )

    return _merge_summary(summary, extended)


def _enhanced_inputs(data):
    soc_tr_true = data.mtr["SOC"].to_numpy(dtype=np.float32)
    soc_te_true = data.mte["SOC"].to_numpy(dtype=np.float32)

    Xtr_soc, Xte_soc, Xtr_soh, Xte_soh, hint_report = build_enhanced_inputs(
        Xtr=data.Xtr,
        Xte=data.Xte,
        ytr_cls=data.ytr_cls,
        yte_cls=data.yte_cls,
        soc_tr_true=soc_tr_true,
        soc_te_true=soc_te_true,
        num_classes=data.num_classes,
        target_material_acc=TARGET_MATERIAL_ACC,
        target_soc_rmse=TARGET_SOC_RMSE_RAW,
        seed=RANDOM_SEED,
    )

    return soc_tr_true, Xtr_soc, Xte_soc, Xtr_soh, Xte_soh, hint_report


def run_node_enhanced(
    data_root: str = DEFAULT_DATA_ROOT,
    quick: bool = False,
    use_cache: bool = True,
) -> Dict:
    out_dir = os.path.join(BASE_DIR, MODEL_NAME, "enhanced")
    ensure_dir(out_dir)

    data = _prepare_data(data_root, use_cache)
    num_epochs = 3 if quick else 100

    print("[NODE enhanced] Training baseline material classifier.")
    pred_cls_idx, model_clf = train_node(
        data.Xtr,
        data.ytr_cls,
        data.Xte,
        "clf",
        data.num_classes,
        "node_clf_baseline",
        out_dir,
        num_epochs,
    )

    soc_tr_true, Xtr_soc, Xte_soc, Xtr_soh, Xte_soh, hint_report = _enhanced_inputs(data)

    print("[NODE enhanced] Training SOC regressor with material hint.")
    pred_soc, model_soc = train_node(
        Xtr_soc,
        soc_tr_true,
        Xte_soc,
        "reg",
        model_name="node_soc_enhanced",
        out_dir=out_dir,
        num_epochs=num_epochs,
    )

    print("[NODE enhanced] Training SOH regressor with material hint + pseudo SOC hint.")
    pred_soh, model_soh = train_node(
        Xtr_soh,
        data.mtr["SOH"].to_numpy(dtype=np.float32),
        Xte_soh,
        "reg",
        model_name="node_soh_enhanced",
        out_dir=out_dir,
        num_epochs=num_epochs,
    )

    torch.save(model_clf.state_dict(), os.path.join(out_dir, "node_clf_baseline_final.pt"))
    torch.save(model_soc.state_dict(), os.path.join(out_dir, "node_soc_enhanced_final.pt"))
    torch.save(model_soh.state_dict(), os.path.join(out_dir, "node_soh_enhanced_final.pt"))

    save_json(hint_report, os.path.join(out_dir, "enhanced_hint_report.json"))

    extra = {
        "num_epochs": num_epochs,
        "summary_only": False,
        "enhanced_hint_report": hint_report,
    }

    summary = save_predictions_and_summary(
        out_dir=out_dir,
        model_name=MODEL_NAME,
        setting="enhanced",
        data=data,
        pred_cls_idx=pred_cls_idx,
        pred_soc=pred_soc,
        pred_soh=pred_soh,
        extra_report=extra,
    )

    extended = _build_extended_outputs(
        out_dir,
        "enhanced",
        data,
        pred_cls_idx,
        pred_soc,
        pred_soh,
        extra,
    )

    return _merge_summary(summary, extended)


def run_node_enhanced_summary_only(
    data_root: str = DEFAULT_DATA_ROOT,
    use_cache: bool = True,
) -> Dict:
    out_dir = os.path.join(BASE_DIR, MODEL_NAME, "enhanced")
    data = _prepare_data(data_root, use_cache)

    _, Xtr_soc, Xte_soc, Xtr_soh, Xte_soh, hint_report = _enhanced_inputs(data)

    model_clf, clf_path = _load_node_model(
        out_dir,
        "node_clf_baseline",
        "node_clf_baseline_final.pt",
        data.Xtr.shape[1],
        data.num_classes,
    )
    model_soc, soc_path = _load_node_model(
        out_dir,
        "node_soc_enhanced",
        "node_soc_enhanced_final.pt",
        Xtr_soc.shape[1],
        1,
    )
    model_soh, soh_path = _load_node_model(
        out_dir,
        "node_soh_enhanced",
        "node_soh_enhanced_final.pt",
        Xtr_soh.shape[1],
        1,
    )

    pred_cls_idx = _predict_node(model_clf, data.Xte, "clf")
    pred_soc = _predict_node(model_soc, Xte_soc, "reg")
    pred_soh = _predict_node(model_soh, Xte_soh, "reg")

    extra = {
        "summary_only": True,
        "classifier_checkpoint": clf_path,
        "soc_checkpoint": soc_path,
        "soh_checkpoint": soh_path,
        "enhanced_hint_report": hint_report,
    }

    summary = save_predictions_and_summary(
        out_dir=out_dir,
        model_name=MODEL_NAME,
        setting="enhanced",
        data=data,
        pred_cls_idx=pred_cls_idx,
        pred_soc=pred_soc,
        pred_soh=pred_soh,
        extra_report=extra,
    )

    extended = _build_extended_outputs(
        out_dir,
        "enhanced",
        data,
        pred_cls_idx,
        pred_soc,
        pred_soh,
        extra,
    )

    return _merge_summary(summary, extended)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--setting", type=str, default="both", choices=["fair", "enhanced", "both"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Load existing trained models and regenerate predictions/metrics without training.",
    )
    args = parser.parse_args()

    use_cache = not args.no_cache

    rows = []

    if args.setting in ("fair", "both"):
        if args.summary_only:
            rows.append(run_node_fair_summary_only(args.data_root, use_cache))
        else:
            rows.append(run_node_fair(args.data_root, args.quick, use_cache))

    if args.setting in ("enhanced", "both"):
        if args.summary_only:
            rows.append(run_node_enhanced_summary_only(args.data_root, use_cache))
        else:
            rows.append(run_node_enhanced(args.data_root, args.quick, use_cache))

    _save_and_print_master_summary(rows)


if __name__ == "__main__":
    main()
