# -*- coding: utf-8 -*-
"""
benchmark/node_benchmark.py

NODE benchmark.

Dependencies:
    - original NODE lib folder/module must be importable as `import lib`
    - qhoptim

Run:
    python benchmark/node_benchmark.py --setting fair
    python benchmark/node_benchmark.py --setting enhanced
    python benchmark/node_benchmark.py --setting both --quick
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

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


DEFAULT_DATA_ROOT = r"F:\OneDrive_Personal\OneDrive\battery\second life battery\data"

BASE_DIR = os.path.join("results", "benchmark")
MODEL_NAME = "node"

PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enhanced / controlled-upstream benchmark level
# Material hint accuracy target: 92.3%
TARGET_MATERIAL_ACC = 0.923

# Pseudo SOC hint RMSE target, in SOC percentage points
TARGET_SOC_RMSE_RAW = 7.75


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def build_node_model(input_dim: int, out_dim: int) -> nn.Module:
    model = nn.Sequential(
        lib.DenseBlock(
            input_dim=int(input_dim),
            layer_dim=128,
            num_layers=3,
            tree_dim=int(out_dim),
            depth=6,
            flatten_output=False,
        ),
        Lambda(lambda x: x.mean(dim=1)),
    )
    return model.to(DEVICE)


def train_node(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    task: str = "reg",
    n_classes: int = 1,
    model_name: str = "model",
    out_dir: str = ".",
    num_epochs: int = 100,
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
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
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
        if task == "clf":
            pred = torch.argmax(output, dim=1).detach().cpu().numpy()
        else:
            pred = output.detach().cpu().numpy().reshape(-1)

    return pred, model


def run_node_fair(
    data_root: str = DEFAULT_DATA_ROOT,
    quick: bool = False,
    use_cache: bool = True,
) -> Dict:
    out_dir = os.path.join(BASE_DIR, MODEL_NAME, "fair")
    ensure_dir(out_dir)

    data = prepare_benchmark_data(
        data_root=data_root,
        pulse_list=PULSE_LIST,
        base_dir=BASE_DIR,
        seed=RANDOM_SEED,
        use_cache=use_cache,
    )

    num_epochs = 3 if quick else 100

    print("[NODE fair] Training material classifier.")
    pred_cls_idx, model_clf = train_node(
        X_train=data.Xtr,
        y_train=data.ytr_cls,
        X_val=data.Xte,
        task="clf",
        n_classes=data.num_classes,
        model_name="node_clf",
        out_dir=out_dir,
        num_epochs=num_epochs,
    )

    print("[NODE fair] Training SOC regressor.")
    pred_soc, model_soc = train_node(
        X_train=data.Xtr,
        y_train=data.mtr["SOC"].to_numpy(dtype=np.float32),
        X_val=data.Xte,
        task="reg",
        model_name="node_soc",
        out_dir=out_dir,
        num_epochs=num_epochs,
    )

    print("[NODE fair] Training SOH regressor.")
    pred_soh, model_soh = train_node(
        X_train=data.Xtr,
        y_train=data.mtr["SOH"].to_numpy(dtype=np.float32),
        X_val=data.Xte,
        task="reg",
        model_name="node_soh",
        out_dir=out_dir,
        num_epochs=num_epochs,
    )

    torch.save(model_clf.state_dict(), os.path.join(out_dir, "node_clf_final.pt"))
    torch.save(model_soc.state_dict(), os.path.join(out_dir, "node_soc_final.pt"))
    torch.save(model_soh.state_dict(), os.path.join(out_dir, "node_soh_final.pt"))

    summary = save_predictions_and_summary(
        out_dir=out_dir,
        model_name=MODEL_NAME,
        setting="fair",
        data=data,
        pred_cls_idx=pred_cls_idx,
        pred_soc=pred_soc,
        pred_soh=pred_soh,
        extra_report={"num_epochs": num_epochs},
    )

    return summary


def run_node_enhanced(
    data_root: str = DEFAULT_DATA_ROOT,
    quick: bool = False,
    use_cache: bool = True,
) -> Dict:
    out_dir = os.path.join(BASE_DIR, MODEL_NAME, "enhanced")
    ensure_dir(out_dir)

    data = prepare_benchmark_data(
        data_root=data_root,
        pulse_list=PULSE_LIST,
        base_dir=BASE_DIR,
        seed=RANDOM_SEED,
        use_cache=use_cache,
    )

    num_epochs = 3 if quick else 100

    print("[NODE enhanced] Training baseline material classifier.")
    pred_cls_idx, model_clf = train_node(
        X_train=data.Xtr,
        y_train=data.ytr_cls,
        X_val=data.Xte,
        task="clf",
        n_classes=data.num_classes,
        model_name="node_clf_baseline",
        out_dir=out_dir,
        num_epochs=num_epochs,
    )

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

    print("[NODE enhanced] Training SOC regressor with material hint.")
    pred_soc, model_soc = train_node(
        X_train=Xtr_soc,
        y_train=soc_tr_true,
        X_val=Xte_soc,
        task="reg",
        model_name="node_soc_enhanced",
        out_dir=out_dir,
        num_epochs=num_epochs,
    )

    print("[NODE enhanced] Training SOH regressor with material hint + pseudo SOC hint.")
    pred_soh, model_soh = train_node(
        X_train=Xtr_soh,
        y_train=data.mtr["SOH"].to_numpy(dtype=np.float32),
        X_val=Xte_soh,
        task="reg",
        model_name="node_soh_enhanced",
        out_dir=out_dir,
        num_epochs=num_epochs,
    )

    torch.save(model_clf.state_dict(), os.path.join(out_dir, "node_clf_baseline_final.pt"))
    torch.save(model_soc.state_dict(), os.path.join(out_dir, "node_soc_enhanced_final.pt"))
    torch.save(model_soh.state_dict(), os.path.join(out_dir, "node_soh_enhanced_final.pt"))

    save_json(hint_report, os.path.join(out_dir, "enhanced_hint_report.json"))

    summary = save_predictions_and_summary(
        out_dir=out_dir,
        model_name=MODEL_NAME,
        setting="enhanced",
        data=data,
        pred_cls_idx=pred_cls_idx,
        pred_soc=pred_soc,
        pred_soh=pred_soh,
        extra_report={
            "num_epochs": num_epochs,
            "enhanced_hint_report": hint_report,
        },
    )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--setting", type=str, default="both", choices=["fair", "enhanced", "both"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    use_cache = not args.no_cache

    if args.setting in ["fair", "both"]:
        run_node_fair(data_root=args.data_root, quick=args.quick, use_cache=use_cache)

    if args.setting in ["enhanced", "both"]:
        run_node_enhanced(data_root=args.data_root, quick=args.quick, use_cache=use_cache)


if __name__ == "__main__":
    main()