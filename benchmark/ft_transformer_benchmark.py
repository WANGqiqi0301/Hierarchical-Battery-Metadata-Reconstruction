# -*- coding: utf-8 -*-
"""
benchmark/ft_transformer_benchmark.py

FT-Transformer benchmark.

Dependencies:
    pip install rtdl-revisiting-models

Run:
    python benchmark/ft_transformer_benchmark.py --setting fair
    python benchmark/ft_transformer_benchmark.py --setting enhanced
    python benchmark/ft_transformer_benchmark.py --setting both --quick
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
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from rtdl_revisiting_models import FTTransformer

from benchmark.common import (
    prepare_benchmark_data,
    save_predictions_and_summary,
    ensure_dir,
    save_json,
)
from benchmark.enhanced_inputs import build_enhanced_inputs


DEFAULT_DATA_ROOT = r"F:\OneDrive_Personal\OneDrive\battery\second life battery\data"

BASE_DIR = os.path.join("results", "benchmark")
MODEL_NAME = "ft_transformer"

PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enhanced / controlled-upstream benchmark level
# Material hint accuracy target: 92.3%
TARGET_MATERIAL_ACC = 0.923

# Pseudo SOC hint RMSE target, in SOC percentage points
TARGET_SOC_RMSE_RAW = 7.75


def build_ft_model(n_cont_features: int, d_out: int) -> FTTransformer:
    model = FTTransformer(
        n_cont_features=int(n_cont_features),
        cat_cardinalities=None,
        d_out=int(d_out),
        n_blocks=3,
        d_block=192,
        attention_n_heads=8,
        attention_dropout=0.1,
        ffn_d_hidden_multiplier=4 / 3,
        ffn_dropout=0.1,
        residual_dropout=0.0,
    )
    return model.to(DEVICE)


def train_ft_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    task: str,
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
        d_out = int(n_classes)
    elif task == "reg":
        y_tr_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE).view(-1, 1)
        d_out = 1
    else:
        raise ValueError(f"Unknown task: {task}")

    model = build_ft_model(n_cont_features=X_train.shape[1], d_out=d_out)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss() if task == "clf" else nn.MSELoss()

    checkpoint_path = os.path.join(out_dir, f"{model_name}_checkpoint.pt")
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"[FT] Resuming from checkpoint: {checkpoint_path}")
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
            output = model(x_cont=batch_x, x_cat=None)
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
                    "n_cont_features": int(X_train.shape[1]),
                },
                checkpoint_path,
            )

    model.eval()
    with torch.no_grad():
        output = model(x_cont=X_va_t, x_cat=None)
        if task == "clf":
            pred = torch.argmax(output, dim=1).detach().cpu().numpy()
        else:
            pred = output.detach().cpu().numpy().reshape(-1)

    return pred, model


def run_ft_transformer_fair(
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

    print("[FT fair] Training material classifier.")
    pred_cls_idx, model_clf = train_ft_transformer(
        X_train=data.Xtr,
        y_train=data.ytr_cls,
        X_val=data.Xte,
        task="clf",
        n_classes=data.num_classes,
        model_name="ft_clf",
        out_dir=out_dir,
        num_epochs=num_epochs,
    )

    print("[FT fair] Training SOC regressor.")
    pred_soc, model_soc = train_ft_transformer(
        X_train=data.Xtr,
        y_train=data.mtr["SOC"].to_numpy(dtype=np.float32),
        X_val=data.Xte,
        task="reg",
        model_name="ft_soc",
        out_dir=out_dir,
        num_epochs=num_epochs,
    )

    print("[FT fair] Training SOH regressor.")
    pred_soh, model_soh = train_ft_transformer(
        X_train=data.Xtr,
        y_train=data.mtr["SOH"].to_numpy(dtype=np.float32),
        X_val=data.Xte,
        task="reg",
        model_name="ft_soh",
        out_dir=out_dir,
        num_epochs=num_epochs,
    )

    torch.save(model_clf.state_dict(), os.path.join(out_dir, "ft_clf_final.pt"))
    torch.save(model_soc.state_dict(), os.path.join(out_dir, "ft_soc_final.pt"))
    torch.save(model_soh.state_dict(), os.path.join(out_dir, "ft_soh_final.pt"))

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


def run_ft_transformer_enhanced(
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

    print("[FT enhanced] Training baseline material classifier.")
    pred_cls_idx, model_clf = train_ft_transformer(
        X_train=data.Xtr,
        y_train=data.ytr_cls,
        X_val=data.Xte,
        task="clf",
        n_classes=data.num_classes,
        model_name="ft_clf_baseline",
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

    print("[FT enhanced] Training SOC regressor with material hint.")
    pred_soc, model_soc = train_ft_transformer(
        X_train=Xtr_soc,
        y_train=soc_tr_true,
        X_val=Xte_soc,
        task="reg",
        model_name="ft_soc_enhanced",
        out_dir=out_dir,
        num_epochs=num_epochs,
    )

    print("[FT enhanced] Training SOH regressor with material hint + pseudo SOC hint.")
    pred_soh, model_soh = train_ft_transformer(
        X_train=Xtr_soh,
        y_train=data.mtr["SOH"].to_numpy(dtype=np.float32),
        X_val=Xte_soh,
        task="reg",
        model_name="ft_soh_enhanced",
        out_dir=out_dir,
        num_epochs=num_epochs,
    )

    torch.save(model_clf.state_dict(), os.path.join(out_dir, "ft_clf_baseline_final.pt"))
    torch.save(model_soc.state_dict(), os.path.join(out_dir, "ft_soc_enhanced_final.pt"))
    torch.save(model_soh.state_dict(), os.path.join(out_dir, "ft_soh_enhanced_final.pt"))

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
        run_ft_transformer_fair(data_root=args.data_root, quick=args.quick, use_cache=use_cache)

    if args.setting in ["enhanced", "both"]:
        run_ft_transformer_enhanced(data_root=args.data_root, quick=args.quick, use_cache=use_cache)


if __name__ == "__main__":
    main()