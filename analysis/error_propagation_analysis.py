# -*- coding: utf-8 -*-
"""
analysis/error_propagation_analysis_exact_e0_e3.py

Exact E0-E3 counterfactual / error-propagation analysis with retrospective metrics.

E3 is evaluated FIRST using the exact inference path from:
    analysis/run_further_analysis_proposed.py

This preserves the validated proposed-method result.

Definitions:
    E0: oracle material    + true SOC
    E1: predicted material + true SOC
    E2: oracle material    + predicted SOC
    E3: predicted material + predicted SOC

SOC pairing:
    E0 SOC == E2 SOC
    E1 SOC == E3 SOC

Important:
    - E3 is computed first via F.infer_rows(), exactly as in further analysis.
    - E1 reuses the exact per-sample E3 SOC predictions.
    - E0/E2 share the same on-site oracle-material SOC predictions.
"""

from __future__ import annotations

import os
import sys
import random
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as TF
from torch.utils.data import DataLoader


# =============================================================================
# Project path
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Reuse further-analysis implementation directly
# =============================================================================
import analysis.run_further_analysis_proposed as F


# =============================================================================
# Output
# =============================================================================
SAVE_DIR = str(
    PROJECT_ROOT
    / "results"
    / "analysis"
    / "error_propagation"
)

COUNTERFACTUAL_SEED = 42


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def onehot_from_y(y_cls: torch.Tensor, num_classes: int) -> torch.Tensor:
    return TF.one_hot(
        y_cls.view(-1),
        num_classes=int(num_classes),
    ).float()


def raw_soc_to_z(
    soc_raw: np.ndarray,
    soc_norm,
) -> np.ndarray:
    soc = np.asarray(soc_raw, dtype=np.float64)

    if F.NORMALIZE_SOC:
        soc = soc / 100.0

    if F.ZSCORE_NORMALIZE:
        soc = (soc - float(soc_norm[0])) / float(soc_norm[1])

    return soc.astype(np.float32)


def inverse_targets(
    soc_z: np.ndarray,
    soh_z: np.ndarray,
    soc_norm,
    soh_norm,
):
    return F.inverse_targets(
        soc_z=soc_z,
        soh_z=soh_z,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
        normalize_soc=F.NORMALIZE_SOC,
        zscore_normalize=F.ZSCORE_NORMALIZE,
    )


def soh_to_percent(soh_raw: np.ndarray) -> np.ndarray:
    """
    Convert SOH from fraction scale to percentage scale for reporting.

    The upstream further-analysis inverse transform returns SOH in the raw
    dataset scale. In this project that scale is fractional SOH, e.g. 0.95.
    This function converts it to percentage units, e.g. 95.
    """
    return np.asarray(soh_raw, dtype=np.float64) * 1.0


def label_to_material(label: str) -> str:
    return str(label).split("_")[0]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_pred - y_true)))


def _medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.median(np.abs(y_pred - y_true)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), float(eps))
    return float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0)


def _medape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), float(eps))
    return float(np.median(np.abs((y_pred - y_true) / denom)) * 100.0)


def standardize_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add material-level columns and error columns.

    SOH is converted from fractional scale to percentage scale here, so all
    SOH absolute-error metrics become percentage-point errors.
    """
    out = df.copy()

    out["true_label"] = out["true_label"].astype(str)
    out["pred_label"] = out["pred_label"].astype(str)

    out["true_material"] = out["true_label"].map(label_to_material)
    out["pred_material"] = out["pred_label"].map(label_to_material)

    out["cls_correct"] = out["true_label"] == out["pred_label"]
    out["material_correct"] = out["true_material"] == out["pred_material"]

    out["soc_true"] = out["soc_true"].astype(float)
    out["soc_pred"] = out["soc_pred"].astype(float)

    already_scaled = (
        "soh_scaled_by" in out.columns
        and np.allclose(
            out["soh_scaled_by"].astype(float).to_numpy(),
            100.0,
        )
    )

    if not already_scaled:
        out["soh_true"] = soh_to_percent(out["soh_true"].to_numpy(dtype=np.float64))
        out["soh_pred"] = soh_to_percent(out["soh_pred"].to_numpy(dtype=np.float64))
        out["soh_scaled_by"] = 100.0
    else:
        out["soh_true"] = out["soh_true"].astype(float)
        out["soh_pred"] = out["soh_pred"].astype(float)

    out["soc_ae"] = np.abs(
        out["soc_pred"].to_numpy(dtype=np.float64)
        - out["soc_true"].to_numpy(dtype=np.float64)
    )
    out["soh_ae"] = np.abs(
        out["soh_pred"].to_numpy(dtype=np.float64)
        - out["soh_true"].to_numpy(dtype=np.float64)
    )

    return out


def summarize_predictions_extended(df: pd.DataFrame, tag: str) -> dict:
    """
    Summarize E0-E3 prediction tables using both class-level and material-level
    classification metrics plus MedAE.
    """
    d = standardize_prediction_df(df)

    soc_true = d["soc_true"].to_numpy(dtype=np.float64)
    soc_pred = d["soc_pred"].to_numpy(dtype=np.float64)
    soh_true = d["soh_true"].to_numpy(dtype=np.float64)
    soh_pred = d["soh_pred"].to_numpy(dtype=np.float64)

    cls_acc = float(d["cls_correct"].mean())
    material_acc = float(d["material_correct"].mean())

    return {
        "experiment": tag,
        "n": int(len(d)),
        "cls_acc": cls_acc,
        "material_acc": material_acc,
        "cls_acc_pct": cls_acc * 100.0,
        "material_acc_pct": material_acc * 100.0,

        "soc_rmse": _rmse(soc_true, soc_pred),
        "soc_mae": _mae(soc_true, soc_pred),
        "soc_medae": _medae(soc_true, soc_pred),
        "soc_mape": _mape(soc_true, soc_pred),
        "soc_medape": _medape(soc_true, soc_pred),

        "soh_rmse": _rmse(soh_true, soh_pred),
        "soh_mae": _mae(soh_true, soh_pred),
        "soh_medae": _medae(soh_true, soh_pred),
        "soh_mape": _mape(soh_true, soh_pred),
        "soh_medape": _medape(soh_true, soh_pred),

        "soc_unit": "percentage point",
        "soh_unit": "percentage point",
        "soh_scaled_by": 100.0,
    }


def build_prediction_df(
    *,
    true_labels: List[str],
    pred_labels: List[str],
    soc_true_raw: np.ndarray,
    soc_pred_raw: np.ndarray,
    soh_true_raw: np.ndarray,
    soh_pred_raw: np.ndarray,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    n = len(soc_true_raw)

    if not (
        len(true_labels) == n
        and len(pred_labels) == n
        and len(soc_pred_raw) == n
        and len(soh_true_raw) == n
        and len(soh_pred_raw) == n
        and len(meta_df) == n
    ):
        raise RuntimeError("Prediction-array lengths are inconsistent.")

    rows = []
    for i in range(n):
        row = {
            "true_label": str(true_labels[i]),
            "pred_label": str(pred_labels[i]),
            "soc_true": float(soc_true_raw[i]),
            "soc_pred": float(soc_pred_raw[i]),
            "soh_true": float(soh_true_raw[i]),
            "soh_pred": float(soh_pred_raw[i]),
        }

        if F.ID_COL in meta_df.columns:
            row["ID"] = str(meta_df.iloc[i][F.ID_COL])

        if F.PT_COL in meta_df.columns:
            row["pulse_ms"] = float(meta_df.iloc[i][F.PT_COL])

        rows.append(row)

    return standardize_prediction_df(pd.DataFrame(rows))


# =============================================================================
# Counterfactual inference
# =============================================================================
@torch.no_grad()
def infer_e0_e1_e2(
    model: torch.nn.Module,
    loader: DataLoader,
    meta_df: pd.DataFrame,
    label_encoder,
    device: str,
    num_classes: int,
    soc_norm,
    soh_norm,
    df_e3: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Build E0/E1/E2 after exact E3 has already been evaluated.

    E1 SOC is copied from the exact E3 prediction table.
    E0/E2 SOC are sampled once under oracle material and shared.
    """

    model.eval()

    # -------------------------------------------------------------------------
    # Common true labels / predicted labels / true targets
    # -------------------------------------------------------------------------
    true_labels = df_e3["true_label"].astype(str).tolist()
    pred_labels = df_e3["pred_label"].astype(str).tolist()

    soc_true_raw = df_e3["soc_true"].to_numpy(dtype=np.float64)
    soh_true_raw = df_e3["soh_true"].to_numpy(dtype=np.float64)

    # Exact proposed/E3 SOC predictions.
    soc_pred_e3_raw = df_e3["soc_pred"].to_numpy(dtype=np.float64)
    soc_pred_e3_z = raw_soc_to_z(
        soc_raw=soc_pred_e3_raw,
        soc_norm=soc_norm,
    )

    # -------------------------------------------------------------------------
    # E0/E2 SOC:
    # oracle material -> SOC
    #
    # Sample once and share exactly between E0 and E2.
    # -------------------------------------------------------------------------
    set_seed(COUNTERFACTUAL_SEED)

    oracle_soc_z_batches = []

    for x3, pt, y_cls, soc_z, soh_z in loader:
        x3 = x3.to(device)
        pt = pt.to(device)
        y_cls = y_cls.to(device).view(-1)

        p_oracle = onehot_from_y(
            y_cls=y_cls,
            num_classes=num_classes,
        ).to(device)

        _, _, soc_pred_oracle_z = F.infer_soc_given_p(
            model=model,
            x3=x3,
            pt=pt,
            p_used=p_oracle,
            n_mc=F.N_MC_SOC,
        )

        oracle_soc_z_batches.append(
            soc_pred_oracle_z.detach().cpu().view(-1).numpy()
        )

    oracle_soc_z = np.concatenate(oracle_soc_z_batches)

    oracle_soc_raw, _ = inverse_targets(
        soc_z=oracle_soc_z,
        soh_z=np.zeros_like(oracle_soc_z),
        soc_norm=soc_norm,
        soh_norm=soh_norm,
    )

    # -------------------------------------------------------------------------
    # E0 SOH:
    # oracle material + true SOC
    # -------------------------------------------------------------------------
    set_seed(COUNTERFACTUAL_SEED)

    e0_soh_z_batches = []

    for x3, pt, y_cls, soc_z, soh_z in loader:
        x3 = x3.to(device)
        pt = pt.to(device)
        y_cls = y_cls.to(device).view(-1)
        soc_z = soc_z.to(device).view(-1)

        p_oracle = onehot_from_y(
            y_cls=y_cls,
            num_classes=num_classes,
        ).to(device)

        _, soh_pred_z = F.infer_soh_given_p_and_soc(
            model=model,
            x3=x3,
            pt=pt,
            p_used=p_oracle,
            soc_val=soc_z,
            n_mc=F.N_MC_SOH,
        )

        e0_soh_z_batches.append(
            soh_pred_z.detach().cpu().view(-1).numpy()
        )

    e0_soh_z = np.concatenate(e0_soh_z_batches)

    _, e0_soh_raw = inverse_targets(
        soc_z=np.zeros_like(e0_soh_z),
        soh_z=e0_soh_z,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
    )

    # -------------------------------------------------------------------------
    # E1 SOH:
    # predicted material + true SOC
    #
    # E1 SOC itself is the exact E3 SOC prediction.
    # -------------------------------------------------------------------------
    set_seed(COUNTERFACTUAL_SEED)

    e1_soh_z_batches = []

    for x3, pt, y_cls, soc_z, soh_z in loader:
        x3 = x3.to(device)
        pt = pt.to(device)
        soc_z = soc_z.to(device).view(-1)

        _, _, p_pred = F.predict_material_prob(
            model=model,
            x3=x3,
            pt=pt,
        )

        _, soh_pred_z = F.infer_soh_given_p_and_soc(
            model=model,
            x3=x3,
            pt=pt,
            p_used=p_pred,
            soc_val=soc_z,
            n_mc=F.N_MC_SOH,
        )

        e1_soh_z_batches.append(
            soh_pred_z.detach().cpu().view(-1).numpy()
        )

    e1_soh_z = np.concatenate(e1_soh_z_batches)

    _, e1_soh_raw = inverse_targets(
        soc_z=np.zeros_like(e1_soh_z),
        soh_z=e1_soh_z,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
    )

    # -------------------------------------------------------------------------
    # E2 SOH:
    # oracle material + exact shared oracle-material predicted SOC
    # -------------------------------------------------------------------------
    set_seed(COUNTERFACTUAL_SEED)

    e2_soh_z_batches = []
    idx_base = 0

    for x3, pt, y_cls, soc_z, soh_z in loader:
        batch_size = int(x3.size(0))

        x3 = x3.to(device)
        pt = pt.to(device)
        y_cls = y_cls.to(device).view(-1)

        p_oracle = onehot_from_y(
            y_cls=y_cls,
            num_classes=num_classes,
        ).to(device)

        soc_pred_oracle_batch = torch.as_tensor(
            oracle_soc_z[idx_base: idx_base + batch_size],
            dtype=torch.float32,
            device=device,
        ).view(-1)

        idx_base += batch_size

        _, soh_pred_z = F.infer_soh_given_p_and_soc(
            model=model,
            x3=x3,
            pt=pt,
            p_used=p_oracle,
            soc_val=soc_pred_oracle_batch,
            n_mc=F.N_MC_SOH,
        )

        e2_soh_z_batches.append(
            soh_pred_z.detach().cpu().view(-1).numpy()
        )

    e2_soh_z = np.concatenate(e2_soh_z_batches)

    _, e2_soh_raw = inverse_targets(
        soc_z=np.zeros_like(e2_soh_z),
        soh_z=e2_soh_z,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
    )

    # -------------------------------------------------------------------------
    # Build per-sample tables
    # -------------------------------------------------------------------------
    df_e0 = build_prediction_df(
        true_labels=true_labels,
        pred_labels=true_labels,
        soc_true_raw=soc_true_raw,
        soc_pred_raw=oracle_soc_raw,
        soh_true_raw=soh_true_raw,
        soh_pred_raw=e0_soh_raw,
        meta_df=meta_df,
    )

    df_e1 = build_prediction_df(
        true_labels=true_labels,
        pred_labels=pred_labels,
        soc_true_raw=soc_true_raw,
        soc_pred_raw=soc_pred_e3_raw,
        soh_true_raw=soh_true_raw,
        soh_pred_raw=e1_soh_raw,
        meta_df=meta_df,
    )

    df_e2 = build_prediction_df(
        true_labels=true_labels,
        pred_labels=true_labels,
        soc_true_raw=soc_true_raw,
        soc_pred_raw=oracle_soc_raw,
        soh_true_raw=soh_true_raw,
        soh_pred_raw=e2_soh_raw,
        meta_df=meta_df,
    )

    # -------------------------------------------------------------------------
    # Hard SOC-pair consistency checks
    # -------------------------------------------------------------------------
    if not np.array_equal(
        df_e0["soc_pred"].to_numpy(),
        df_e2["soc_pred"].to_numpy(),
    ):
        raise RuntimeError("SOC consistency failure: E0 SOC != E2 SOC")

    if not np.array_equal(
        df_e1["soc_pred"].to_numpy(),
        df_e3["soc_pred"].to_numpy(),
    ):
        raise RuntimeError("SOC consistency failure: E1 SOC != E3 SOC")

    return {
        "E0": df_e0,
        "E1": df_e1,
        "E2": df_e2,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Compatibility flag. This script is evaluation-only and never trains.",
    )
    _ = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # EXACT further-analysis seed handling
    # -------------------------------------------------------------------------
    np.random.seed(F.SEED)
    torch.manual_seed(F.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(F.SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    # -------------------------------------------------------------------------
    # EXACT further-analysis dataset construction
    # -------------------------------------------------------------------------
    ds_tr, ds_val, ds_te, mtr, mval, mte, info = F.build_datasets()
    exp_dir = info["exp_dir"]

    # -------------------------------------------------------------------------
    # EXACT further-analysis model loading
    # -------------------------------------------------------------------------
    model = F.load_model(
        exp_dir=exp_dir,
        num_classes=info["num_classes"],
        device=device,
        run_cfg=info["run_cfg"],
    )

    # -------------------------------------------------------------------------
    # EXACT further-analysis TEST DataLoader
    # -------------------------------------------------------------------------
    dl_te = DataLoader(
        ds_te,
        batch_size=F.BATCH_SIZE,
        shuffle=False,
        num_workers=F.NUM_WORKERS,
        drop_last=False,
    )

    F.infer_rows.soc_norm = info["soc_norm"]
    F.infer_rows.soh_norm = info["soh_norm"]

    # =========================================================================
    # E3 FIRST.
    #
    # Do not place any stochastic model inference before this line.
    # This is the validated exact further-analysis inference path.
    # =========================================================================
    df_e3 = F.infer_rows(
        model=model,
        loader=dl_te,
        meta_df=mte,
        label_encoder=info["label_encoder"],
        device=device,
    )

    # -------------------------------------------------------------------------
    # E0/E1/E2 only after exact E3 is finished
    # -------------------------------------------------------------------------
    cf = infer_e0_e1_e2(
        model=model,
        loader=dl_te,
        meta_df=mte,
        label_encoder=info["label_encoder"],
        device=device,
        num_classes=info["num_classes"],
        soc_norm=info["soc_norm"],
        soh_norm=info["soh_norm"],
        df_e3=df_e3,
    )

    dfs = {
        "E0": cf["E0"],
        "E1": cf["E1"],
        "E2": cf["E2"],
        "E3": standardize_prediction_df(df_e3),
    }

    # -------------------------------------------------------------------------
    # Summaries
    # -------------------------------------------------------------------------
    summary_rows = []

    for tag in ["E0", "E1", "E2", "E3"]:
        summary = summarize_predictions_extended(
            dfs[tag],
            tag,
        )
        summary_rows.append(summary)

    df_summary = pd.DataFrame(summary_rows)

    # Put experiment first.
    cols = ["experiment"] + [
        c for c in df_summary.columns
        if c != "experiment"
    ]
    df_summary = df_summary[cols]

    # -------------------------------------------------------------------------
    # Save per-sample predictions
    # -------------------------------------------------------------------------
    for tag in ["E0", "E1", "E2", "E3"]:
        path = os.path.join(
            SAVE_DIR,
            f"{tag.lower()}_predictions_per_sample.csv",
        )
        dfs[tag].to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[SAVED] {path}")

    summary_path = os.path.join(
        SAVE_DIR,
        "counterfactual_summary.csv",
    )
    df_summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_json_path = os.path.join(
        SAVE_DIR,
        "counterfactual_summary.json",
    )
    df_summary.to_json(
        summary_json_path,
        orient="records",
        force_ascii=False,
        indent=2,
    )

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------
    key_cols = [
        "experiment",
        "n",
        "cls_acc",
        "material_acc",
        "soc_rmse",
        "soc_mae",
        "soc_medae",
        "soc_mape",
        "soc_medape",
        "soh_rmse",
        "soh_mae",
        "soh_medae",
        "soh_mape",
        "soh_medape",
    ]

    print("\n===== E0-E3 counterfactual summary =====")
    print(
        df_summary[key_cols].to_string(index=False)
    )

    print("\n===== SOC consistency =====")
    print(
        "E0 SOC MedAPE == E2 SOC MedAPE:",
        df_summary.loc[
            df_summary["experiment"] == "E0",
            "soc_medape",
        ].iloc[0],
        "==",
        df_summary.loc[
            df_summary["experiment"] == "E2",
            "soc_medape",
        ].iloc[0],
    )
    print(
        "E1 SOC MedAPE == E3 SOC MedAPE:",
        df_summary.loc[
            df_summary["experiment"] == "E1",
            "soc_medape",
        ].iloc[0],
        "==",
        df_summary.loc[
            df_summary["experiment"] == "E3",
            "soc_medape",
        ].iloc[0],
    )

    print(f"\n[SAVED] {summary_path}")
    print(f"[SAVED] {summary_json_path}")
    print("[DONE] Exact E0-E3 error-propagation analysis finished.")

    return dfs, df_summary


if __name__ == "__main__":
    main()
