# -*- coding: utf-8 -*-
"""
figures/plot_fig3b_material_soc_error.py

Reproduce Figure 3b second panel: SOC median APE heatmap.

This script:
1. Uses an existing per-sample prediction table if available.
2. Otherwise regenerates SOC predictions from raw data and a trained checkpoint.
3. Saves a single complete PNG figure.

The color logic matches the original plotting script:
- cmap: YlGnBu
- vmin: 0
- vmax: 20
- over color: #800000
- grid line width: 3.0
- grid line color: #55555530
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.preprocessing import LabelEncoder


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


DEFAULT_PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

DEFAULT_CONFIG = {
    "seed": 42,
    "test_id_frac": 0.2,
    "test_id_count": 0,
    "u_start": 1,
    "u_end": 41,
    "drop_first_class": True,
    "soc_col": "SOC",
    "soh_col": "SOH",
    "use_pt_as_feature": True,
    "normalize_soc": True,
    "zscore_normalize": True,
    "width": 32,
    "blocks": 4,
    "drop2d": 0.0,
    "head_dropout": 0.2,
    "batch_size": 256,
    "n_mc_test": 32,
    "pulse_list": DEFAULT_PULSE_LIST,

    # Plot style. These values match the original script.
    "figsize": (12, 10),
    "cmap": "YlGnBu",
    "vmin": 0.0,
    "vmax": 20.0,
    "over_color": "#800000",
    "line_width": 3.0,
    "line_color": "#55555530",
}


def log(message: str, verbose: bool = False) -> None:
    if verbose:
        print(message)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch_load(path: str | Path, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        return json.load(f)


def read_csv_robust(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    encodings = ["utf-8-sig", "utf-8", "gbk", "cp1252"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error


def write_lines_utf8(path: str | Path, lines: List[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def read_lines_utf8(path: str | Path) -> List[str]:
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_runtime_config(exp_dir: Path, verbose: bool = False) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    run_cfg = load_json(exp_dir / "run_config.json")

    if run_cfg:
        for key in [
            "seed",
            "test_id_frac",
            "test_id_count",
            "u_start",
            "u_end",
            "drop_first_class",
            "soc_col",
            "soh_col",
            "use_pt_as_feature",
            "normalize_soc",
            "zscore_normalize",
            "width",
            "blocks",
            "drop2d",
            "head_dropout",
            "batch_size",
        ]:
            if key in run_cfg:
                cfg[key] = run_cfg[key]

        if "pulse_list" in run_cfg:
            cfg["pulse_list"] = run_cfg["pulse_list"]

        log("[CONFIG] Loaded run_config.json.", verbose)
    else:
        log("[CONFIG] run_config.json not found. Using default settings.", verbose)

    return cfg


def inverse_targets(
    soc_z: np.ndarray,
    soh_z: np.ndarray,
    soc_norm: Optional[Tuple[float, float]],
    soh_norm: Optional[Tuple[float, float]],
    normalize_soc: bool,
    zscore_normalize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    soc = soc_z.astype(np.float64).copy()
    soh = soh_z.astype(np.float64).copy()

    if zscore_normalize:
        if soc_norm is None or soh_norm is None:
            raise RuntimeError("zscore_normalize=True requires soc_norm and soh_norm.")
        soc = soc * float(soc_norm[1]) + float(soc_norm[0])
        soh = soh * float(soh_norm[1]) + float(soh_norm[0])

    if normalize_soc:
        soc = soc * 100.0

    return soc, soh


def load_u_norm_from_file(exp_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    path = exp_dir / "u41_norm_train_only.npz"
    if not path.exists():
        return None, None
    obj = np.load(path)
    return obj["u_mean"], obj["u_std"]


def normalize_u_features(
    Xtr: np.ndarray,
    Xte: np.ndarray,
    exp_dir: Path,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    u_mean, u_std = load_u_norm_from_file(exp_dir)

    if u_mean is None or u_std is None:
        log("[NORM] Computing U1-U41 normalization from the train split.", verbose)
        u_mean = Xtr.mean(axis=0, keepdims=True)
        u_std = Xtr.std(axis=0, keepdims=True) + 1e-8
        np.savez_compressed(
            exp_dir / "u41_norm_train_only.npz",
            u_mean=u_mean.astype(np.float32),
            u_std=u_std.astype(np.float32),
        )

    Xtr_norm = (Xtr - u_mean) / (u_std + 1e-8)
    Xte_norm = (Xte - u_mean) / (u_std + 1e-8)

    return Xtr_norm, Xte_norm


def load_target_norm_from_file(
    exp_dir: Path,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    path = exp_dir / "target_norm_train_only.npz"
    if not path.exists():
        return None, None

    obj = np.load(path)
    soc_norm = (float(obj["soc_mean"][0]), float(obj["soc_std"][0]))
    soh_norm = (float(obj["soh_mean"][0]), float(obj["soh_std"][0]))
    return soc_norm, soh_norm


def compute_target_norm_from_train_meta(
    mtr: pd.DataFrame,
    soc_col: str,
    soh_col: str,
    normalize_soc: bool,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    soc = mtr[soc_col].astype(float).to_numpy(dtype=np.float64)
    if normalize_soc:
        soc = soc / 100.0

    soh = mtr[soh_col].astype(float).to_numpy(dtype=np.float64)

    soc_norm = (float(soc.mean()), float(soc.std() + 1e-8))
    soh_norm = (float(soh.mean()), float(soh.std() + 1e-8))

    return soc_norm, soh_norm


def build_pt_norm(mtr: pd.DataFrame) -> Tuple[float, float]:
    if "pulse_ms" not in mtr.columns:
        return (0.0, 1.0)

    pt_train_ms = mtr["pulse_ms"].astype(float).to_numpy(dtype=np.float32)
    pt_log = np.log1p(pt_train_ms)

    return float(pt_log.mean()), float(pt_log.std() + 1e-8)


def get_split_name_from_label_mapping(exp_dir: Path) -> Optional[str]:
    label_mapping = load_json(exp_dir / "label_mapping.json")
    return label_mapping.get("split_name", None)


def get_classes_from_label_mapping(exp_dir: Path) -> Optional[List[str]]:
    label_mapping = load_json(exp_dir / "label_mapping.json")
    classes = label_mapping.get("classes", None)
    if classes is None:
        return None
    return [str(x) for x in classes]


def load_raw_data_and_split(
    data_root: str | Path,
    exp_dir: Path,
    cfg: dict,
    verbose: bool = False,
):
    cache_dir = ensure_dir(exp_dir / "cache")

    train_kwargs = {
        "data_root": str(data_root),
        "soc_list": list(range(5, 90, 5)),
        "pulse_list": list(map(int, cfg["pulse_list"])),
        "u_start": int(cfg["u_start"]),
        "u_end": int(cfg["u_end"]),
        "drop_first_class": bool(cfg["drop_first_class"]),
    }

    test_kwargs = {
        "data_root": str(data_root),
        "pulse_list": list(map(int, cfg["pulse_list"])),
        "u_start": int(cfg["u_start"]),
        "u_end": int(cfg["u_end"]),
        "drop_first_class": bool(cfg["drop_first_class"]),
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
        test_kwargs,
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

    split_name = get_split_name_from_label_mapping(exp_dir)
    split_path = exp_dir / "splits" / f"{split_name}.txt" if split_name else None

    if split_path is not None and split_path.exists():
        test_ids = read_lines_utf8(split_path)
    else:
        fallback_split_path = (
            exp_dir / "splits" / f"testIDs_seed{cfg['seed']}_frac{cfg['test_id_frac']}.txt"
        )

        if fallback_split_path.exists():
            test_ids = read_lines_utf8(fallback_split_path)
        else:
            all_ids = pd.concat([mtr_raw["ID"], mte_raw["ID"]], axis=0).astype(str).to_numpy()
            test_ids = pick_test_ids(
                all_ids=all_ids,
                test_id_frac=float(cfg["test_id_frac"]),
                test_id_count=int(cfg["test_id_count"]),
                seed=int(cfg["seed"]),
            )
            write_lines_utf8(fallback_split_path, list(map(str, test_ids)))

    Xtr, ytr_str, mtr, Xte, yte_str, mte = apply_id_split(
        Xtr=Xtr_raw,
        ytr_str=ytr_raw,
        mtr=mtr_raw,
        Xte=Xte_raw,
        yte_str=yte_raw,
        mte=mte_raw,
        test_ids=np.array(test_ids, dtype=object),
    )

    train_classes = set(pd.Series(ytr_str).astype(str).tolist())
    mask_known = np.array([label in train_classes for label in yte_str], dtype=bool)

    if not mask_known.all():
        Xte = Xte[mask_known]
        yte_str = yte_str[mask_known]
        mte = mte.loc[mask_known].reset_index(drop=True)

    log(f"[DATA] Final train samples = {len(ytr_str)}", verbose)
    log(f"[DATA] Final test samples = {len(yte_str)}", verbose)

    return Xtr, ytr_str, mtr, Xte, yte_str, mte


def get_label_encoder(exp_dir: Path, ytr_str: np.ndarray) -> LabelEncoder:
    classes = get_classes_from_label_mapping(exp_dir)
    le = LabelEncoder()

    if classes is not None:
        le.classes_ = np.array(classes, dtype=object)
    else:
        le.fit(pd.Series(ytr_str).astype(str))

    return le


def build_dataset_for_inference(
    X_u: np.ndarray,
    y_cls: np.ndarray,
    meta: pd.DataFrame,
    pt_norm: Optional[Tuple[float, float]],
    cfg: dict,
    soc_norm: Tuple[float, float],
    soh_norm: Tuple[float, float],
):
    return HierPulseDataset(
        X_u=X_u,
        y_cls=y_cls,
        meta=meta,
        soc_col=cfg["soc_col"],
        soh_col=cfg["soh_col"],
        use_pt_as_feature=bool(cfg["use_pt_as_feature"]),
        pt_norm=pt_norm,
        normalize_soc=bool(cfg["normalize_soc"]),
        zscore_normalize=bool(cfg["zscore_normalize"]),
        soc_norm=soc_norm if bool(cfg["zscore_normalize"]) else None,
        soh_norm=soh_norm if bool(cfg["zscore_normalize"]) else None,
        c_rate_combo=None,
    )


def load_model(
    exp_dir: Path,
    cfg: dict,
    num_classes: int,
    device: str,
    verbose: bool = False,
) -> torch.nn.Module:
    model = Hier3HeadModel(
        num_classes=num_classes,
        width=int(cfg["width"]),
        blocks=int(cfg["blocks"]),
        drop2d=float(cfg["drop2d"]),
        use_pt_as_feature=bool(cfg["use_pt_as_feature"]),
        head_dropout=float(cfg["head_dropout"]),
    ).to(device)

    candidate_ckpts = [
        exp_dir / "checkpoints" / "finetune" / "best.pt",
        exp_dir / "checkpoints" / "stage1_soc" / "best.pt",
    ]

    ckpt_path = next((p for p in candidate_ckpts if p.exists()), None)

    if ckpt_path is None:
        raise FileNotFoundError(
            "No checkpoint found. Tried:\n"
            + "\n".join(str(p) for p in candidate_ckpts)
        )

    log(f"[MODEL] Loading checkpoint: {ckpt_path}", verbose)
    checkpoint = _torch_load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


@torch.no_grad()
def predict_soc(
    model: torch.nn.Module,
    dataset,
    device: str,
    cfg: dict,
    soc_norm: Tuple[float, float],
    soh_norm: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
    )

    soc_true_all = []
    soc_pred_all = []

    for x3, pt, _, soc_z, soh_z in loader:
        x3 = x3.to(device)
        pt = pt.to(device)

        _, soc_pred_z, _, _, _, _ = model(
            x_img=x3,
            x_pt=pt,
            soc_tf=None,
            n_mc=int(cfg["n_mc_test"]),
        )

        soc_true_z = soc_z.detach().cpu().numpy().reshape(-1)
        soc_pred_z_np = soc_pred_z.detach().cpu().numpy().reshape(-1)

        soc_true_raw, _ = inverse_targets(
            soc_true_z,
            np.zeros_like(soc_true_z),
            soc_norm=soc_norm,
            soh_norm=soh_norm,
            normalize_soc=bool(cfg["normalize_soc"]),
            zscore_normalize=bool(cfg["zscore_normalize"]),
        )
        soc_pred_raw, _ = inverse_targets(
            soc_pred_z_np,
            np.zeros_like(soc_pred_z_np),
            soc_norm=soc_norm,
            soh_norm=soh_norm,
            normalize_soc=bool(cfg["normalize_soc"]),
            zscore_normalize=bool(cfg["zscore_normalize"]),
        )

        soc_true_all.append(soc_true_raw)
        soc_pred_all.append(soc_pred_raw)

    return np.concatenate(soc_true_all), np.concatenate(soc_pred_all)


def find_existing_prediction_table(
    exp_dir: Path,
    prediction_csv: Path,
) -> Optional[Path]:
    candidates = [
        prediction_csv,
        exp_dir / "further_analysis" / "tables" / "test_predictions_per_sample.csv",
        exp_dir / "metrics" / "test_predictions_per_sample.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def ensure_soc_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "true_label" not in df.columns:
        if "label" in df.columns:
            df["true_label"] = df["label"]
        else:
            raise RuntimeError("Prediction table must contain true_label or label.")

    if "soc_ape_pct" in df.columns:
        df["ape"] = df["soc_ape_pct"].astype(float)
    elif {"soc_true", "soc_pred"}.issubset(df.columns):
        df["ape"] = (
            np.abs((df["soc_true"].astype(float) - df["soc_pred"].astype(float))
                   / (df["soc_true"].astype(float) + 1e-5))
            * 100.0
        )
    else:
        raise RuntimeError("Prediction table must contain soc_ape_pct or soc_true/soc_pred.")

    if "soc_bin" not in df.columns:
        df["soc_bin"] = pd.cut(df["soc_true"].astype(float), bins=10)

    return df


def generate_prediction_table_from_raw(
    data_root: str | Path,
    exp_dir: Path,
    prediction_csv: Path,
    cfg: dict,
    pt_norm_mode: str,
    verbose: bool = False,
) -> pd.DataFrame:
    ensure_dir(prediction_csv.parent)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr, ytr_str, mtr, Xte, yte_str, mte = load_raw_data_and_split(
        data_root=data_root,
        exp_dir=exp_dir,
        cfg=cfg,
        verbose=verbose,
    )

    Xtr_norm, Xte_norm = normalize_u_features(
        Xtr=Xtr,
        Xte=Xte,
        exp_dir=exp_dir,
        verbose=verbose,
    )

    soc_norm, soh_norm = load_target_norm_from_file(exp_dir)

    if soc_norm is None or soh_norm is None:
        soc_norm, soh_norm = compute_target_norm_from_train_meta(
            mtr=mtr,
            soc_col=cfg["soc_col"],
            soh_col=cfg["soh_col"],
            normalize_soc=bool(cfg["normalize_soc"]),
        )

        np.savez_compressed(
            exp_dir / "target_norm_train_only.npz",
            soc_mean=np.array([soc_norm[0]], dtype=np.float32),
            soc_std=np.array([soc_norm[1]], dtype=np.float32),
            soh_mean=np.array([soh_norm[0]], dtype=np.float32),
            soh_std=np.array([soh_norm[1]], dtype=np.float32),
        )

    label_encoder = get_label_encoder(exp_dir, ytr_str)
    yte_cls = label_encoder.transform(pd.Series(yte_str).astype(str))

    if pt_norm_mode == "train":
        pt_norm = build_pt_norm(mtr)
    elif pt_norm_mode == "none":
        pt_norm = None
    else:
        raise ValueError("pt_norm_mode must be 'train' or 'none'.")

    dataset = build_dataset_for_inference(
        X_u=Xte_norm,
        y_cls=yte_cls,
        meta=mte,
        pt_norm=pt_norm,
        cfg=cfg,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
    )

    model = load_model(
        exp_dir=exp_dir,
        cfg=cfg,
        num_classes=len(label_encoder.classes_),
        device=device,
        verbose=verbose,
    )

    soc_true, soc_pred = predict_soc(
        model=model,
        dataset=dataset,
        device=device,
        cfg=cfg,
        soc_norm=soc_norm,
        soh_norm=soh_norm,
    )

    df = pd.DataFrame(
        {
            "true_label": pd.Series(yte_str).astype(str).to_numpy(),
            "soc_true": soc_true,
            "soc_pred": soc_pred,
        }
    )

    if "ID" in mte.columns:
        df.insert(0, "ID", mte["ID"].astype(str).to_numpy())

    if "pulse_ms" in mte.columns:
        df["pulse_ms"] = mte["pulse_ms"].to_numpy()

    if cfg["soh_col"] in mte.columns:
        df["soh_true"] = mte[cfg["soh_col"]].to_numpy()

    df["soc_ape_pct"] = (
        np.abs((df["soc_true"] - df["soc_pred"]) / (df["soc_true"] + 1e-5))
        * 100.0
    )
    df["soc_bin"] = pd.cut(df["soc_true"], bins=10)

    df.to_csv(prediction_csv, index=False, encoding="utf-8-sig")

    return df


def load_or_generate_prediction_table(
    data_root: str | Path,
    exp_dir: Path,
    prediction_csv: Path,
    cfg: dict,
    force_regenerate: bool,
    pt_norm_mode: str,
    verbose: bool = False,
) -> pd.DataFrame:
    existing = None if force_regenerate else find_existing_prediction_table(exp_dir, prediction_csv)

    if existing is not None:
        log(f"[CACHE] Loading prediction table: {existing}", verbose)
        df = read_csv_robust(existing)
        df = ensure_soc_error_columns(df)

        if existing != prediction_csv:
            ensure_dir(prediction_csv.parent)
            df.to_csv(prediction_csv, index=False, encoding="utf-8-sig")

        return df

    df = generate_prediction_table_from_raw(
        data_root=data_root,
        exp_dir=exp_dir,
        prediction_csv=prediction_csv,
        cfg=cfg,
        pt_norm_mode=pt_norm_mode,
        verbose=verbose,
    )

    return ensure_soc_error_columns(df)


def get_bin_start(bin_value) -> float:
    text = str(bin_value)
    try:
        return float(text.strip("()[]").split(",")[0])
    except Exception:
        return 0.0


def build_soc_median_pivot(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index="soc_bin",
        columns="true_label",
        values="ape",
        aggfunc="median",
    )

    sorted_index = sorted(pivot.index, key=get_bin_start, reverse=True)
    pivot = pivot.reindex(sorted_index)

    return pivot


def plot_soc_median_ape(
    df: pd.DataFrame,
    out_path: str | Path,
    cfg: dict,
    dpi: int,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    pivot = build_soc_median_pivot(df)
    mask = pivot.isnull()

    cmap = plt.get_cmap(cfg["cmap"]).copy()
    cmap.set_over(cfg["over_color"])

    fig, ax = plt.subplots(figsize=tuple(cfg["figsize"]))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        mask=mask,
        vmin=float(cfg["vmin"]),
        vmax=float(cfg["vmax"]),
        linewidths=float(cfg["line_width"]),
        linecolor=cfg["line_color"],
        cbar=True,
        cbar_kws={"extend": "max"},
        ax=ax,
    )

    ax.set_title(f"SOC Median APE (%)")
    ax.set_xlabel("Material-capacity group")
    ax.set_ylabel("SOC bin")
    ax.set_ylim(len(pivot), 0)
    ax.tick_params(axis="x", rotation=45)

    for label in ax.get_xticklabels():
        label.set_ha("right")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Figure 3b SOC median APE heatmap."
    )

    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--exp_dir", type=str, default="results/proposed_framework")

    parser.add_argument(
        "--prediction_csv",
        type=str,
        default="results/figures/cache/fig3b_material_soc_error/test_predictions_per_sample.csv",
    )

    parser.add_argument(
        "--fig_out_path",
        type=str,
        default="results/figures/main/fig3b/fig3b_soc_median_ape.png",
    )

    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force_regenerate", action="store_true")
    parser.add_argument("--force_zscore", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_mc_test", type=int, default=None)

    parser.add_argument(
        "--pt_norm_mode",
        type=str,
        default="train",
        choices=["train", "none"],
    )

    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    exp_dir = Path(args.exp_dir)
    prediction_csv = Path(args.prediction_csv)

    cfg = resolve_runtime_config(exp_dir, verbose=bool(args.verbose))

    if args.force_zscore:
        cfg["zscore_normalize"] = True

    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)

    if args.n_mc_test is not None:
        cfg["n_mc_test"] = int(args.n_mc_test)

    set_random_seed(int(cfg["seed"]))

    print("[FIGURE] Figure 3b SOC median APE heatmap")
    print(f"[EXP_DIR] {exp_dir}")
    print(f"[PREDICTION CSV] {prediction_csv}")
    print(f"[FIGURE OUT] {args.fig_out_path}")

    df = load_or_generate_prediction_table(
        data_root=args.data_root,
        exp_dir=exp_dir,
        prediction_csv=prediction_csv,
        cfg=cfg,
        force_regenerate=bool(args.force_regenerate),
        pt_norm_mode=args.pt_norm_mode,
        verbose=bool(args.verbose),
    )

    plot_soc_median_ape(
        df=df,
        out_path=args.fig_out_path,
        cfg=cfg,
        dpi=int(args.dpi),
    )

    print("[DONE] Figure 3b SOC median APE heatmap saved successfully.")


if __name__ == "__main__":
    main()