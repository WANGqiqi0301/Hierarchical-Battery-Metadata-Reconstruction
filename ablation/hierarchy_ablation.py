# ablation/hierarchy_ablation.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import sys
import json
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# =============================================================================
# Project path
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Local imports
# =============================================================================

from utils.cache import ensure_dir, load_or_build_cache, drop_nan_inf_rows
from utils.seed import set_random_seed

from proposed_framework.data.build_dataset import (
    build_train_mix_soc_mix_pt,
    build_test_random_mix_pt,
    pick_test_ids,
    apply_id_split,
)
from proposed_framework.data.pulse_dataset import HierPulseDataset
from proposed_framework.models.encoder import MicroResNetEncoder2D3Ch
from proposed_framework.models.conditional_flow import Conditional1DFlow
from proposed_framework.training.trainer import train_one_epoch
from proposed_framework.training.evaluator import eval_one_epoch


# =============================================================================
# Constants
# =============================================================================

DEFAULT_PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

STRUCTURE_MODES: Dict[str, str] = {
    "independent": "Independent parallel prediction: z->M, z->SOC, z->SOH",
    "soc_to_soh": "Partial hierarchy: z->M, z->SOC, z+SOC->SOH",
    "hierarchical": "Full hierarchy: z->M, z+M->SOC, z+M+SOC->SOH",
}


# =============================================================================
# Model
# =============================================================================

class HierarchyAblationModel(nn.Module):
    """
    Hierarchical-structure ablation model.

    struct_mode options
    -------------------
    hierarchical:
        Material:
            z + pt -> material logits
        SOC:
            z + material probability + pt -> SOC flow
        SOH:
            z + material probability + SOC + pt -> SOH flow

    independent:
        Material:
            z + pt -> material logits
        SOC:
            z + pt -> SOC flow
        SOH:
            z + pt -> SOH flow

    soc_to_soh:
        Material:
            z + pt -> material logits
        SOC:
            z + pt -> SOC flow
        SOH:
            z + SOC + pt -> SOH flow
    """

    def __init__(
        self,
        num_classes: int,
        struct_mode: str = "hierarchical",
        width: int = 32,
        blocks: int = 4,
        drop2d: float = 0.0,
        use_pt_as_feature: bool = True,
        soc_hidden: int = 64,
        soh_hidden: int = 64,
        head_dropout: float = 0.2,
        flow_layers: int = 6,
        flow_bins: int = 8,
        flow_tail_bound: float = 3.0,
    ):
        super().__init__()

        if struct_mode not in STRUCTURE_MODES:
            raise ValueError(
                f"Unknown struct_mode={struct_mode}. "
                f"Choose from {list(STRUCTURE_MODES.keys())}."
            )

        self.struct_mode = str(struct_mode)
        self.use_pt = bool(use_pt_as_feature)

        self.encoder = MicroResNetEncoder2D3Ch(
            width=width,
            blocks=blocks,
            drop2d=drop2d,
        )

        pt_dim = 1 if self.use_pt else 0
        p_dim = int(num_classes)

        # Keep material head consistent with the updated proposed model:
        # material classification uses encoder embedding plus optional pulse width.
        mat_input_dim = int(width) + pt_dim

        self.head_mat = nn.Sequential(
            nn.Linear(mat_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(64, num_classes),
        )

        if self.struct_mode == "hierarchical":
            soc_context_dim = int(width) + p_dim + pt_dim
            soh_context_dim = int(width) + p_dim + 1 + pt_dim

        elif self.struct_mode == "independent":
            soc_context_dim = int(width) + pt_dim
            soh_context_dim = int(width) + pt_dim

        elif self.struct_mode == "soc_to_soh":
            soc_context_dim = int(width) + pt_dim
            soh_context_dim = int(width) + 1 + pt_dim

        else:
            raise ValueError(f"Unknown struct_mode={self.struct_mode}.")

        self.soc_flow = Conditional1DFlow(
            context_dim=soc_context_dim,
            hidden_features=int(soc_hidden),
            num_transforms=int(flow_layers),
            num_bins=int(flow_bins),
            tail_bound=float(flow_tail_bound),
        )

        self.soh_flow = Conditional1DFlow(
            context_dim=soh_context_dim,
            hidden_features=int(soh_hidden),
            num_transforms=int(flow_layers),
            num_bins=int(flow_bins),
            tail_bound=float(flow_tail_bound),
        )

    @staticmethod
    def _sample_mean_1d(
        samples: torch.Tensor,
        batch_size: int,
        num_samples: int,
        name: str,
    ) -> torch.Tensor:
        if samples.ndim == 3:
            if samples.shape[0] == int(num_samples) and samples.shape[1] == batch_size:
                return samples.mean(dim=0).squeeze(-1)

            if samples.shape[0] == batch_size and samples.shape[1] == int(num_samples):
                return samples.mean(dim=1).squeeze(-1)

            samples = samples.reshape(int(num_samples), batch_size, 1)
            return samples.mean(dim=0).squeeze(-1)

        if samples.ndim == 2:
            samples = samples.view(int(num_samples), batch_size, 1)
            return samples.mean(dim=0).squeeze(-1)

        raise RuntimeError(f"Unexpected {name} sample shape: {tuple(samples.shape)}")

    def forward(
        self,
        x_img: torch.Tensor,
        x_pt: torch.Tensor,
        soc_tf: Optional[torch.Tensor] = None,
        n_mc: int = 16,
    ):
        z = self.encoder(x_img)
        batch_size = z.size(0)

        if self.use_pt:
            z_mat = torch.cat([z, x_pt], dim=1)
        else:
            z_mat = z

        logits_mat = self.head_mat(z_mat)
        p_mat = torch.softmax(logits_mat, dim=1)

        # ---------------------------------------------------------------------
        # SOC context
        # ---------------------------------------------------------------------

        soc_context_parts = [z]

        if self.struct_mode == "hierarchical":
            soc_context_parts.append(p_mat)

        if self.use_pt:
            soc_context_parts.append(x_pt)

        cond_soc = torch.cat(soc_context_parts, dim=1)

        soc_logp = None

        if soc_tf is not None:
            soc_tf = soc_tf.view(-1)
            soc_logp = self.soc_flow.log_prob(soc_tf, cond_soc)

        with torch.no_grad():
            try:
                soc_samples = self.soc_flow.sample(
                    cond_soc,
                    num_samples=int(n_mc),
                )

                soc_pred = self._sample_mean_1d(
                    samples=soc_samples,
                    batch_size=batch_size,
                    num_samples=int(n_mc),
                    name="SOC",
                )

            except AssertionError:
                if self.struct_mode != "soc_to_soh" or soc_tf is None:
                    raise

                print(
                    "[WARN] soc_to_soh: numerical instability during "
                    "training-time SOC flow sampling; using teacher-forced "
                    "SOC for this forward pass."
                )

                soc_pred = soc_tf.detach().view(-1)

        soc_pred = soc_pred.view(-1)

        if soc_tf is not None:
            soc_value = soc_tf.detach().view(-1, 1)
        else:
            soc_value = soc_pred.detach().view(-1, 1)

        # ---------------------------------------------------------------------
        # SOH context
        # ---------------------------------------------------------------------

        soh_context_parts = [z]

        if self.struct_mode == "hierarchical":
            soh_context_parts.extend([p_mat, soc_value])

        elif self.struct_mode == "soc_to_soh":
            soh_context_parts.append(soc_value)

        elif self.struct_mode == "independent":
            pass

        else:
            raise ValueError(f"Unknown struct_mode={self.struct_mode}.")

        if self.use_pt:
            soh_context_parts.append(x_pt)

        cond_soh = torch.cat(soh_context_parts, dim=1)

        with torch.no_grad():
            soh_samples = self.soh_flow.sample(
                cond_soh,
                num_samples=int(n_mc),
            )

            soh_pred = self._sample_mean_1d(
                samples=soh_samples,
                batch_size=batch_size,
                num_samples=int(n_mc),
                name="SOH",
            )

        soh_pred = soh_pred.view(-1)

        return logits_mat, soc_pred, soc_logp, cond_soc, soh_pred, cond_soh


# =============================================================================
# Helpers
# =============================================================================

def _torch_load(path: str | Path, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _save_json(path: str | Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _stage_score(
    stage: str,
    te: dict,
    alpha_score: float,
) -> float:
    if stage == "stage1_soc":
        return float(te["cls_acc"] - 0.3 * te["soc_rmse"])

    if stage == "stage2_soh":
        return float(-te["soh_rmse"])

    return float(
        te["cls_acc"]
        - float(alpha_score) * (te["soc_rmse"] + te["soh_rmse"])
    )


def _inverse_targets(
    soc_z: np.ndarray,
    soh_z: np.ndarray,
    soc_norm: Optional[Tuple[float, float]],
    soh_norm: Optional[Tuple[float, float]],
    normalize_soc: bool,
    zscore_normalize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert model-space SOC/SOH values back to raw reporting units.
    """
    soc = np.asarray(soc_z, dtype=np.float64)
    soh = np.asarray(soh_z, dtype=np.float64)

    if zscore_normalize:
        if soc_norm is None or soh_norm is None:
            raise RuntimeError(
                "soc_norm and soh_norm are required when zscore_normalize=True."
            )
        soc = soc * float(soc_norm[1]) + float(soc_norm[0])
        soh = soh * float(soh_norm[1]) + float(soh_norm[0])

    if normalize_soc:
        soc = soc * 100.0
    soh = soh * 100.0

    return soc, soh


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _safe_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_pred - y_true)))


def _safe_medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64))))


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0)


def _safe_medape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.median(np.abs((y_pred - y_true) / denom)) * 100.0)


@torch.no_grad()
def eval_one_epoch_hierarchy(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    w_cls: float,
    w_soc: float,
    w_soh: float,
    criterion_cls: nn.Module,
    criterion_reg: nn.Module,
    soc_nll_weight: float,
    soc_norm: Optional[Tuple[float, float]],
    soh_norm: Optional[Tuple[float, float]],
    normalize_soc: bool,
    zscore_normalize: bool,
    n_mc: int = 16,
    use_pred_soc_for_soh: bool = False,
) -> dict:
    """
    Local evaluation helper for hierarchy ablation.

    It keeps the same metric keys as the public evaluator, while allowing the
    final stable test evaluation to use a larger Monte Carlo sample count.
    """
    model.eval()

    total_loss = 0.0
    total_n = 0

    y_true_all = []
    y_pred_all = []
    soc_true_all = []
    soc_pred_all = []
    soh_true_all = []
    soh_pred_all = []

    for x_img, x_pt, y_cls, soc_tf, soh_tf in loader:
        x_img = x_img.to(device)
        x_pt = x_pt.to(device)
        y_cls = y_cls.to(device)
        soc_tf = soc_tf.to(device).view(-1)
        soh_tf = soh_tf.to(device).view(-1)

        # Keep the original evaluation scheme by default. Only the final test
        # for soc_to_soh can opt into true hierarchical inference:
        # predicted SOC -> SOH.
        model_soc_tf = None if use_pred_soc_for_soh else soc_tf

        logits_mat, soc_pred, soc_logp, cond_soc, soh_pred, cond_soh = model(
            x_img=x_img,
            x_pt=x_pt,
            soc_tf=model_soc_tf,
            n_mc=int(n_mc),
        )

        # When predicted SOC is used for the SOH condition, SOC log-probability
        # is computed separately so the evaluation loss keeps the original
        # SOC-NLL term without teacher-forcing the SOH branch.
        if use_pred_soc_for_soh:
            soc_logp = model.soc_flow.log_prob(soc_tf, cond_soc)

        loss = torch.zeros((), dtype=torch.float32, device=device)

        if float(w_cls) > 0:
            loss = loss + float(w_cls) * criterion_cls(logits_mat, y_cls)

        if float(w_soc) > 0:
            soc_reg = criterion_reg(soc_pred.view(-1), soc_tf.view(-1)).mean()
            if soc_logp is not None:
                soc_nll = -soc_logp.mean()
                soc_reg = soc_reg + float(soc_nll_weight) * soc_nll
            loss = loss + float(w_soc) * soc_reg

        if float(w_soh) > 0:
            soh_reg = criterion_reg(soh_pred.view(-1), soh_tf.view(-1)).mean()
            loss = loss + float(w_soh) * soh_reg

        bs = int(y_cls.shape[0])
        total_loss += float(loss.detach().cpu()) * bs
        total_n += bs

        y_true_all.append(y_cls.detach().cpu().numpy())
        y_pred_all.append(torch.argmax(logits_mat, dim=1).detach().cpu().numpy())
        soc_true_all.append(soc_tf.detach().cpu().numpy())
        soc_pred_all.append(soc_pred.detach().cpu().numpy())
        soh_true_all.append(soh_tf.detach().cpu().numpy())
        soh_pred_all.append(soh_pred.detach().cpu().numpy())

    if total_n <= 0:
        raise RuntimeError("Empty dataloader in evaluation.")

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    soc_true = np.concatenate(soc_true_all)
    soc_pred = np.concatenate(soc_pred_all)
    soh_true = np.concatenate(soh_true_all)
    soh_pred = np.concatenate(soh_pred_all)

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

    return {
        "loss": float(total_loss / total_n),
        "cls_acc": float(accuracy_score(y_true, y_pred)),
        "soc_rmse": _safe_rmse(soc_true, soc_pred),
        "soc_mae": _safe_mae(soc_true, soc_pred),
        "soc_mape": _safe_mape(soc_true, soc_pred),
        "soc_medape": _safe_medape(soc_true, soc_pred),
        "soh_rmse": _safe_rmse(soh_true, soh_pred),
        "soh_mae": _safe_mae(soh_true, soh_pred),
        "soh_mape": _safe_mape(soh_true, soh_pred),
        "soh_medape": _safe_medape(soh_true, soh_pred),
        "soc_rmse_raw": _safe_rmse(soc_true_raw, soc_pred_raw),
        "soc_mae_raw": _safe_mae(soc_true_raw, soc_pred_raw),
        "soc_mape_raw": _safe_mape(soc_true_raw, soc_pred_raw),
        "soc_medape_raw": _safe_medape(soc_true_raw, soc_pred_raw),
        "soh_rmse_raw": _safe_rmse(soh_true_raw, soh_pred_raw),
        "soh_mae_raw": _safe_mae(soh_true_raw, soh_pred_raw),
        "soh_mape_raw": _safe_mape(soh_true_raw, soh_pred_raw),
        "soh_medape_raw": _safe_medape(soh_true_raw, soh_pred_raw),
    }


def _first_existing_path(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path is not None and Path(path).exists():
            return Path(path)
    return None


def _find_proposed_summary(proposed_summary_path: Optional[str | Path] = None) -> Path:
    """
    Locate the proposed-method further-analysis summary used as the
    hierarchical reference row.
    """
    candidates = []
    if proposed_summary_path is not None:
        candidates.append(Path(proposed_summary_path))

    candidates.extend(
        [
            PROJECT_ROOT
            / "results"
            / "proposed_framework"
            / "further_analysis"
            / "tables"
            / "proposed_method_summary.csv",
            PROJECT_ROOT
            / "results"
            / "proposed_framework"
            / "further_analysis"
            / "proposed_method_summary.csv",
            PROJECT_ROOT
            / "results"
            / "proposed_framework"
            / "proposed_method_summary.csv",
        ]
    )

    found = _first_existing_path(candidates)
    if found is None:
        msg = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(
            "Could not find proposed further-analysis summary. Checked:\n" + msg
        )
    return found


def _get_metric_value(row: pd.Series, names: List[str], default=np.nan) -> float:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            value = float(row[name])
            if name.endswith("_pct") or name in {"mat_acc_pct", "material_acc_pct"}:
                if "acc" in name:
                    return value / 100.0
            return value
    return float(default)


def load_hierarchical_reference_from_proposed(
    proposed_summary_path: Optional[str | Path] = None,
) -> dict:
    """Load the hierarchical TEST reference from proposed further analysis."""
    path = _find_proposed_summary(proposed_summary_path)
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Proposed summary is empty: {path}")

    if "split" in df.columns:
        test_rows = df.loc[df["split"].astype(str).str.strip().str.lower() == "test"]
        if test_rows.empty:
            raise RuntimeError(f"Could not find split='test' in proposed summary: {path}")
        row = test_rows.iloc[0]
    else:
        row = df.iloc[0]

    cls_acc = _get_metric_value(row, ["test_cls_acc", "cls_acc", "fine_acc"])
    material_acc = _get_metric_value(
        row,
        ["test_material_acc", "material_acc", "material_accuracy", "material_acc_pct"],
        default=cls_acc,
    )
    soc_mae = _get_metric_value(row, ["test_soc_mae_raw", "soc_mae", "soc_mae_raw"])
    soc_medae = _get_metric_value(
        row, ["test_soc_medae_raw", "soc_medae", "soc_ae_p50"], default=np.nan
    )
    soc_rmse = _get_metric_value(row, ["test_soc_rmse_raw", "soc_rmse", "soc_rmse_raw"])
    soc_mape = _get_metric_value(row, ["test_soc_mape_raw", "soc_mape", "soc_mape_raw"])
    soc_medape = _get_metric_value(
        row, ["test_soc_medape_raw", "soc_medape", "soc_medape_raw", "soc_medape_pct"]
    )
    soh_mae = _get_metric_value(row, ["test_soh_mae_raw", "soh_mae", "soh_mae_raw"])
    soh_medae = _get_metric_value(
        row, ["test_soh_medae_raw", "soh_medae", "soh_ae_p50"], default=np.nan
    )
    soh_rmse = _get_metric_value(row, ["test_soh_rmse_raw", "soh_rmse", "soh_rmse_raw"])
    soh_mape = _get_metric_value(row, ["test_soh_mape_raw", "soh_mape", "soh_mape_raw"])
    soh_medape = _get_metric_value(
        row, ["test_soh_medape_raw", "soh_medape", "soh_medape_raw", "soh_medape_pct"]
    )

    out = {
        "struct_mode": "hierarchical",
        "description": STRUCTURE_MODES["hierarchical"],
        "final_stage": str(row.get("final_stage", "proposed_further_analysis")),
        "test_cls_acc": cls_acc,
        "test_material_acc": material_acc,
        "test_soc_mae": soc_mae,
        "test_soc_medae": soc_medae,
        "test_soc_rmse": soc_rmse,
        "test_soc_mape": soc_mape,
        "test_soc_medape": soc_medape,
        "test_soh_mae": soh_mae,
        "test_soh_medae": soh_medae,
        "test_soh_rmse": soh_rmse,
        "test_soh_mape": soh_mape,
        "test_soh_medape": soh_medape,
        "test_soc_mae_raw": soc_mae,
        "test_soc_medae_raw": soc_medae,
        "test_soc_rmse_raw": soc_rmse,
        "test_soc_mape_raw": soc_mape,
        "test_soc_medape_raw": soc_medape,
        "test_soh_mae_raw": soh_mae,
        "test_soh_medae_raw": soh_medae,
        "test_soh_rmse_raw": soh_rmse,
        "test_soh_mape_raw": soh_mape,
        "test_soh_medape_raw": soh_medape,
        "n_train": int(_get_metric_value(row, ["n_train"], default=0)),
        "n_test": int(_get_metric_value(row, ["n_test", "n"], default=0)),
        "num_classes": int(_get_metric_value(row, ["num_classes"], default=8)),
        "device": str(row.get("device", "proposed_further_analysis")),
        "elapsed_sec": float(row.get("elapsed_sec", 0.0)) if "elapsed_sec" in row.index else 0.0,
        "checkpoint_path": str(path),
    }
    print(f"[REFERENCE] Loaded hierarchical TEST row: {path}")
    print(
        f"[REFERENCE] fine={out['test_cls_acc']:.4f}, material={out['test_material_acc']:.4f}, "
        f"SOC MedAE={out['test_soc_medae_raw']:.4f}, SOH MedAE={out['test_soh_medae_raw']:.4f}"
    )
    return out


# =============================================================================
# Retrospective checkpoint evaluation
# =============================================================================

def _resolve_checkpoint(exp_dir: Path) -> Path:
    run_cfg = _load_json(exp_dir / "run_config.json") if '_load_json' in globals() else {}
    final_stage = str(run_cfg.get("final_best_stage", "finetune"))
    candidates = [
        exp_dir / "checkpoints" / final_stage / "best.pt",
        exp_dir / "checkpoints" / "finetune" / "best.pt",
        exp_dir / "checkpoints" / "stage2_soh" / "best.pt",
        exp_dir / "checkpoints" / "stage1_soc" / "best.pt",
        exp_dir / "checkpoints" / "single" / "best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No checkpoint found under {exp_dir / 'checkpoints'}")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_existing_test_context(data_root: Path, exp_dir: Path, struct_mode: str) -> dict:
    run_cfg = _load_json(exp_dir / "run_config.json")
    data_root = Path(data_root)
    saved_root = run_cfg.get("data_root")
    if saved_root and Path(saved_root) != data_root:
        print(f"[PATH] Ignoring saved data_root={saved_root}; using {data_root}")

    pulse_list = list(map(int, run_cfg.get("pulse_list", DEFAULT_PULSE_LIST)))
    seed = int(run_cfg.get("seed", 42))
    test_id_frac = float(run_cfg.get("test_id_frac", 0.2))
    test_id_count = int(run_cfg.get("test_id_count", 0))
    val_id_frac = run_cfg.get("val_id_frac", None)
    val_id_count = int(run_cfg.get("val_id_count", 0))
    val_id_frac_eff = 0.1 if val_id_frac is None and val_id_count <= 0 else float(val_id_frac or 0.0)
    batch_size = int(run_cfg.get("batch_size", 128))
    u_start = int(run_cfg.get("u_start", 1))
    u_end = int(run_cfg.get("u_end", 41))
    drop_first_class = bool(run_cfg.get("drop_first_class", True))
    normalize_soc = bool(run_cfg.get("normalize_soc", True))
    zscore_normalize = bool(run_cfg.get("zscore_normalize", True))
    use_pt_as_feature = bool(run_cfg.get("use_pt_as_feature", True))
    soc_col = str(run_cfg.get("soc_col", "SOC"))
    soh_col = str(run_cfg.get("soh_col", "SOH"))

    cache_dir = exp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    soc_list = list(range(5, 90, 5))
    Xtr_raw, ytr_raw, mtr_raw, _, _ = load_or_build_cache(
        str(cache_dir), "raw_train", build_train_mix_soc_mix_pt,
        {"data_root": str(data_root), "soc_list": soc_list, "pulse_list": pulse_list,
         "u_start": u_start, "u_end": u_end, "drop_first_class": drop_first_class},
    )
    Xte_raw, yte_raw, mte_raw, _, _ = load_or_build_cache(
        str(cache_dir), "raw_test", build_test_random_mix_pt,
        {"data_root": str(data_root), "pulse_list": pulse_list,
         "u_start": u_start, "u_end": u_end, "drop_first_class": drop_first_class},
    )
    Xtr_raw, ytr_raw, mtr_raw = drop_nan_inf_rows(Xtr_raw, ytr_raw, mtr_raw, name="RAW_TRAIN")
    Xte_raw, yte_raw, mte_raw = drop_nan_inf_rows(Xte_raw, yte_raw, mte_raw, name="RAW_TEST")

    all_ids = pd.concat([mtr_raw["ID"], mte_raw["ID"]], axis=0).astype(str).to_numpy()
    test_ids = np.asarray(pick_test_ids(
        all_ids=all_ids, test_id_frac=test_id_frac, test_id_count=test_id_count, seed=seed
    )).astype(str)
    train_candidate_ids = pd.Series(mtr_raw["ID"].astype(str).unique())
    train_candidate_ids = train_candidate_ids.loc[~train_candidate_ids.isin(set(test_ids))].to_numpy()
    val_ids = np.asarray(pick_test_ids(
        all_ids=train_candidate_ids, test_id_frac=val_id_frac_eff,
        test_id_count=val_id_count, seed=seed + 1,
    )).astype(str)

    train_mask = (
        ~mtr_raw["ID"].astype(str).isin(set(test_ids))
        & ~mtr_raw["ID"].astype(str).isin(set(val_ids))
    ).to_numpy()
    test_mask = mte_raw["ID"].astype(str).isin(set(test_ids)).to_numpy()
    Xtr = Xtr_raw[train_mask]
    ytr_str = np.asarray(ytr_raw)[train_mask]
    mtr = mtr_raw.loc[train_mask].reset_index(drop=True)
    Xte = Xte_raw[test_mask]
    yte_str = np.asarray(yte_raw)[test_mask]
    mte = mte_raw.loc[test_mask].reset_index(drop=True)

    norm_path = exp_dir / "u41_norm_train_only.npz"
    if norm_path.exists():
        obj = np.load(norm_path)
        u_mean, u_std = obj["u_mean"], obj["u_std"]
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
        soc_train = mtr[soc_col].astype(float).to_numpy(dtype=np.float64)
        if normalize_soc:
            soc_train /= 100.0
        soh_train = mtr[soh_col].astype(float).to_numpy(dtype=np.float64)
        soc_norm = (float(soc_train.mean()), float(soc_train.std() + 1e-8))
        soh_norm = (float(soh_train.mean()), float(soh_train.std() + 1e-8))

    mapping = _load_json(exp_dir / "label_mapping.json")
    label_encoder = LabelEncoder()
    if mapping.get("classes"):
        label_encoder.classes_ = np.asarray(mapping["classes"], dtype=object)
    else:
        label_encoder.fit(ytr_str)
    known = np.array([label in set(label_encoder.classes_.tolist()) for label in yte_str], dtype=bool)
    if not known.all():
        print(f"[WARN] Removing {int((~known).sum())} test samples with unseen labels.")
        Xte, yte_str = Xte[known], yte_str[known]
        mte = mte.loc[known].reset_index(drop=True)
    yte_cls = label_encoder.transform(yte_str)

    if use_pt_as_feature and "pulse_ms" in mtr.columns:
        pt_log = np.log1p(mtr["pulse_ms"].astype(float).to_numpy(dtype=np.float64))
        pt_norm = (float(pt_log.mean()), float(pt_log.std() + 1e-8))
    else:
        pt_norm = (0.0, 1.0)

    ds_te = HierPulseDataset(
        X_u=Xte, y_cls=yte_cls, meta=mte, soc_col=soc_col, soh_col=soh_col,
        use_pt_as_feature=use_pt_as_feature, pt_norm=pt_norm,
        normalize_soc=normalize_soc, zscore_normalize=zscore_normalize,
        soc_norm=soc_norm if zscore_normalize else None,
        soh_norm=soh_norm if zscore_normalize else None,
    )
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    return {
        "run_cfg": run_cfg, "dl_te": dl_te, "mte": mte, "label_encoder": label_encoder,
        "soc_norm": soc_norm, "soh_norm": soh_norm, "normalize_soc": normalize_soc,
        "zscore_normalize": zscore_normalize, "num_classes": len(label_encoder.classes_),
        "pulse_list": pulse_list, "n_train": len(Xtr), "n_test": len(ds_te),
    }


@torch.no_grad()
def evaluate_existing_hierarchy_checkpoint(
    data_root: str | Path,
    output_root: str | Path,
    struct_mode: str,
) -> dict:
    if struct_mode not in {"independent", "soc_to_soh"}:
        raise ValueError("Existing checkpoint evaluation supports independent or soc_to_soh.")
    data_root, output_root = Path(data_root), Path(output_root)
    exp_dir = output_root / struct_mode
    ctx = _build_existing_test_context(data_root, exp_dir, struct_mode)
    run_cfg = ctx["run_cfg"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HierarchyAblationModel(
        num_classes=int(ctx["num_classes"]), struct_mode=struct_mode,
        width=int(run_cfg.get("width", 32)), blocks=int(run_cfg.get("blocks", 4)),
        drop2d=float(run_cfg.get("drop2d", 0.0)),
        use_pt_as_feature=bool(run_cfg.get("use_pt_as_feature", True)),
        head_dropout=float(run_cfg.get("head_dropout", 0.2)),
    ).to(device)
    ckpt_path = _resolve_checkpoint(exp_dir)
    ckpt = _torch_load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt.get("model_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    n_mc_soc = int(run_cfg.get("final_eval_n_mc_soc", 500))
    n_mc_soh = int(run_cfg.get("final_eval_n_mc_soh", 500))
    n_mc = max(n_mc_soc, n_mc_soh)
    true_cls, pred_cls, soc_true_z, soc_pred_z, soh_true_z, soh_pred_z = [], [], [], [], [], []
    for x_img, x_pt, y_cls, soc_tf, soh_tf in ctx["dl_te"]:
        x_img, x_pt = x_img.to(device), x_pt.to(device)
        logits, soc_pred, _, _, soh_pred, _ = model(
            x_img=x_img, x_pt=x_pt,
            soc_tf=None if struct_mode == "soc_to_soh" else soc_tf.to(device).view(-1),
            n_mc=n_mc,
        )
        true_cls.append(y_cls.numpy())
        pred_cls.append(logits.argmax(1).cpu().numpy())
        soc_true_z.append(soc_tf.numpy())
        soc_pred_z.append(soc_pred.cpu().numpy())
        soh_true_z.append(soh_tf.numpy())
        soh_pred_z.append(soh_pred.cpu().numpy())

    true_cls, pred_cls = np.concatenate(true_cls), np.concatenate(pred_cls)
    soc_true_z, soc_pred_z = np.concatenate(soc_true_z), np.concatenate(soc_pred_z)
    soh_true_z, soh_pred_z = np.concatenate(soh_true_z), np.concatenate(soh_pred_z)
    soc_true, soh_true = _inverse_targets(
        soc_true_z, soh_true_z, ctx["soc_norm"], ctx["soh_norm"],
        ctx["normalize_soc"], ctx["zscore_normalize"],
    )
    soc_pred, soh_pred = _inverse_targets(
        soc_pred_z, soh_pred_z, ctx["soc_norm"], ctx["soh_norm"],
        ctx["normalize_soc"], ctx["zscore_normalize"],
    )
    le = ctx["label_encoder"]
    true_labels, pred_labels = le.inverse_transform(true_cls), le.inverse_transform(pred_cls)
    true_material = np.asarray([x.split("_")[0] for x in true_labels])
    pred_material = np.asarray([x.split("_")[0] for x in pred_labels])

    pred_df = pd.DataFrame({
        "true_label": true_labels, "pred_label": pred_labels,
        "true_material": true_material, "pred_material": pred_material,
        "material_correct": true_material == pred_material,
        "soc_true": soc_true, "soc_pred": soc_pred,
        "soh_true": soh_true, "soh_pred": soh_pred,
    })
    if "ID" in ctx["mte"].columns:
        pred_df.insert(0, "ID", ctx["mte"]["ID"].astype(str).to_numpy())
    metrics_dir = exp_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred_path = metrics_dir / "retrospective_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    out = {
        "struct_mode": struct_mode, "description": STRUCTURE_MODES[struct_mode],
        "final_stage": str(run_cfg.get("final_best_stage", "finetune")),
        "test_cls_acc": float(accuracy_score(true_cls, pred_cls)),
        "test_material_acc": float(np.mean(true_material == pred_material)),
        "test_soc_mae": _safe_mae(soc_true, soc_pred),
        "test_soc_medae": _safe_medae(soc_true, soc_pred),
        "test_soc_rmse": _safe_rmse(soc_true, soc_pred),
        "test_soc_mape": _safe_mape(soc_true, soc_pred),
        "test_soc_medape": _safe_medape(soc_true, soc_pred),
        "test_soh_mae": _safe_mae(soh_true, soh_pred),
        "test_soh_medae": _safe_medae(soh_true, soh_pred),
        "test_soh_rmse": _safe_rmse(soh_true, soh_pred),
        "test_soh_mape": _safe_mape(soh_true, soh_pred),
        "test_soh_medape": _safe_medape(soh_true, soh_pred),
        "test_soc_mae_raw": _safe_mae(soc_true, soc_pred),
        "test_soc_medae_raw": _safe_medae(soc_true, soc_pred),
        "test_soc_rmse_raw": _safe_rmse(soc_true, soc_pred),
        "test_soc_mape_raw": _safe_mape(soc_true, soc_pred),
        "test_soc_medape_raw": _safe_medape(soc_true, soc_pred),
        "test_soh_mae_raw": _safe_mae(soh_true, soh_pred),
        "test_soh_medae_raw": _safe_medae(soh_true, soh_pred),
        "test_soh_rmse_raw": _safe_rmse(soh_true, soh_pred),
        "test_soh_mape_raw": _safe_mape(soh_true, soh_pred),
        "test_soh_medape_raw": _safe_medape(soh_true, soh_pred),
        "n_train": int(ctx["n_train"]), "n_test": int(ctx["n_test"]),
        "num_classes": int(ctx["num_classes"]), "device": device,
        "n_mc_soc": n_mc_soc, "n_mc_soh": n_mc_soh,
        "checkpoint_path": str(ckpt_path), "predictions_path": str(pred_path),
    }
    pd.DataFrame([out]).to_csv(metrics_dir / "retrospective_metrics.csv", index=False, encoding="utf-8-sig")
    _save_json(metrics_dir / "retrospective_metrics.json", out)
    print(
        f"[RESULT] {struct_mode}: fine={out['test_cls_acc']:.4f}, "
        f"material={out['test_material_acc']:.4f}, SOC MedAE={out['test_soc_medae_raw']:.4f}, "
        f"SOH MedAE={out['test_soh_medae_raw']:.4f}"
    )
    return out


def run_summary_only_hierarchy_ablation(
    data_root: str | Path,
    output_root: str | Path,
    selected_config: str = "all",
    proposed_summary_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    modes = ["independent", "soc_to_soh"] if selected_config == "all" else []
    if selected_config in {"independent", "soc_to_soh"}:
        modes = [selected_config]
    for mode in modes:
        rows.append(evaluate_existing_hierarchy_checkpoint(data_root, output_root, mode))
    if selected_config in {"all", "hierarchical"}:
        rows.append(load_hierarchical_reference_from_proposed(proposed_summary_path))
    summary = _add_summary_columns(pd.DataFrame(rows))
    summary.to_csv(output_root / "hierarchy_ablation_summary.csv", index=False, encoding="utf-8-sig")
    _save_json(output_root / "hierarchy_ablation_summary.json", summary.to_dict(orient="records"))
    return summary


# =============================================================================
# One experiment
# =============================================================================

def run_hierarchy_experiment(
    data_root: str | Path,
    pulse_list: List[int],
    struct_mode: str,
    exp_dir: str | Path,
    # U window
    u_start: int = 1,
    u_end: int = 41,
    drop_first_class: bool = True,
    # targets
    soc_col: str = "SOC",
    soh_col: str = "SOH",
    # features
    use_pt_as_feature: bool = True,
    # training
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
    max_epochs: int = 200,
    early_stopping: bool = True,
    patience: int = 20,
    resume: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    # model
    width: int = 32,
    blocks: int = 4,
    drop2d: float = 0.0,
    head_dropout: float = 0.2,
    # losses
    w_cls: float = 1.0,
    w_soc: float = 1.0,
    w_soh: float = 1.0,
    # split
    test_id_frac: float = 0.2,
    test_id_count: int = 0,
    val_id_frac: Optional[float] = None,
    val_id_count: int = 0,
    # target normalization
    normalize_soc: bool = True,
    zscore_normalize: bool = True,
    # two-stage
    two_stage: bool = True,
    stage1_epochs: int = 200,
    stage2_epochs: int = 200,
    finetune_epochs: int = 30,
    freeze_encoder_stage2: bool = True,
    freeze_mat_soc_stage2: bool = True,
    # prior bin weighting
    use_soc_prior_weighting: bool = True,
    use_soh_prior_weighting: bool = True,
    soc_prior_bins: int = 10,
    soh_prior_bins: int = 10,
    soc_prior_low: float = 0.5,
    soc_prior_mid: float = 1.0,
    soc_prior_high: float = 0.8,
    soh_prior_low: float = 0.8,
    soh_prior_mid: float = 1.0,
    soh_prior_high: float = 0.9,
    # scoring
    alpha_score: float = 0.1,
    final_best_stage: str = "finetune",
    final_eval_n_mc_soc: int = 500,
    final_eval_n_mc_soh: int = 500,
) -> dict:
    """
    Train and evaluate one hierarchy-ablation structure.
    """
    if struct_mode not in STRUCTURE_MODES:
        raise ValueError(
            f"Unknown struct_mode={struct_mode}. "
            f"Choose from {list(STRUCTURE_MODES.keys())}."
        )

    start_time = time.time()

    data_root = Path(data_root)
    exp_dir = Path(exp_dir)

    cache_dir = exp_dir / "cache"
    ckpt_dir = exp_dir / "checkpoints"
    logs_dir = exp_dir / "logs"
    splits_dir = exp_dir / "splits"
    metrics_dir = exp_dir / "metrics"

    ensure_dir(
        str(exp_dir),
        str(cache_dir),
        str(ckpt_dir),
        str(logs_dir),
        str(splits_dir),
        str(metrics_dir),
    )

    set_random_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Structure mode: {struct_mode}")
    print(f"[INFO] Meaning: {STRUCTURE_MODES[struct_mode]}")

    run_config = {
        "struct_mode": struct_mode,
        "struct_mode_description": STRUCTURE_MODES[struct_mode],
        "data_root": str(data_root),
        "pulse_list": list(map(int, pulse_list)),
        "u_start": u_start,
        "u_end": u_end,
        "drop_first_class": drop_first_class,
        "soc_col": soc_col,
        "soh_col": soh_col,
        "use_pt_as_feature": use_pt_as_feature,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "max_epochs": max_epochs,
        "early_stopping": early_stopping,
        "patience": patience,
        "resume": resume,
        "num_workers": num_workers,
        "seed": seed,
        "width": width,
        "blocks": blocks,
        "drop2d": drop2d,
        "head_dropout": head_dropout,
        "w_cls": w_cls,
        "w_soc": w_soc,
        "w_soh": w_soh,
        "test_id_frac": test_id_frac,
        "test_id_count": test_id_count,
        "val_id_frac": val_id_frac,
        "val_id_count": val_id_count,
        "normalize_soc": normalize_soc,
        "zscore_normalize": zscore_normalize,
        "two_stage": two_stage,
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "finetune_epochs": finetune_epochs,
        "freeze_encoder_stage2": freeze_encoder_stage2,
        "freeze_mat_soc_stage2": freeze_mat_soc_stage2,
        "use_soc_prior_weighting": use_soc_prior_weighting,
        "use_soh_prior_weighting": use_soh_prior_weighting,
        "soc_prior_bins": soc_prior_bins,
        "soh_prior_bins": soh_prior_bins,
        "soc_prior_low": soc_prior_low,
        "soc_prior_mid": soc_prior_mid,
        "soc_prior_high": soc_prior_high,
        "soh_prior_low": soh_prior_low,
        "soh_prior_mid": soh_prior_mid,
        "soh_prior_high": soh_prior_high,
        "alpha_score": alpha_score,
        "final_best_stage": final_best_stage,
        "final_eval_n_mc_soc": int(final_eval_n_mc_soc),
        "final_eval_n_mc_soh": int(final_eval_n_mc_soh),
        "exp_dir": str(exp_dir),
    }

    _save_json(exp_dir / "run_config.json", run_config)

    # =========================================================================
    # 1. Load raw train/test data
    # =========================================================================

    soc_list = list(range(5, 90, 5))

    train_kwargs = {
        "data_root": str(data_root),
        "soc_list": soc_list,
        "pulse_list": list(map(int, pulse_list)),
        "u_start": u_start,
        "u_end": u_end,
        "drop_first_class": drop_first_class,
    }

    Xtr_raw, ytr_raw, mtr_raw, tag_tr, hit_tr = load_or_build_cache(
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

    Xte_raw, yte_raw, mte_raw, tag_te, hit_te = load_or_build_cache(
        str(cache_dir),
        "raw_test",
        build_test_random_mix_pt,
        test_kwargs,
    )

    print(f"[CACHE] Train tag: {tag_tr} | hit={hit_tr}")
    print(f"[CACHE] Test  tag: {tag_te} | hit={hit_te}")

    if Xtr_raw.shape[1] != 41 or Xte_raw.shape[1] != 41:
        raise ValueError(
            f"Expected X dimension = 41 for U1-U41. "
            f"Got train={Xtr_raw.shape}, test={Xte_raw.shape}."
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

    if len(ytr_raw) == 0 or len(yte_raw) == 0:
        raise RuntimeError("Empty raw train or raw test after loading/cleaning.")

    if "ID" not in mtr_raw.columns or "ID" not in mte_raw.columns:
        raise RuntimeError("Metadata must contain an 'ID' column for group split.")

    for col in (soc_col, soh_col):
        if col not in mtr_raw.columns or col not in mte_raw.columns:
            raise RuntimeError(f"Metadata must contain column '{col}'.")

    # =========================================================================
    # 2. ID-level split
    # =========================================================================

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
    test_ids = np.asarray(test_ids).astype(str)

    if test_id_count and test_id_count > 0:
        split_name = f"testIDs_seed{seed}_n{test_id_count}"
    else:
        split_name = f"testIDs_seed{seed}_frac{test_id_frac}"

    with open(splits_dir / f"{split_name}.txt", "w", encoding="utf-8") as f:
        for test_id in test_ids:
            f.write(str(test_id) + "\n")

    train_candidate_ids = (
        pd.Series(mtr_raw["ID"].astype(str).unique())
        .loc[lambda s: ~s.isin(set(test_ids))]
        .to_numpy()
    )

    if len(train_candidate_ids) == 0:
        raise RuntimeError("No train candidate IDs remain after test split.")

    if val_id_frac is None and not (val_id_count and val_id_count > 0):
        val_id_frac_eff = 0.1
    elif val_id_frac is None:
        val_id_frac_eff = 0.0
    else:
        val_id_frac_eff = float(val_id_frac)

    val_ids = pick_test_ids(
        all_ids=train_candidate_ids,
        test_id_frac=val_id_frac_eff,
        test_id_count=val_id_count,
        seed=seed + 1,
    )
    val_ids = np.asarray(val_ids).astype(str)

    if len(val_ids) == 0:
        raise RuntimeError(
            "Empty validation ID set. Please increase val_id_frac or val_id_count."
        )

    if val_id_count and val_id_count > 0:
        val_split_name = f"valIDs_seed{seed + 1}_n{val_id_count}"
    else:
        val_split_name = f"valIDs_seed{seed + 1}_frac{val_id_frac_eff}"

    with open(splits_dir / f"{val_split_name}.txt", "w", encoding="utf-8") as f:
        for val_id in val_ids:
            f.write(str(val_id) + "\n")

    test_id_set = set(map(str, test_ids))
    val_id_set = set(map(str, val_ids))

    train_mask = (
        ~mtr_raw["ID"].astype(str).isin(test_id_set)
        & ~mtr_raw["ID"].astype(str).isin(val_id_set)
    ).to_numpy()
    val_mask = mte_raw["ID"].astype(str).isin(val_id_set).to_numpy()
    test_mask = mte_raw["ID"].astype(str).isin(test_id_set).to_numpy()

    Xtr = Xtr_raw[train_mask]
    ytr_str = ytr_raw[train_mask]
    mtr = mtr_raw.loc[train_mask].reset_index(drop=True)

    Xval = Xte_raw[val_mask]
    yval_str = yte_raw[val_mask]
    mval = mte_raw.loc[val_mask].reset_index(drop=True)

    Xte = Xte_raw[test_mask]
    yte_str = yte_raw[test_mask]
    mte = mte_raw.loc[test_mask].reset_index(drop=True)

    if len(ytr_str) == 0 or len(yval_str) == 0 or len(yte_str) == 0:
        raise RuntimeError(
            "Empty train, val or test data after applying ID split. "
            f"n_train={len(ytr_str)}, n_val={len(yval_str)}, n_test={len(yte_str)}"
        )

    print(
        f"[DATA] Final TRAIN samples = {len(ytr_str)} | "
        f"unique IDs = {mtr['ID'].astype(str).nunique()}"
    )
    print(
        f"[DATA] Final VAL   samples = {len(yval_str)} | "
        f"unique IDs = {mval['ID'].astype(str).nunique()} "
        f"(from TEST_RANDOM, IDs drawn from training IDs)"
    )
    print(
        f"[DATA] Final TEST  samples = {len(yte_str)} | "
        f"unique IDs = {mte['ID'].astype(str).nunique()}"
    )

    # =========================================================================
    # 3. Train-only normalization of U1-U41
    # =========================================================================

    u_mean = Xtr.mean(axis=0, keepdims=True)
    u_std = Xtr.std(axis=0, keepdims=True) + 1e-8

    Xtr = (Xtr - u_mean) / u_std
    Xval = (Xval - u_mean) / u_std
    Xte = (Xte - u_mean) / u_std

    np.savez_compressed(
        exp_dir / "u41_norm_train_only.npz",
        u_mean=u_mean.astype(np.float32),
        u_std=u_std.astype(np.float32),
    )

    print("[NORM] Applied U1-U41 z-score using TRAIN statistics only.")

    # =========================================================================
    # 4. Target normalization
    # =========================================================================

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

    np.savez_compressed(
        exp_dir / "target_norm_train_only.npz",
        soc_mean=np.array([soc_norm[0]], dtype=np.float32),
        soc_std=np.array([soc_norm[1]], dtype=np.float32),
        soh_mean=np.array([soh_norm[0]], dtype=np.float32),
        soh_std=np.array([soh_norm[1]], dtype=np.float32),
    )

    # =========================================================================
    # 5. Label encoding
    # =========================================================================

    label_encoder = LabelEncoder()
    ytr_cls = label_encoder.fit_transform(ytr_str)

    train_classes = set(label_encoder.classes_.tolist())

    mask_known_val = np.array(
        [label in train_classes for label in yval_str],
        dtype=bool,
    )
    mask_known_test = np.array(
        [label in train_classes for label in yte_str],
        dtype=bool,
    )

    if not mask_known_val.all():
        n_removed = int((~mask_known_val).sum())
        print(
            f"[WARN] Removing {n_removed} validation samples with labels unseen in training."
        )
        Xval = Xval[mask_known_val]
        yval_str = yval_str[mask_known_val]
        mval = mval.loc[mask_known_val].reset_index(drop=True)

    if not mask_known_test.all():
        n_removed = int((~mask_known_test).sum())
        print(
            f"[WARN] Removing {n_removed} test samples with labels unseen in training."
        )
        Xte = Xte[mask_known_test]
        yte_str = yte_str[mask_known_test]
        mte = mte.loc[mask_known_test].reset_index(drop=True)

    if len(yval_str) == 0 or len(yte_str) == 0:
        raise RuntimeError("No validation or test samples remain after filtering unknown labels.")

    yval_cls = label_encoder.transform(yval_str)
    yte_cls = label_encoder.transform(yte_str)

    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)

    _save_json(
        exp_dir / "label_mapping.json",
        {
            "classes": class_names,
            "split_name": split_name,
            "val_split_name": val_split_name,
            "struct_mode": struct_mode,
        },
    )

    print(f"[LABEL] Number of material-capacity classes: {num_classes}")
    print(f"[LABEL] Classes: {class_names}")

    # =========================================================================
    # 6. Pulse-width feature normalization
    # =========================================================================

    if use_pt_as_feature and "pulse_ms" in mtr.columns:
        pt_train_ms = mtr["pulse_ms"].astype(float).to_numpy(dtype=np.float32)
        pt_log = np.log1p(pt_train_ms)

        pt_norm = (
            float(pt_log.mean()),
            float(pt_log.std() + 1e-8),
        )
    else:
        pt_norm = (0.0, 1.0)

    # =========================================================================
    # 7. Dataset and dataloader
    # =========================================================================

    ds_tr = HierPulseDataset(
        X_u=Xtr,
        y_cls=ytr_cls,
        meta=mtr,
        soc_col=soc_col,
        soh_col=soh_col,
        use_pt_as_feature=use_pt_as_feature,
        pt_norm=pt_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
        soc_norm=soc_norm if zscore_normalize else None,
        soh_norm=soh_norm if zscore_normalize else None,
    )

    ds_val = HierPulseDataset(
        X_u=Xval,
        y_cls=yval_cls,
        meta=mval,
        soc_col=soc_col,
        soh_col=soh_col,
        use_pt_as_feature=use_pt_as_feature,
        pt_norm=pt_norm,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
        soc_norm=soc_norm if zscore_normalize else None,
        soh_norm=soh_norm if zscore_normalize else None,
    )

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

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    dl_te = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # =========================================================================
    # 8. Model
    # =========================================================================

    model = HierarchyAblationModel(
        num_classes=num_classes,
        struct_mode=struct_mode,
        width=width,
        blocks=blocks,
        drop2d=drop2d,
        use_pt_as_feature=use_pt_as_feature,
        head_dropout=head_dropout,
    ).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.SmoothL1Loss(beta=1.0, reduction="none")
    soc_nll_weight = 1.0

    # =========================================================================
    # 9. Prior bin weighting
    # =========================================================================

    soc_bin_edges = None
    soc_bin_weights = None

    if use_soc_prior_weighting:
        soc_train_raw = mtr[soc_col].astype(float).to_numpy(dtype=np.float64)

        qs = np.quantile(
            soc_train_raw,
            np.linspace(0, 1, int(soc_prior_bins) + 1),
        )

        soc_bin_edges = [
            (float(qs[i]), float(qs[i + 1]))
            for i in range(len(qs) - 1)
        ]

        soc_bin_weights = np.full(
            (len(soc_bin_edges),),
            float(soc_prior_mid),
            dtype=np.float32,
        )

        if len(soc_bin_weights) >= 1:
            soc_bin_weights[0] = float(soc_prior_low)
            soc_bin_weights[-1] = float(soc_prior_high)

        soc_bin_weights = soc_bin_weights / float(np.mean(soc_bin_weights))

        pd.DataFrame(
            [
                {
                    "lo": lo,
                    "hi": hi,
                    "weight": float(weight),
                }
                for (lo, hi), weight in zip(soc_bin_edges, soc_bin_weights)
            ]
        ).to_csv(
            metrics_dir / "soc_bin_weights_prior.csv",
            index=False,
            encoding="utf-8-sig",
        )

    soh_bin_edges = None
    soh_bin_weights = None

    if use_soh_prior_weighting:
        soh_train_raw = mtr[soh_col].astype(float).to_numpy(dtype=np.float64)

        qs = np.quantile(
            soh_train_raw,
            np.linspace(0, 1, int(soh_prior_bins) + 1),
        )

        soh_bin_edges = [
            (float(qs[i]), float(qs[i + 1]))
            for i in range(len(qs) - 1)
        ]

        soh_bin_weights = np.full(
            (len(soh_bin_edges),),
            float(soh_prior_mid),
            dtype=np.float32,
        )

        if len(soh_bin_weights) >= 1:
            soh_bin_weights[0] = float(soh_prior_low)
            soh_bin_weights[-1] = float(soh_prior_high)

        soh_bin_weights = soh_bin_weights / float(np.mean(soh_bin_weights))

        pd.DataFrame(
            [
                {
                    "lo": lo,
                    "hi": hi,
                    "weight": float(weight),
                }
                for (lo, hi), weight in zip(soh_bin_edges, soh_bin_weights)
            ]
        ).to_csv(
            metrics_dir / "soh_bin_weights_prior.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # =========================================================================
    # 10. Stage helpers
    # =========================================================================

    def set_trainable(stage: str) -> None:
        if stage in {"stage1_soc", "finetune", "single"}:
            for param in model.parameters():
                param.requires_grad = True
            return

        if stage == "stage2_soh":
            for param in model.parameters():
                param.requires_grad = True

            if freeze_encoder_stage2:
                for param in model.encoder.parameters():
                    param.requires_grad = False

            if freeze_mat_soc_stage2:
                for param in model.head_mat.parameters():
                    param.requires_grad = False

                for param in model.soc_flow.parameters():
                    param.requires_grad = False

            for param in model.soh_flow.parameters():
                param.requires_grad = True

            return

        raise ValueError(f"Unknown training stage: {stage}")

    def make_optimizer():
        trainable_params = [
            param for param in model.parameters()
            if param.requires_grad
        ]

        if not trainable_params:
            raise RuntimeError("No trainable parameters found.")

        return torch.optim.AdamW(
            trainable_params,
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

    def stage_paths(stage: str):
        stage_ckpt_dir = ckpt_dir / stage
        ensure_dir(str(stage_ckpt_dir))

        last_path = stage_ckpt_dir / "last.pt"
        best_path = stage_ckpt_dir / "best.pt"
        log_path = logs_dir / f"train_log_{stage}.csv"

        return last_path, best_path, log_path

    def run_stage(
        stage: str,
        epochs: int,
        w_cls_s: float,
        w_soc_s: float,
        w_soh_s: float,
    ) -> Path:
        last_path, best_path, log_path = stage_paths(stage)

        set_trainable(stage)
        optimizer = make_optimizer()

        start_epoch = 0
        best_score = -1e9
        best_epoch = -1
        bad_count = 0

        if resume and last_path.exists():
            ckpt = _torch_load(last_path, map_location=device)

            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optim"])

            start_epoch = int(ckpt.get("epoch", 0) + 1)
            best_score = float(ckpt.get("best_score", -1e9))
            best_epoch = int(ckpt.get("best_epoch", -1))
            bad_count = int(ckpt.get("bad_count", 0))

            print(
                f"[RESUME-{stage}] start_epoch={start_epoch}, "
                f"best_score={best_score:.6f}, best_epoch={best_epoch}"
            )

        for epoch in range(start_epoch, int(epochs)):
            tr = train_one_epoch(
                model=model,
                loader=dl_tr,
                optimizer=optimizer,
                device=device,
                w_cls=w_cls_s,
                w_soc=w_soc_s,
                w_soh=w_soh_s,
                grad_clip=grad_clip,
                criterion_cls=criterion_cls,
                criterion_reg=criterion_reg,
                soc_nll_weight=soc_nll_weight,
                soc_bin_edges=(
                    soc_bin_edges
                    if use_soc_prior_weighting and w_soc_s > 0
                    else None
                ),
                soc_bin_weights=(
                    soc_bin_weights
                    if use_soc_prior_weighting and w_soc_s > 0
                    else None
                ),
                soc_norm=soc_norm if zscore_normalize else None,
                normalize_soc=normalize_soc,
                zscore_normalize=zscore_normalize,
                soh_bin_edges=(
                    soh_bin_edges
                    if use_soh_prior_weighting and w_soh_s > 0
                    else None
                ),
                soh_bin_weights=(
                    soh_bin_weights
                    if use_soh_prior_weighting and w_soh_s > 0
                    else None
                ),
                soh_norm=soh_norm if zscore_normalize else None,
            )

            val = eval_one_epoch_hierarchy(
                model=model,
                loader=dl_val,
                device=device,
                w_cls=w_cls_s,
                w_soc=w_soc_s,
                w_soh=w_soh_s,
                criterion_cls=criterion_cls,
                criterion_reg=criterion_reg,
                soc_nll_weight=soc_nll_weight,
                soc_norm=soc_norm if zscore_normalize else None,
                soh_norm=soh_norm if zscore_normalize else None,
                normalize_soc=normalize_soc,
                zscore_normalize=zscore_normalize,
                n_mc=16,
            )

            score = _stage_score(stage, val, alpha_score=alpha_score)

            row = pd.DataFrame(
                [
                    {
                        "struct_mode": struct_mode,
                        "stage": stage,
                        "epoch": epoch,
                        "train_loss": tr["loss"],
                        "train_cls_acc": tr["cls_acc"],
                        "train_soc_rmse": tr["soc_rmse"],
                        "train_soc_mae": tr["soc_mae"],
                        "train_soc_mape": tr["soc_mape"],
                        "train_soc_medape": tr.get("soc_medape", np.nan),
                        "train_soh_rmse": tr["soh_rmse"],
                        "train_soh_mae": tr["soh_mae"],
                        "train_soh_mape": tr["soh_mape"],
                        "train_soh_medape": tr.get("soh_medape", np.nan),
                        "val_loss": val["loss"],
                        "val_cls_acc": val["cls_acc"],
                        "val_soc_rmse": val["soc_rmse"],
                        "val_soc_mae": val["soc_mae"],
                        "val_soc_mape": val["soc_mape"],
                        "val_soc_medape": val.get("soc_medape", np.nan),
                        "val_soh_rmse": val["soh_rmse"],
                        "val_soh_mae": val["soh_mae"],
                        "val_soh_mape": val["soh_mape"],
                        "val_soh_medape": val.get("soh_medape", np.nan),
                        "val_soc_rmse_raw": val["soc_rmse_raw"],
                        "val_soc_mae_raw": val["soc_mae_raw"],
                        "val_soc_mape_raw": val["soc_mape_raw"],
                        "val_soc_medape_raw": val.get("soc_medape_raw", np.nan),
                        "val_soh_rmse_raw": val["soh_rmse_raw"],
                        "val_soh_mae_raw": val["soh_mae_raw"],
                        "val_soh_mape_raw": val["soh_mape_raw"],
                        "val_soh_medape_raw": val.get("soh_medape_raw", np.nan),
                        "val_score": score,
                        "best_score_so_far": max(best_score, score),
                    }
                ]
            )

            if not log_path.exists():
                row.to_csv(
                    log_path,
                    index=False,
                    encoding="utf-8-sig",
                )
            else:
                row.to_csv(
                    log_path,
                    mode="a",
                    header=False,
                    index=False,
                    encoding="utf-8-sig",
                )

            print(
                f"[{struct_mode} | {stage}] epoch {epoch:03d} | "
                f"VAL cls={val['cls_acc']:.4f} | "
                f"SOC MedAPE(raw)={val.get('soc_medape_raw', np.nan):.3f}% | "
                f"SOH MedAPE(raw)={val.get('soh_medape_raw', np.nan):.3f}% | "
                f"score={score:.6f}"
            )

            improved = score > best_score

            if improved:
                best_score = score
                best_epoch = epoch
                bad_count = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "best_score": best_score,
                        "best_epoch": best_epoch,
                    },
                    best_path,
                )
            else:
                bad_count += 1

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                    "bad_count": bad_count,
                },
                last_path,
            )

            if early_stopping and bad_count >= patience:
                print(
                    f"[EARLY STOP-{struct_mode} | {stage}] "
                    f"best_score={best_score:.6f} at epoch={best_epoch}"
                )
                break

        if best_path.exists():
            ckpt = _torch_load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"])

            print(
                f"[{struct_mode} | {stage}] Loaded BEST checkpoint from "
                f"epoch={ckpt.get('epoch')} | score={ckpt.get('best_score')}"
            )

        return best_path

    # =========================================================================
    # 11. Training
    # =========================================================================

    stage_best_paths = {}

    if two_stage:
        stage_best_paths["stage1_soc"] = run_stage(
            stage="stage1_soc",
            epochs=int(stage1_epochs),
            w_cls_s=float(w_cls),
            w_soc_s=float(w_soc),
            w_soh_s=0.0,
        )

        stage_best_paths["stage2_soh"] = run_stage(
            stage="stage2_soh",
            epochs=int(stage2_epochs),
            w_cls_s=0.0,
            w_soc_s=0.0,
            w_soh_s=float(w_soh),
        )

        if finetune_epochs and finetune_epochs > 0:
            stage_best_paths["finetune"] = run_stage(
                stage="finetune",
                epochs=int(finetune_epochs),
                w_cls_s=float(w_cls) * 0.4,
                w_soc_s=float(w_soc),
                w_soh_s=float(w_soh),
            )
    else:
        stage_best_paths["single"] = run_stage(
            stage="single",
            epochs=int(max_epochs),
            w_cls_s=float(w_cls),
            w_soc_s=float(w_soc),
            w_soh_s=float(w_soh),
        )

    # =========================================================================
    # 12. Final checkpoint selection
    # =========================================================================

    if not two_stage:
        chosen = "single"
    else:
        chosen = final_best_stage

        if chosen == "finetune" and "finetune" not in stage_best_paths:
            chosen = (
                "stage2_soh"
                if "stage2_soh" in stage_best_paths
                else "stage1_soc"
            )

        if chosen not in stage_best_paths:
            chosen = (
                "stage2_soh"
                if "stage2_soh" in stage_best_paths
                else "stage1_soc"
            )

    best_path = stage_best_paths[chosen]

    if best_path.exists():
        ckpt = _torch_load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

        print(
            f"[FINAL] Using BEST checkpoint from stage='{chosen}' | "
            f"epoch={ckpt.get('epoch')} | score={ckpt.get('best_score')}"
        )

    if chosen == "stage1_soc":
        w_cls_eval = float(w_cls)
        w_soc_eval = float(w_soc)
        w_soh_eval = 0.0
    elif chosen == "stage2_soh":
        w_cls_eval = 0.0
        w_soc_eval = 0.0
        w_soh_eval = float(w_soh)
    else:
        w_cls_eval = float(w_cls)
        w_soc_eval = float(w_soc)
        w_soh_eval = float(w_soh)

    # =========================================================================
    # 13. Final evaluation
    # =========================================================================

    final_n_mc = max(int(final_eval_n_mc_soc), int(final_eval_n_mc_soh))

    use_pred_soc_for_soh_final = struct_mode == "soc_to_soh"

    final_scheme = (
        "predicted_SOC_to_SOH"
        if use_pred_soc_for_soh_final
        else "original_scheme"
    )

    print(
        f"[FINAL] Stable test evaluation with "
        f"n_mc_soc={int(final_eval_n_mc_soc)}, "
        f"n_mc_soh={int(final_eval_n_mc_soh)} | "
        f"scheme={final_scheme}"
    )

    te = eval_one_epoch_hierarchy(
        model=model,
        loader=dl_te,
        device=device,
        w_cls=w_cls_eval,
        w_soc=w_soc_eval,
        w_soh=w_soh_eval,
        criterion_cls=criterion_cls,
        criterion_reg=criterion_reg,
        soc_nll_weight=1.0,
        soc_norm=soc_norm if zscore_normalize else None,
        soh_norm=soh_norm if zscore_normalize else None,
        normalize_soc=normalize_soc,
        zscore_normalize=zscore_normalize,
        n_mc=final_n_mc,
        use_pred_soc_for_soh=use_pred_soc_for_soh_final,
    )

    elapsed_sec = time.time() - start_time

    out = {
        "struct_mode": struct_mode,
        "description": STRUCTURE_MODES[struct_mode],
        "final_stage": chosen,
        "test_cls_acc": float(te["cls_acc"]),

        "test_soc_rmse": float(te["soc_rmse"]),
        "test_soc_mae": float(te["soc_mae"]),
        "test_soc_mape": float(te["soc_mape"]),
        "test_soc_medape": float(te.get("soc_medape", np.nan)),

        "test_soh_rmse": float(te["soh_rmse"]),
        "test_soh_mae": float(te["soh_mae"]),
        "test_soh_mape": float(te["soh_mape"]),
        "test_soh_medape": float(te.get("soh_medape", np.nan)),

        "test_soc_rmse_raw": float(te["soc_rmse_raw"]),
        "test_soc_mae_raw": float(te["soc_mae_raw"]),
        "test_soc_mape_raw": float(te["soc_mape_raw"]),
        "test_soc_medape_raw": float(te.get("soc_medape_raw", np.nan)),

        "test_soh_rmse_raw": float(te["soh_rmse_raw"]),
        "test_soh_mae_raw": float(te["soh_mae_raw"]),
        "test_soh_mape_raw": float(te["soh_mape_raw"]),
        "test_soh_medape_raw": float(te.get("soh_medape_raw", np.nan)),

        "n_train": int(len(ds_tr)),
        "n_test": int(len(ds_te)),
        "num_classes": int(num_classes),
        "device": device,
        "elapsed_sec": float(elapsed_sec),
    }

    pd.DataFrame([out]).to_csv(
        metrics_dir / "final_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _save_json(metrics_dir / "final_metrics.json", out)

    print("\n[FINAL METRICS]")
    for key, value in out.items():
        print(f"{key}: {value}")

    return out


# =============================================================================
# Runner
# =============================================================================

def _add_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    summary = summary.copy()
    summary["fine_acc_pct"] = summary["test_cls_acc"].astype(float) * 100.0
    if "test_material_acc" in summary.columns:
        summary["material_acc_pct"] = summary["test_material_acc"].astype(float) * 100.0
    summary["soc_medae_pp"] = summary.get("test_soc_medae_raw", np.nan)
    summary["soh_medae_pp"] = summary.get("test_soh_medae_raw", np.nan)
    summary["soc_medape_pct"] = summary["test_soc_medape_raw"].astype(float)
    summary["soh_medape_pct"] = summary["test_soh_medape_raw"].astype(float)

    if "hierarchical" in set(summary["struct_mode"]):
        ref = summary.loc[summary["struct_mode"] == "hierarchical"].iloc[0]
        summary["fine_acc_change_pp_vs_hierarchical"] = (
            summary["test_cls_acc"].astype(float) - float(ref["test_cls_acc"])
        ) * 100.0
        if "test_material_acc" in summary.columns:
            summary["material_acc_change_pp_vs_hierarchical"] = (
                summary["test_material_acc"].astype(float) - float(ref["test_material_acc"])
            ) * 100.0
        summary["soc_medae_change_pp_vs_hierarchical"] = (
            summary["test_soc_medae_raw"].astype(float) - float(ref["test_soc_medae_raw"])
        )
        summary["soh_medae_change_pp_vs_hierarchical"] = (
            summary["test_soh_medae_raw"].astype(float) - float(ref["test_soh_medae_raw"])
        )
    return summary


def run_hierarchy_ablation(
    data_root: str | Path,
    output_root: str | Path,
    smoke: bool = False,
    resume: bool = True,
    proposed_summary_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Run hierarchy-structure ablation.

    Modes:
    - independent
    - soc_to_soh
    - hierarchical
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if smoke:
        struct_modes = ["independent", "soc_to_soh"]
        pulse_list = [5000]

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
            "final_eval_n_mc_soc": 500,
            "final_eval_n_mc_soh": 500,
        }

    else:
        struct_modes = ["independent", "soc_to_soh"]
        pulse_list = DEFAULT_PULSE_LIST

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
            "final_eval_n_mc_soc": 500,
            "final_eval_n_mc_soh": 500,
        }

    rows = []

    for struct_mode in struct_modes:
        exp_dir = output_root / struct_mode

        print("\n" + "=" * 90)
        print(f"[RUN] Hierarchy ablation: {struct_mode}")
        print(f"[RUN] Meaning: {STRUCTURE_MODES[struct_mode]}")
        print(f"[RUN] Pulse list: {pulse_list}")
        print(f"[RUN] Output directory: {exp_dir}")
        print("=" * 90)

        out = run_hierarchy_experiment(
            data_root=data_root,
            pulse_list=pulse_list,
            struct_mode=struct_mode,
            exp_dir=exp_dir,

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

            **run_kwargs,
        )

        row = {
            "config": struct_mode,
            "pulse_widths_ms": ",".join(map(str, pulse_list)),
            "num_pulse_widths": len(pulse_list),
            **out,
        }

        rows.append(row)

        partial = pd.DataFrame(rows)
        partial = _add_summary_columns(partial)

        partial.to_csv(
            output_root / "hierarchy_ablation_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )

        _save_json(
            output_root / "hierarchy_ablation_partial.json",
            rows,
        )

    print("\n" + "=" * 90)
    print("[RUN] Hierarchy ablation: hierarchical")
    print("[RUN] Source: proposed further-analysis summary; no retraining")
    print("=" * 90)

    out = load_hierarchical_reference_from_proposed(
        proposed_summary_path=proposed_summary_path,
    )
    row = {
        "config": "hierarchical",
        "pulse_widths_ms": ",".join(map(str, pulse_list)),
        "num_pulse_widths": len(pulse_list),
        **out,
    }
    rows.append(row)

    partial = pd.DataFrame(rows)
    partial = _add_summary_columns(partial)
    partial.to_csv(
        output_root / "hierarchy_ablation_partial.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _save_json(
        output_root / "hierarchy_ablation_partial.json",
        rows,
    )

    summary = pd.DataFrame(rows)
    summary = _add_summary_columns(summary)

    summary.to_csv(
        output_root / "hierarchy_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    _save_json(
        output_root / "hierarchy_ablation_summary.json",
        summary.to_dict(orient="records"),
    )

    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true", help="Reload existing checkpoints and evaluate only.")
    parser.add_argument(
        "--config", default="all",
        choices=["all", "independent", "soc_to_soh", "hierarchical"],
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--proposed-summary-path", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else PROJECT_ROOT / "data"
    output_root = PROJECT_ROOT / "results" / "ablation" / "hierarchy_ablation"

    if args.summary_only:
        summary = run_summary_only_hierarchy_ablation(
            data_root=data_root,
            output_root=output_root,
            selected_config=args.config,
            proposed_summary_path=args.proposed_summary_path,
        )
    else:
        if args.config not in {"all"}:
            raise ValueError("Training mode supports --config all only. Use --summary-only for one configuration.")
        summary = run_hierarchy_ablation(
            data_root=data_root,
            output_root=output_root,
            smoke=False,
            resume=not args.no_resume,
            proposed_summary_path=args.proposed_summary_path,
        )

    print("\n[SUMMARY]")
    print(summary)


if __name__ == "__main__":
    main()