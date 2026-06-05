# measurement_sensitivity/pulse_width_sensitivity.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import sys
import json
from typing import Dict, List

import pandas as pd


# =============================================================================
# Project path
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Local imports
# =============================================================================

from proposed_framework.run_proposed_framework import run_experiment


# =============================================================================
# Pulse-width configurations
# =============================================================================

PULSE_WIDTH_CONFIGS: Dict[str, List[int]] = {
    "P1_70": [70],
    "P2_3000": [3000],
    "P3_30_50_70_100": [30, 50, 70, 100],
    "P4_300_500_700": [300, 500, 700],
    "P5_1000_3000_5000": [1000, 3000, 5000],
    "P6_30_50_300_500": [30, 50, 300, 500],
    "P7_30_50_3000_5000": [30, 50, 3000, 5000],
    "P8_300_500_3000_5000": [300, 500, 3000, 5000],
    "P9_All": [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000],
}


def _add_pulse_width_summary_columns(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add relative change columns using P9_All as the reference.

    The raw MedAPE values are already in percentage-point units because SOC is
    transformed back to percentage space before MedAPE calculation.
    """
    if summary.empty:
        return summary

    if "P9_All" not in set(summary["config"]):
        return summary

    ref = summary.loc[summary["config"] == "P9_All"].iloc[0]

    ref_acc = float(ref["test_cls_acc"])
    ref_soc_medape = float(ref["test_soc_medape_raw"])
    ref_soh_medape = float(ref["test_soh_medape_raw"])
    ref_width_sum = float(ref["pulse_width_sum_ms"])

    summary = summary.copy()

    summary["mat_acc_pct"] = summary["test_cls_acc"].astype(float) * 100.0
    summary["soc_medape_pct"] = summary["test_soc_medape_raw"].astype(float)
    summary["soh_medape_pct"] = summary["test_soh_medape_raw"].astype(float)

    summary["mat_acc_change_pp_vs_all"] = (
        summary["test_cls_acc"].astype(float) - ref_acc
    ) * 100.0

    summary["soc_medape_change_pp_vs_all"] = (
        summary["test_soc_medape_raw"].astype(float) - ref_soc_medape
    )

    summary["soh_medape_change_pp_vs_all"] = (
        summary["test_soh_medape_raw"].astype(float) - ref_soh_medape
    )

    summary["pulse_width_sum_relative_pct"] = (
        summary["pulse_width_sum_ms"].astype(float) / ref_width_sum * 100.0
    )

    summary["pulse_width_sum_reduction_pct"] = (
        100.0 - summary["pulse_width_sum_relative_pct"]
    )

    return summary


def run_pulse_width_sensitivity(
    data_root: str | Path,
    output_root: str | Path,
    smoke: bool = False,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Run pulse-width sensitivity experiments.

    Parameters
    ----------
    data_root:
        Root directory containing material-capacity subfolders.

    output_root:
        Root directory to save pulse-width sensitivity results.

    smoke:
        If True, run a lightweight smoke test with one configuration, one epoch
        and a small model.

    resume:
        Whether to resume from existing checkpoints.

    Returns
    -------
    pd.DataFrame
        Summary table of all pulse-width sensitivity results.
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if smoke:
        configs = {
            "SMOKE_5000": [5000],
        }

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
        configs = PULSE_WIDTH_CONFIGS

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

    for config_name, pulse_list in configs.items():
        exp_dir = output_root / config_name

        print("\n" + "=" * 90)
        print(f"[RUN] Pulse-width configuration: {config_name}")
        print(f"[RUN] Pulse widths: {pulse_list}")
        print(f"[RUN] Output directory: {exp_dir}")
        print("=" * 90)

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

            **run_kwargs,
        )

        row = {
            "config": config_name,
            "pulse_widths_ms": ",".join(map(str, pulse_list)),
            "num_pulse_widths": int(len(pulse_list)),
            "pulse_width_sum_ms": int(sum(pulse_list)),
            **out,
        }

        rows.append(row)

        partial = pd.DataFrame(rows)
        partial = _add_pulse_width_summary_columns(partial)

        partial.to_csv(
            output_root / "pulse_width_sensitivity_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )

        with open(
            output_root / "pulse_width_sensitivity_partial.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    summary = pd.DataFrame(rows)
    summary = _add_pulse_width_summary_columns(summary)

    summary.to_csv(
        output_root / "pulse_width_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        output_root / "pulse_width_sensitivity_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    """
    Run full pulse-width sensitivity experiments.

    Change data_root if your data folder is not PROJECT_ROOT / "data".
    """
    data_root = PROJECT_ROOT / "data"
    output_root = PROJECT_ROOT / "results" / "measurement_sensitivity" / "pulse_width"

    summary = run_pulse_width_sensitivity(
        data_root=data_root,
        output_root=output_root,
        smoke=False,
        resume=True,
    )

    print("\n[SUMMARY]")
    print(summary)


if __name__ == "__main__":
    main()