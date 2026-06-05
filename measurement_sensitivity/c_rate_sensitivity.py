# measurement_sensitivity/c_rate_sensitivity.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import sys
import json
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from proposed_framework.run_proposed_framework import run_experiment


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

    "C11_All": [1, 2, 3, 4, 5],
}


def _combo_to_label(combo: List[int]) -> str:
    return ",".join(C_RATE_NAMES[i] for i in combo)


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

    summary["relative_input_length_pct"] = (
        summary["num_c_rates"].astype(float) / 5.0 * 100.0
    )

    return summary


def run_c_rate_sensitivity(
    data_root: str | Path,
    output_root: str | Path,
    smoke: bool = False,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Run C-rate sensitivity experiments.

    The selected C-rate rows are passed to run_experiment through c_rate_combo.

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

    if smoke:
        configs = {
            "SMOKE_C3_1p5C": [3],
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
        configs = C_RATE_CONFIGS

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
            pulse_list=[30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000],
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

    summary = pd.DataFrame(rows)
    summary = _add_c_rate_summary_columns(summary)

    summary.to_csv(
        output_root / "c_rate_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        output_root / "c_rate_sensitivity_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    data_root = PROJECT_ROOT / "data"
    output_root = PROJECT_ROOT / "results" / "measurement_sensitivity" / "c_rate"

    summary = run_c_rate_sensitivity(
        data_root=data_root,
        output_root=output_root,
        smoke=False,
        resume=True,
    )

    print("\n[SUMMARY]")
    print(summary)


if __name__ == "__main__":
    main()