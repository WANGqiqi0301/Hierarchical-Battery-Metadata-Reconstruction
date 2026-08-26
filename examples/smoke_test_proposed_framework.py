# examples/smoke_test.py
# -*- coding: utf-8 -*-

"""
Smoke test for proposed_framework/run_proposed_framework.py

Run from project root:

    python examples/smoke_test.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import traceback
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from proposed_framework.run_proposed_framework import run_experiment


def run_basic_smoke_test(data_root: Path, exp_dir: Path) -> None:
    print("=" * 80)
    print("[SMOKE TEST] Starting basic proposed framework smoke test")
    print("=" * 80)

    results = run_experiment(
        data_root=str(data_root),
        pulse_list=[1000],

        width=8,
        blocks=1,
        drop2d=0.0,
        head_dropout=0.1,

        u_start=1,
        u_end=41,
        drop_first_class=True,
        soc_col="SOC",
        soh_col="SOH",
        use_pt_as_feature=True,

        two_stage=False,
        max_epochs=1,
        batch_size=32,
        lr=3e-4,
        weight_decay=1e-4,
        grad_clip=5.0,

        early_stopping=False,
        patience=1,
        resume=False,

        seed=2026,
        num_workers=0,

        test_id_frac=0.1,
        test_id_count=0,

        normalize_soc=True,
        zscore_normalize=True,

        w_cls=1.0,
        w_soc=1.0,
        w_soh=1.0,

        use_soc_prior_weighting=False,
        use_soh_prior_weighting=False,

        stage1_epochs=1,
        stage2_epochs=1,
        finetune_epochs=0,
        freeze_encoder_stage2=True,
        freeze_mat_soc_stage2=True,

        alpha_score=0.1,
        final_best_stage="single",

        exp_dir=exp_dir,
    )

    print("\n[Returned metrics]")
    for key, value in results.items():
        print(f"{key}: {value}")

    final_metrics_csv = exp_dir / "metrics" / "final_metrics.csv"
    final_metrics_json = exp_dir / "metrics" / "final_metrics.json"
    best_ckpt = exp_dir / "checkpoints" / "single" / "best.pt"
    last_ckpt = exp_dir / "checkpoints" / "single" / "last.pt"

    print("\n[Expected output files]")
    print(f"final_metrics.csv  : {final_metrics_csv} | exists={final_metrics_csv.exists()}")
    print(f"final_metrics.json : {final_metrics_json} | exists={final_metrics_json.exists()}")
    print(f"best checkpoint    : {best_ckpt} | exists={best_ckpt.exists()}")
    print(f"last checkpoint    : {last_ckpt} | exists={last_ckpt.exists()}")

    if not final_metrics_csv.exists():
        raise RuntimeError("Smoke test finished but final_metrics.csv was not saved.")
    if not final_metrics_json.exists():
        raise RuntimeError("Smoke test finished but final_metrics.json was not saved.")
    if not best_ckpt.exists():
        raise RuntimeError("Smoke test finished but best.pt was not saved.")
    if not last_ckpt.exists():
        raise RuntimeError("Smoke test finished but last.pt was not saved.")

    print("\n[SMOKE TEST PASSED]")


def run_resume_smoke_test(data_root: Path, exp_dir: Path) -> None:
    print("\n" + "=" * 80)
    print("[RESUME TEST] Testing checkpoint resume")
    print("=" * 80)

    # First run: only epoch 0
    run_experiment(
        data_root=str(data_root),
        pulse_list=[1000],

        width=8,
        blocks=1,
        drop2d=0.0,
        head_dropout=0.1,

        u_start=1,
        u_end=41,
        drop_first_class=True,
        soc_col="SOC",
        soh_col="SOH",
        use_pt_as_feature=True,

        two_stage=False,
        max_epochs=1,
        batch_size=32,
        lr=3e-4,
        weight_decay=1e-4,
        grad_clip=5.0,

        early_stopping=False,
        patience=1,
        resume=False,

        seed=2026,
        num_workers=0,

        test_id_frac=0.1,
        test_id_count=0,

        normalize_soc=True,
        zscore_normalize=True,

        w_cls=1.0,
        w_soc=1.0,
        w_soh=1.0,

        use_soc_prior_weighting=False,
        use_soh_prior_weighting=False,

        stage1_epochs=1,
        stage2_epochs=1,
        finetune_epochs=0,
        freeze_encoder_stage2=True,
        freeze_mat_soc_stage2=True,

        alpha_score=0.1,
        final_best_stage="single",

        exp_dir=exp_dir,
    )

    last_ckpt = exp_dir / "checkpoints" / "single" / "last.pt"

    if not last_ckpt.exists():
        raise RuntimeError("Resume test failed: first last.pt was not saved.")

    ckpt1 = torch.load(last_ckpt, map_location="cpu")
    epoch1 = int(ckpt1["epoch"])

    if epoch1 != 0:
        raise RuntimeError(
            f"Resume test failed: expected first run last epoch = 0, got {epoch1}"
        )

    print(f"[RESUME TEST] First run OK: last epoch = {epoch1}")

    # Second run: should resume from epoch 1 and finish epoch 1
    run_experiment(
        data_root=str(data_root),
        pulse_list=[1000],

        width=8,
        blocks=1,
        drop2d=0.0,
        head_dropout=0.1,

        u_start=1,
        u_end=41,
        drop_first_class=True,
        soc_col="SOC",
        soh_col="SOH",
        use_pt_as_feature=True,

        two_stage=False,
        max_epochs=2,
        batch_size=32,
        lr=3e-4,
        weight_decay=1e-4,
        grad_clip=5.0,

        early_stopping=False,
        patience=1,
        resume=True,

        seed=2026,
        num_workers=0,

        test_id_frac=0.1,
        test_id_count=0,

        normalize_soc=True,
        zscore_normalize=True,

        w_cls=1.0,
        w_soc=1.0,
        w_soh=1.0,

        use_soc_prior_weighting=False,
        use_soh_prior_weighting=False,

        stage1_epochs=1,
        stage2_epochs=1,
        finetune_epochs=0,
        freeze_encoder_stage2=True,
        freeze_mat_soc_stage2=True,

        alpha_score=0.1,
        final_best_stage="single",

        exp_dir=exp_dir,
    )

    ckpt2 = torch.load(last_ckpt, map_location="cpu")
    epoch2 = int(ckpt2["epoch"])

    if epoch2 != 1:
        raise RuntimeError(
            f"Resume test failed: expected resumed last epoch = 1, got {epoch2}"
        )

    print(f"[RESUME TEST] Second run OK: resumed last epoch = {epoch2}")
    print("[RESUME TEST PASSED]")


def main() -> None:
    data_root = PROJECT_ROOT / "data"
    exp_root = PROJECT_ROOT / "results" / "smoke_test" / "proposed_framework"

    basic_exp_dir = exp_root / "basic"
    resume_exp_dir = exp_root / "resume"

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data folder not found: {data_root}\n"
            f"Please make sure your dataset is placed under:\n"
            f"  {data_root}"
        )

    print("=" * 80)
    print("[INFO] PROJECT_ROOT =", PROJECT_ROOT)
    print("[INFO] data_root    =", data_root)
    print("[INFO] exp_root     =", exp_root)
    print("=" * 80)

    run_basic_smoke_test(data_root=data_root, exp_dir=basic_exp_dir)
    run_resume_smoke_test(data_root=data_root, exp_dir=resume_exp_dir)

    print("\n" + "=" * 80)
    print("[ALL TESTS PASSED]")
    print("=" * 80)
    print(f"[DONE] Outputs saved under: {exp_root}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 80)
        print("[SMOKE TEST FAILED]")
        print("=" * 80)
        traceback.print_exc()
        sys.exit(1)