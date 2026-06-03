# examples/smoke_test_proposed_framework.py
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from proposed_framework.run_proposed_framework import run_experiment


def main() -> None:
    data_root = PROJECT_ROOT / "data"
    exp_dir = PROJECT_ROOT / "results" / "smoke_test_proposed_framework"

    print("[TEST] Data root:", data_root)
    print("[TEST] Exp dir:", exp_dir)

    out = run_experiment(
        data_root=str(data_root),
        pulse_list=[5000],
        u_start=1,
        u_end=41,
        drop_first_class=True,
        soc_col="SOC",
        soh_col="SOH",
        use_pt_as_feature=True,
        batch_size=32,
        lr=3e-4,
        weight_decay=1e-4,
        grad_clip=5.0,
        max_epochs=10,
        early_stopping=False,
        patience=1,
        resume=False,
        num_workers=0,
        seed=42,
        width=16,
        blocks=1,
        drop2d=0.0,
        head_dropout=0.1,
        w_cls=1.0,
        w_soc=1.0,
        w_soh=1.0,
        test_id_frac=0.2,
        test_id_count=0,
        normalize_soc=True,
        zscore_normalize=False,
        two_stage=False,
        stage1_epochs=1,
        stage2_epochs=1,
        finetune_epochs=0,
        freeze_encoder_stage2=True,
        freeze_mat_soc_stage2=True,
        use_soc_prior_weighting=False,
        use_soh_prior_weighting=False,
        soc_prior_bins=10,
        soh_prior_bins=10,
        soc_prior_low=0.5,
        soc_prior_mid=1.0,
        soc_prior_high=0.8,
        soh_prior_low=0.8,
        soh_prior_mid=1.0,
        soh_prior_high=0.9,
        alpha_score=0.1,
        final_best_stage="single",
        exp_dir=exp_dir,
    )

    print("[TEST] Smoke test output:")
    for k, v in out.items():
        print(f"{k}: {v}")

    required_keys = [
        "test_cls_acc",
        "test_soc_medape_raw",
        "test_soh_medape_raw",
        "n_train",
        "n_test",
    ]

    missing = [k for k in required_keys if k not in out]
    if missing:
        raise RuntimeError(f"Missing expected output keys: {missing}")

    print("[PASS] Smoke test passed.")


if __name__ == "__main__":
    main()