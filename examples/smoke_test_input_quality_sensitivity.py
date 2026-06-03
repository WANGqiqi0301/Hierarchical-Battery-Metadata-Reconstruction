# examples/smoke_test_input_quality_sensitivity.py
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from measurement_sensitivity.input_quality_sensitivity import (
    run_input_quality_sensitivity,
)


def main() -> None:
    data_root = PROJECT_ROOT / "data"

    output_root = PROJECT_ROOT / "results" / "smoke_test_input_quality_sensitivity"
    clean_exp_dir = output_root / "clean_model"

    summary = run_input_quality_sensitivity(
        data_root=data_root,
        output_root=output_root,
        clean_exp_dir=clean_exp_dir,
        smoke=True,
        train_clean_if_needed=True,
        resume_clean=False,
    )

    print("\n[SMOKE SUMMARY]")
    print(summary)

    assert len(summary) >= 4
    assert "type" in summary.columns
    assert "level" in summary.columns
    assert "cls_acc" in summary.columns
    assert "soc_medape_raw" in summary.columns
    assert "soh_medape_raw" in summary.columns
    assert "n_test" in summary.columns

    print("[PASS] Input-quality sensitivity smoke test passed.")


if __name__ == "__main__":
    main()