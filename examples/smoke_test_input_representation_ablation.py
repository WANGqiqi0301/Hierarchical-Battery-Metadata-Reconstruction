# examples/smoke_test_input_representation_ablation.py
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ablation.input_representation_ablation import (
    run_input_representation_ablation,
)


def main() -> None:
    data_root = PROJECT_ROOT / "data"
    output_root = PROJECT_ROOT / "results" / "smoke_test_input_representation_ablation"

    summary = run_input_representation_ablation(
        data_root=data_root,
        output_root=output_root,
        smoke=True,
        resume=False,
        run_structured=True,
    )

    print("\n[SMOKE SUMMARY]")
    print(summary)

    assert len(summary) == 2
    assert "input_representation" in summary.columns
    assert "test_cls_acc" in summary.columns
    assert "test_soc_medape_raw" in summary.columns
    assert "test_soh_medape_raw" in summary.columns
    assert "n_train" in summary.columns
    assert "n_test" in summary.columns

    print("[PASS] Input-representation ablation smoke test passed.")


if __name__ == "__main__":
    main()