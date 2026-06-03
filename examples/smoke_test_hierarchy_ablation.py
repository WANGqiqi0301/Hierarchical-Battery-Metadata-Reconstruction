# examples/smoke_test_hierarchy_ablation.py
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ablation.hierarchy_ablation import run_hierarchy_ablation


def main() -> None:
    data_root = PROJECT_ROOT / "data"
    output_root = PROJECT_ROOT / "results" / "smoke_test_hierarchy_ablation"

    summary = run_hierarchy_ablation(
        data_root=data_root,
        output_root=output_root,
        smoke=True,
        resume=False,
    )

    print("\n[SMOKE SUMMARY]")
    print(summary)

    assert len(summary) == 3
    assert "struct_mode" in summary.columns
    assert "test_cls_acc" in summary.columns
    assert "test_soc_medape_raw" in summary.columns
    assert "test_soh_medape_raw" in summary.columns
    assert "n_train" in summary.columns
    assert "n_test" in summary.columns

    print("[PASS] Hierarchy ablation smoke test passed.")


if __name__ == "__main__":
    main()