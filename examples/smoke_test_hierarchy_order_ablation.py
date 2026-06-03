# examples/smoke_test_hierarchy_order_ablation.py
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ablation.hierarchy_order_ablation import run_hierarchy_order_ablation


def main() -> None:
    data_root = PROJECT_ROOT / "data"
    output_root = PROJECT_ROOT / "results" / "smoke_test_hierarchy_order_ablation"

    summary = run_hierarchy_order_ablation(
        data_root=data_root,
        output_root=output_root,
        smoke=True,
        seed=42,
        normalize_soc=True,
        zscore_normalize=True,
        use_pt_as_feature=True,
    )

    print("\n[SMOKE SUMMARY]")
    print(summary)

    assert len(summary) == 2
    assert "order" in summary.columns
    assert "cls_acc" in summary.columns
    assert "soc_medape_raw" in summary.columns
    assert "soh_medape_raw" in summary.columns
    assert "n_train" in summary.columns
    assert "n_test" in summary.columns

    print("[PASS] Hierarchy-order ablation smoke test passed.")


if __name__ == "__main__":
    main()