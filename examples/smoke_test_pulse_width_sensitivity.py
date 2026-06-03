# examples/smoke_test_pulse_width_sensitivity.py
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from measurement_sensitivity.pulse_width_sensitivity import run_pulse_width_sensitivity


def main() -> None:
    data_root = PROJECT_ROOT / "data"
    output_root = PROJECT_ROOT / "results" / "smoke_test_pulse_width_sensitivity"

    summary = run_pulse_width_sensitivity(
        data_root=data_root,
        output_root=output_root,
        smoke=True,
        resume=False,
    )

    print("\n[SMOKE SUMMARY]")
    print(summary)

    assert len(summary) == 1
    assert "test_cls_acc" in summary.columns
    assert "test_soc_medape_raw" in summary.columns
    assert "test_soh_medape_raw" in summary.columns
    assert "n_train" in summary.columns
    assert "n_test" in summary.columns

    print("[PASS] Pulse-width sensitivity smoke test passed.")


if __name__ == "__main__":
    main()