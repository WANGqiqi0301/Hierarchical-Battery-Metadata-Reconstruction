# examples/smoke_test_benchmark_full_flow.py
# -*- coding: utf-8 -*-

"""
Full-flow smoke test for benchmark models.

This script runs the original benchmark pipeline with smaller settings:

- models: xgboost, tabnet, ft_transformer, node
- settings: fair + enhanced
- quick=True
- smaller data by using fewer pulse widths

Outputs:
    results/smoke_test/benchmark/

Run from project root:

    python examples/smoke_test_benchmark_full_flow.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import json
import traceback
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    data_root = PROJECT_ROOT / "data"
    out_dir = PROJECT_ROOT / "results" / "smoke_test" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data folder not found: {data_root}\n"
            f"Please make sure your dataset is placed under:\n"
            f"  {data_root}"
        )

    # Import original benchmark modules.
    import benchmark.run_all_benchmarks as runner
    import benchmark.xgboost_benchmark as xgb_bench
    import benchmark.tabnet_benchmark as tabnet_bench
    import benchmark.ft_transformer_benchmark as ft_bench
    import benchmark.node_benchmark as node_bench

    # -------------------------------------------------------------------------
    # Smoke-test settings
    # -------------------------------------------------------------------------
    smoke_pulse_list = [1000]  # use only one pulse width to reduce data amount

    # Redirect all benchmark outputs to smoke_test folder.
    smoke_base_dir = str(out_dir)

    runner.BASE_DIR = smoke_base_dir
    xgb_bench.BASE_DIR = smoke_base_dir
    tabnet_bench.BASE_DIR = smoke_base_dir
    ft_bench.BASE_DIR = smoke_base_dir
    node_bench.BASE_DIR = smoke_base_dir

    # Reduce data amount by overriding pulse list in the original modules.
    xgb_bench.PULSE_LIST = smoke_pulse_list
    tabnet_bench.PULSE_LIST = smoke_pulse_list
    ft_bench.PULSE_LIST = smoke_pulse_list
    node_bench.PULSE_LIST = smoke_pulse_list

    smoke_config = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "data_root": str(data_root),
        "out_dir": smoke_base_dir,
        "models": ["xgboost", "tabnet", "ft_transformer", "node"],
        "setting": "both",
        "quick": True,
        "use_cache": False,
        "pulse_list": smoke_pulse_list,
    }

    with open(out_dir / "smoke_test_config.json", "w", encoding="utf-8") as f:
        json.dump(smoke_config, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[BENCHMARK FULL-FLOW SMOKE TEST]")
    print("=" * 80)
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] data_root    = {data_root}")
    print(f"[INFO] out_dir      = {out_dir}")
    print(f"[INFO] pulse_list   = {smoke_pulse_list}")
    print("[INFO] models       = xgboost, tabnet, ft_transformer, node")
    print("[INFO] setting      = both")
    print("[INFO] quick        = True")
    print("[INFO] use_cache    = False")

    df = runner.run_all(
        data_root=str(data_root),
        models=["xgboost", "tabnet", "ft_transformer", "node"],
        setting="both",
        quick=True,
        use_cache=False,
    )

    expected_csv = out_dir / "benchmark_comparison_summary.csv"

    if not expected_csv.exists():
        raise RuntimeError(
            f"Benchmark smoke test finished, but summary CSV was not saved:\n"
            f"{expected_csv}"
        )

    expected_rows = 8  # 4 models × 2 settings
    if len(df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, got {len(df)} rows.\n"
            f"Something may have skipped."
        )

    print("\n" + "=" * 80)
    print("[SMOKE TEST PASSED]")
    print("=" * 80)
    print(f"[SAVED] {expected_csv}")
    # print(df.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 80)
        print("[BENCHMARK FULL-FLOW SMOKE TEST FAILED]")
        print("=" * 80)
        traceback.print_exc()
        sys.exit(1)