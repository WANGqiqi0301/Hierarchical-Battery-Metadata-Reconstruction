# examples/smoke_test_ablation.py
# -*- coding: utf-8 -*-

"""
Smoke test for all ablation modules.

This script runs small smoke-mode ablation experiments:

1. channel_ablation
2. hierarchy_ablation
3. hierarchy_order_ablation
4. input_representation_ablation
5. transfer_ablation
6. material_conditioning_ablation

Outputs:
    results/smoke_test/ablation/

Run from project root:

    python examples/smoke_test_ablation.py

Run one case only:

    python examples/smoke_test_ablation.py --case channel_ablation
"""

from __future__ import annotations

from pathlib import Path
import sys
import json
import traceback
import argparse
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_one(name: str, func, kwargs: dict, expected_file: Path, log_path: Path) -> dict:
    print(f"\n[RUN] {name}")

    try:
        with open(log_path, "a", encoding="utf-8") as log_f:
            log_f.write("\n\n" + "=" * 80 + "\n")
            log_f.write(f"[RUN] {name}\n")
            log_f.write("=" * 80 + "\n")

            with redirect_stdout(log_f), redirect_stderr(log_f):
                df = func(**kwargs)

        if not expected_file.exists():
            raise RuntimeError(f"Expected output file not found: {expected_file}")

        if df is None or len(df) == 0:
            raise RuntimeError(f"{name} returned empty summary.")

        print(f"[OK] {name} | rows={len(df)}")
        print(f"[SAVED] {expected_file}")

        return {
            "name": name,
            "status": "OK",
            "rows": int(len(df)),
            "expected_file": str(expected_file),
            "error_type": "",
            "error_message": "",
        }

    except Exception as e:
        print(f"[FAIL] {name}")
        print(f"       {type(e).__name__}: {e}")

        with open(log_path, "a", encoding="utf-8") as log_f:
            log_f.write("\n[FAILED]\n")
            log_f.write(traceback.format_exc())

        return {
            "name": name,
            "status": "FAIL",
            "rows": 0,
            "expected_file": str(expected_file),
            "error_type": type(e).__name__,
            "error_message": str(e),
        }


def build_tests(data_root: Path, out_root: Path):
    from ablation.channel_ablation import run_channel_ablation
    from ablation.hierarchy_ablation import run_hierarchy_ablation
    from ablation.hierarchy_order_ablation import run_hierarchy_order_ablation
    from ablation.input_representation_ablation import run_input_representation_ablation
    from ablation.transfer_ablation import run_transfer_ablation
    from ablation.material_conditioning_ablation import run_material_conditioning_ablation

    tests = [
        {
            "name": "channel_ablation",
            "func": run_channel_ablation,
            "kwargs": {
                "data_root": data_root,
                "output_root": out_root / "channel_ablation",
                "smoke": True,
                "resume": False,
            },
            "expected_file": out_root / "channel_ablation" / "channel_ablation_summary.csv",
        },
        {
            "name": "hierarchy_ablation",
            "func": run_hierarchy_ablation,
            "kwargs": {
                "data_root": data_root,
                "output_root": out_root / "hierarchy_ablation",
                "smoke": True,
                "resume": False,
            },
            "expected_file": out_root / "hierarchy_ablation" / "hierarchy_ablation_summary.csv",
        },
        {
            "name": "hierarchy_order_ablation",
            "func": run_hierarchy_order_ablation,
            "kwargs": {
                "data_root": data_root,
                "output_root": out_root / "hierarchy_order_ablation",
                "orders": None,
                "smoke": True,
                "patience": 1,
                "batch_size": 32,
                "num_workers": 0,
                "seed": 42,
            },
            "expected_file": out_root / "hierarchy_order_ablation" / "hierarchy_order_ablation_summary.csv",
        },
        {
            "name": "input_representation_ablation",
            "func": run_input_representation_ablation,
            "kwargs": {
                "data_root": data_root,
                "output_root": out_root / "input_representation_ablation",
                "smoke": True,
                "resume": False,
                "config": "raw_1d",
            },
            "expected_file": (
                out_root
                / "input_representation_ablation"
                / "input_representation_ablation_summary.csv"
            ),
        },
        {
            "name": "transfer_ablation",
            "func": run_transfer_ablation,
            "kwargs": {
                "data_root": data_root,
                "output_root": out_root / "transfer_ablation",
                "smoke": True,
                "resume": False,
            },
            "expected_file": out_root / "transfer_ablation" / "transfer_ablation_summary.csv",
        },
                {
            "name": "material_conditioning_ablation",
            "func": run_material_conditioning_ablation,
            "kwargs": {
                "data_root": data_root,
                "output_root": out_root / "material_conditioning_ablation",
                "smoke": True,
                "resume": False,
                "config": "hard",
            },
            "expected_file": (
                out_root
                / "material_conditioning_ablation"
                / "material_conditioning_ablation_summary.csv"
            ),
        },
    ]

    return tests


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation smoke tests.")
    parser.add_argument(
        "--case",
        type=str,
        default="all",
        choices=[
            "all",
            "channel_ablation",
            "hierarchy_ablation",
            "hierarchy_order_ablation",
            "input_representation_ablation",
            "transfer_ablation",
            "material_conditioning_ablation",
        ],
        help="Which ablation smoke test to run.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available ablation smoke test cases and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = PROJECT_ROOT / "data"
    out_root = PROJECT_ROOT / "results" / "smoke_test" / "ablation"
    log_path = out_root / "ablation_smoke_test.log"

    out_root.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"Data folder not found: {data_root}")

    tests = build_tests(data_root=data_root, out_root=out_root)

    if args.list:
        print("Available cases:")
        print("  all")
        for test in tests:
            print(f"  {test['name']}")
        return

    if args.case != "all":
        tests = [test for test in tests if test["name"] == args.case]

    print("=" * 80)
    print("[ABLATION SMOKE TEST]")
    print("=" * 80)
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] data_root    = {data_root}")
    print(f"[INFO] out_root     = {out_root}")
    print(f"[INFO] log_path     = {log_path}")
    print(f"[INFO] case         = {args.case}")

    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write("=" * 80 + "\n")
        log_f.write("[ABLATION SMOKE TEST LOG]\n")
        log_f.write("=" * 80 + "\n")
        log_f.write(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"project_root: {PROJECT_ROOT}\n")
        log_f.write(f"data_root: {data_root}\n")
        log_f.write(f"out_root: {out_root}\n")
        log_f.write(f"case: {args.case}\n")

    rows = []

    for test in tests:
        rows.append(
            run_one(
                name=test["name"],
                func=test["func"],
                kwargs=test["kwargs"],
                expected_file=test["expected_file"],
                log_path=log_path,
            )
        )

    summary_df = pd.DataFrame(rows)

    summary_csv = out_root / "ablation_smoke_test_summary.csv"
    summary_json = out_root / "ablation_smoke_test_summary.json"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    save_json(summary_json, rows)

    n_ok = int((summary_df["status"] == "OK").sum())
    n_fail = int((summary_df["status"] == "FAIL").sum())

    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print("=" * 80)
    print(summary_df[["name", "status", "rows", "error_type", "error_message"]].to_string(index=False))
    print(f"\nOK   : {n_ok}")
    print(f"FAIL : {n_fail}")
    print(f"\n[SAVED] summary CSV : {summary_csv}")
    print(f"[SAVED] summary JSON: {summary_json}")
    print(f"[SAVED] full log    : {log_path}")

    if n_fail > 0:
        raise RuntimeError(
            "Some ablation smoke tests failed. "
            "Please check results/smoke_test/ablation/ablation_smoke_test.log"
        )

    print("\n[ABLATION SMOKE TEST PASSED]")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 80)
        print("[ABLATION SMOKE TEST FAILED]")
        print("=" * 80)
        traceback.print_exc()
        sys.exit(1)