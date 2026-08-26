# examples/smoke_test_measurement_sensitivity.py
# -*- coding: utf-8 -*-

"""
Smoke test for measurement_sensitivity modules.

This script tests:

1. measurement_sensitivity/c_rate_sensitivity.py
2. measurement_sensitivity/pulse_width_sensitivity.py
3. measurement_sensitivity/input_quality_sensitivity.py
4. measurement_sensitivity/pulse_polarity_sensitivity.py

All tests run with smoke=True.

Outputs:
    results/smoke_test/measurement_sensitivity/
    results/smoke_test/measurement_sensitivity/smoke_test_measurement_sensitivity_summary.csv
    results/smoke_test/measurement_sensitivity/smoke_test_measurement_sensitivity_summary.json
    results/smoke_test/measurement_sensitivity/logs/*.log

Run from project root:

    python examples/smoke_test_measurement_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import json
import traceback
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

import pandas as pd


# =============================================================================
# Project root
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Paths
# =============================================================================

DATA_ROOT = PROJECT_ROOT / "data"

OUT_ROOT = PROJECT_ROOT / "results" / "smoke_test" / "measurement_sensitivity"
LOG_DIR = OUT_ROOT / "logs"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _summarize_dataframe(df: pd.DataFrame) -> dict:
    if df is None:
        return {
            "n_rows": 0,
            "n_cols": 0,
            "columns": "",
        }

    return {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "columns": ",".join(map(str, df.columns)),
    }


def _run_with_log(name: str, func, log_path: Path) -> tuple[str, pd.DataFrame | None, str, str]:
    """
    Run one smoke test and redirect verbose output to a log file.
    The terminal prints only start/end information.
    """
    print("\n" + "=" * 80)
    print(f"[RUN] {name}")
    print("=" * 80)
    print(f"[LOG] {log_path}")
    print(f"[STATUS] {name} started. Open the log file to see detailed progress.")

    try:
        with open(log_path, "w", encoding="utf-8") as log_f:
            log_f.write("=" * 80 + "\n")
            log_f.write(f"[SMOKE TEST] {name}\n")
            log_f.write(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write("=" * 80 + "\n\n")
            log_f.flush()

            with redirect_stdout(log_f), redirect_stderr(log_f):
                df = func()

            log_f.write("\n[DONE]\n")
            log_f.flush()

        if df is None:
            raise RuntimeError(f"{name} returned None.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} returned {type(df)}, expected pandas.DataFrame.")
        if df.empty:
            raise RuntimeError(f"{name} returned an empty DataFrame.")

        print(f"[OK] {name} finished | rows={len(df)} | cols={len(df.columns)}")
        return "OK", df, "", ""

    except Exception as e:
        print(f"[FAIL] {name}")
        print(f"       {type(e).__name__}: {e}")
        print(f"       Check log: {log_path}")

        with open(log_path, "a", encoding="utf-8") as log_f:
            log_f.write("\n\n" + "=" * 80 + "\n")
            log_f.write("[EXCEPTION]\n")
            log_f.write("=" * 80 + "\n")
            traceback.print_exc(file=log_f)

        return "FAIL", None, type(e).__name__, str(e)

# =============================================================================
# Individual tests
# =============================================================================

def test_c_rate_sensitivity() -> pd.DataFrame:
    from measurement_sensitivity.c_rate_sensitivity import run_c_rate_sensitivity

    output_root = OUT_ROOT / "c_rate"

    return run_c_rate_sensitivity(
        data_root=DATA_ROOT,
        output_root=output_root,
        smoke=True,
        resume=False,
    )


def test_pulse_width_sensitivity() -> pd.DataFrame:
    from measurement_sensitivity.pulse_width_sensitivity import run_pulse_width_sensitivity

    output_root = OUT_ROOT / "pulse_width"

    return run_pulse_width_sensitivity(
        data_root=DATA_ROOT,
        output_root=output_root,
        smoke=True,
        resume=False,
    )


def test_input_quality_sensitivity() -> pd.DataFrame:
    from measurement_sensitivity.input_quality_sensitivity import run_input_quality_sensitivity

    output_root = OUT_ROOT / "input_quality"
    clean_exp_dir = output_root / "clean_model"

    return run_input_quality_sensitivity(
        data_root=DATA_ROOT,
        output_root=output_root,
        clean_exp_dir=clean_exp_dir,
        smoke=True,
        train_clean_if_needed=True,
        resume_clean=False,
    )

def test_pulse_polarity_sensitivity() -> pd.DataFrame:
    from measurement_sensitivity.pulse_polarity_sensitivity import run_pulse_polarity_sensitivity

    output_root = OUT_ROOT / "pulse_polarity"

    return run_pulse_polarity_sensitivity(
        data_root=DATA_ROOT,
        output_root=output_root,
        smoke=True,
        resume=False,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Data folder not found: {DATA_ROOT}\n"
            f"Please make sure your dataset is placed under:\n"
            f"  {DATA_ROOT}"
        )

    print("=" * 80)
    print("[MEASUREMENT SENSITIVITY SMOKE TEST]")
    print("=" * 80)
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] DATA_ROOT    = {DATA_ROOT}")
    print(f"[INFO] OUT_ROOT     = {OUT_ROOT}")
    print(f"[INFO] LOG_DIR      = {LOG_DIR}")
    print("[INFO] smoke        = True")

    tests = [
        {
            "name": "c_rate_sensitivity",
            "func": test_c_rate_sensitivity,
            "expected_summary": OUT_ROOT / "c_rate" / "c_rate_sensitivity_summary.csv",
            "log": LOG_DIR / "c_rate_sensitivity.log",
        },
        {
            "name": "pulse_width_sensitivity",
            "func": test_pulse_width_sensitivity,
            "expected_summary": OUT_ROOT / "pulse_width" / "pulse_width_sensitivity_summary.csv",
            "log": LOG_DIR / "pulse_width_sensitivity.log",
        },
        {
            "name": "input_quality_sensitivity",
            "func": test_input_quality_sensitivity,
            "expected_summary": OUT_ROOT / "input_quality" / "input_quality_sensitivity_summary.csv",
            "log": LOG_DIR / "input_quality_sensitivity.log",
        },
        {
            "name": "pulse_polarity_sensitivity",
            "func": test_pulse_polarity_sensitivity,
            "expected_summary": OUT_ROOT / "pulse_polarity" / "pulse_polarity_sensitivity_summary.csv",
            "log": LOG_DIR / "pulse_polarity_sensitivity.log",
        },
    ]

    rows = []

    for item in tests:
        name = item["name"]
        func = item["func"]
        log_path = item["log"]
        expected_summary = item["expected_summary"]

        status, df, error_type, error_message = _run_with_log(
            name=name,
            func=func,
            log_path=log_path,
        )

        info = _summarize_dataframe(df)

        output_exists = expected_summary.exists()

        if status == "OK" and not output_exists:
            status = "FAIL"
            error_type = "MissingOutput"
            error_message = f"Expected summary CSV was not saved: {expected_summary}"
            print(f"[FAIL] {name} did not save expected summary CSV.")
            print(f"       {expected_summary}")

        rows.append(
            {
                "module": name,
                "status": status,
                "n_rows": info["n_rows"],
                "n_cols": info["n_cols"],
                "expected_summary": str(expected_summary),
                "expected_summary_exists": bool(output_exists),
                "log_file": str(log_path),
                "error_type": error_type,
                "error_message": error_message,
            }
        )

    summary_df = pd.DataFrame(rows)

    summary_csv = OUT_ROOT / "smoke_test_measurement_sensitivity_summary.csv"
    summary_json = OUT_ROOT / "smoke_test_measurement_sensitivity_summary.json"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    _save_json(summary_json, rows)

    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print("=" * 80)
    print(summary_df[["module", "status", "n_rows", "expected_summary_exists", "log_file"]].to_string(index=False))

    n_ok = int((summary_df["status"] == "OK").sum())
    n_fail = int((summary_df["status"] == "FAIL").sum())

    print(f"\nOK   : {n_ok}")
    print(f"FAIL : {n_fail}")
    print(f"\n[SAVED] summary CSV : {summary_csv}")
    print(f"[SAVED] summary JSON: {summary_json}")

    if n_fail > 0:
        print("\n[FAILED MODULES]")
        failed = summary_df[summary_df["status"] == "FAIL"]
        for _, r in failed.iterrows():
            print(f"- {r['module']}: {r['error_type']} | {r['error_message']}")
            print(f"  log: {r['log_file']}")

        raise RuntimeError(
            "Some measurement-sensitivity smoke tests failed. "
            "Please check the corresponding log files."
        )

    print("\n[MEASUREMENT SENSITIVITY SMOKE TEST PASSED]")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 80)
        print("[MEASUREMENT SENSITIVITY SMOKE TEST FAILED]")
        print("=" * 80)
        traceback.print_exc()
        sys.exit(1)