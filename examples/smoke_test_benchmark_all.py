# -*- coding: utf-8 -*-
"""
examples/smoke_test_benchmark_all.py

快速测试 benchmark 模块是否基本可用。

这个 smoke test 不读取真实电池数据，也不训练模型。
它主要检查：

1. benchmark.common 可以正常 import
2. benchmark.enhanced_inputs 可以正常 import
3. xgboost_benchmark / tabnet_benchmark / ft_transformer_benchmark / node_benchmark
   是否可以正常 import
4. enhanced input 构造逻辑是否正确
5. 指标函数是否正常
6. 每个模型的核心依赖是否缺失

运行方式：
    直接右键运行这个文件即可
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np


# =============================================================================
# Project root
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def check_import(module_name: str):
    """
    Try importing a module and return (ok, module_or_none, error_or_none).
    """
    try:
        module = __import__(module_name, fromlist=["*"])
        print(f"[OK] import {module_name}")
        return True, module, None
    except Exception as e:
        print(f"[FAIL] import {module_name}")
        print(f"       {type(e).__name__}: {e}")
        return False, None, e


def test_common_metrics():
    from benchmark.common import rmse, mae, mape, median_ape

    y_true = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    y_pred = np.array([12.0, 18.0, 33.0], dtype=np.float32)

    out = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "median_ape": median_ape(y_true, y_pred),
    }

    assert out["rmse"] > 0
    assert out["mae"] > 0
    assert out["mape"] > 0
    assert out["median_ape"] > 0

    print("[OK] benchmark.common metrics")
    print("     ", out)


def test_enhanced_inputs():
    from benchmark.enhanced_inputs import (
        make_controlled_material_hint,
        synthesize_soc_with_target_rmse,
        build_enhanced_inputs,
    )
    from benchmark.common import rmse

    rng = np.random.RandomState(42)

    n_train = 40
    n_test = 16
    base_dim = 42
    num_classes = 4

    Xtr = rng.normal(size=(n_train, base_dim)).astype(np.float32)
    Xte = rng.normal(size=(n_test, base_dim)).astype(np.float32)

    ytr_cls = rng.randint(0, num_classes, size=n_train)
    yte_cls = rng.randint(0, num_classes, size=n_test)

    soc_tr = rng.uniform(5, 85, size=n_train).astype(np.float32)
    soc_te = rng.uniform(5, 85, size=n_test).astype(np.float32)

    hint_cls, hint_oh, acc = make_controlled_material_hint(
        y_true_cls=yte_cls,
        num_classes=num_classes,
        target_acc=0.9,
        seed=123,
    )

    assert hint_cls.shape == (n_test,)
    assert hint_oh.shape == (n_test, num_classes)
    assert 0.0 <= acc <= 1.0

    soc_hint = synthesize_soc_with_target_rmse(
        soc_true=soc_te,
        target_rmse=8.0,
        seed=123,
    )

    assert soc_hint.shape == (n_test,)

    Xtr_soc, Xte_soc, Xtr_soh, Xte_soh, report = build_enhanced_inputs(
        Xtr=Xtr,
        Xte=Xte,
        ytr_cls=ytr_cls,
        yte_cls=yte_cls,
        soc_tr_true=soc_tr,
        soc_te_true=soc_te,
        num_classes=num_classes,
        target_material_acc=0.9,
        target_soc_rmse=8.0,
        seed=42,
    )

    assert Xtr_soc.shape == (n_train, base_dim + num_classes)
    assert Xte_soc.shape == (n_test, base_dim + num_classes)
    assert Xtr_soh.shape == (n_train, base_dim + num_classes + 1)
    assert Xte_soh.shape == (n_test, base_dim + num_classes + 1)

    print("[OK] benchmark.enhanced_inputs")
    print("     material hint acc:", acc)
    print("     pseudo SOC RMSE:", rmse(soc_te, soc_hint))
    print("     feature dims:", report["feature_dims"])


def test_optional_model_dependencies():
    """
    Check whether optional model dependencies exist.

    Missing TabNet / FT-Transformer / NODE dependency does not mean the whole
    benchmark package is broken. It only means that specific model cannot run
    until dependency is installed.
    """
    checks = [
        ("xgboost", "xgboost"),
        ("pytorch_tabnet", "pytorch-tabnet"),
        ("rtdl_revisiting_models", "rtdl-revisiting-models"),
        ("qhoptim", "qhoptim"),
        ("lib", "NODE local lib module"),
    ]

    print("\n===== Optional dependency check =====")
    for module_name, install_name in checks:
        try:
            __import__(module_name)
            print(f"[OK] {module_name}")
        except Exception as e:
            print(f"[MISSING] {module_name}  -> needed for {install_name}")
            print(f"          {type(e).__name__}: {e}")


def main():
    print("===== Benchmark smoke test starts =====")
    print("Project root:", PROJECT_ROOT)

    required_modules = [
        "benchmark.common",
        "benchmark.enhanced_inputs",
        "benchmark.xgboost_benchmark",
    ]

    optional_benchmark_modules = [
        "benchmark.tabnet_benchmark",
        "benchmark.ft_transformer_benchmark",
        "benchmark.node_benchmark",
        "benchmark.run_all_benchmarks",
    ]

    print("\n===== Required imports =====")
    required_ok = True
    for m in required_modules:
        ok, _, _ = check_import(m)
        required_ok = required_ok and ok

    print("\n===== Optional benchmark imports =====")
    optional_results = {}
    for m in optional_benchmark_modules:
        ok, _, err = check_import(m)
        optional_results[m] = ok

    if not required_ok:
        print("\n[STOP] Required benchmark modules failed to import.")
        print("Please fix the errors above first.")
        return

    print("\n===== Function tests =====")
    try:
        test_common_metrics()
        test_enhanced_inputs()
    except Exception:
        print("[FAIL] Function tests failed.")
        traceback.print_exc()
        return

    test_optional_model_dependencies()

    print("\n===== Summary =====")
    print("[OK] common.py works")
    print("[OK] enhanced_inputs.py works")
    print("[OK] xgboost_benchmark.py imports")

    for m, ok in optional_results.items():
        if ok:
            print(f"[OK] {m} imports")
        else:
            print(f"[SKIP] {m} has missing dependency or import issue")

    print("\n[SMOKE TEST FINISHED]")


if __name__ == "__main__":
    main()