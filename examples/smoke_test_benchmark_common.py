# -*- coding: utf-8 -*-
"""
examples/smoke_test_benchmark_common.py

Smoke test for benchmark/common.py and benchmark/enhanced_inputs.py.

This does NOT require real battery data.
It checks:
1) metrics
2) controlled material hint
3) pseudo SOC hint
4) enhanced input dimensions
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from benchmark.common import rmse, mae, mape, median_ape
from benchmark.enhanced_inputs import (
    make_controlled_material_hint,
    synthesize_soc_with_target_rmse,
    build_enhanced_inputs,
)


def main():
    rng = np.random.RandomState(42)

    n_train = 50
    n_test = 20
    base_dim = 42
    num_classes = 4

    Xtr = rng.normal(size=(n_train, base_dim)).astype(np.float32)
    Xte = rng.normal(size=(n_test, base_dim)).astype(np.float32)

    ytr_cls = rng.randint(0, num_classes, size=n_train)
    yte_cls = rng.randint(0, num_classes, size=n_test)

    soc_tr = rng.uniform(5, 85, size=n_train).astype(np.float32)
    soc_te = rng.uniform(5, 85, size=n_test).astype(np.float32)

    mat_cls, mat_oh, mat_acc = make_controlled_material_hint(
        y_true_cls=yte_cls,
        num_classes=num_classes,
        target_acc=0.9,
        seed=123,
    )

    soc_hint = synthesize_soc_with_target_rmse(
        soc_true=soc_te,
        target_rmse=8.0,
        seed=123,
    )

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

    assert Xtr_soc.shape[1] == base_dim + num_classes
    assert Xte_soc.shape[1] == base_dim + num_classes
    assert Xtr_soh.shape[1] == base_dim + num_classes + 1
    assert Xte_soh.shape[1] == base_dim + num_classes + 1

    print("[SMOKE TEST PASSED]")
    print("material hint acc:", mat_acc)
    print("pseudo SOC hint RMSE:", rmse(soc_te, soc_hint))
    print("example MAE:", mae(soc_te, soc_hint))
    print("example MAPE:", mape(soc_te, soc_hint))
    print("example MedAPE:", median_ape(soc_te, soc_hint))
    print("enhanced report:", report)


if __name__ == "__main__":
    main()