# examples/test_data_loading.py
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_loader import LoadConfig, load_pulsebat_classification, preview
from proposed_framework.data.build_dataset import (
    build_train_mix_soc_mix_pt,
    build_test_random_mix_pt,
)


def main() -> None:
    data_root = PROJECT_ROOT / "data"

    print("[TEST] Data root:", data_root)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    cfg = LoadConfig(
        data_root=data_root,
        soc=85,
        pulse_width_ms=5000,
        u_start=1,
        u_end=41,
        drop_first_21_only_class=True,
        include_soc_in_X=False,
        verbose=True,
    )

    X, y, meta = load_pulsebat_classification(cfg)

    print("[TEST] One SOC + one pulse:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("meta shape:", meta.shape)
    print("meta columns:", list(meta.columns))

    assert X.shape[1] == 41, f"Expected 41 U columns, got {X.shape[1]}"
    assert len(X) == len(y) == len(meta), "X/y/meta length mismatch."

    preview(X, y, meta, n=5)

    Xtr, ytr, mtr = build_train_mix_soc_mix_pt(
        data_root=str(data_root),
        soc_list=[5, 10],
        pulse_list=[5000],
        u_start=1,
        u_end=41,
        drop_first_class=True,
    )

    print("[TEST] Small train mix:")
    print("Xtr shape:", Xtr.shape)
    print("ytr shape:", ytr.shape)
    print("mtr shape:", mtr.shape)

    assert Xtr.shape[1] == 41
    assert len(Xtr) == len(ytr) == len(mtr)

    Xte, yte, mte = build_test_random_mix_pt(
        data_root=str(data_root),
        pulse_list=[5000],
        u_start=1,
        u_end=41,
        drop_first_class=True,
    )

    print("[TEST] Random-SOC test:")
    print("Xte shape:", Xte.shape)
    print("yte shape:", yte.shape)
    print("mte shape:", mte.shape)

    assert Xte.shape[1] == 41
    assert len(Xte) == len(yte) == len(mte)

    required_meta_cols = ["ID", "SOC", "SOH"]
    missing = [c for c in required_meta_cols if c not in mtr.columns or c not in mte.columns]

    if missing:
        raise RuntimeError(f"Missing required metadata columns: {missing}")

    print("[PASS] Data loading test passed.")


if __name__ == "__main__":
    main()