# examples/test_feature_builder.py
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from proposed_framework.data.feature_builder import (
    build_three_channel_representation,
    build_3ch_5x8_from_u41,
)


def main() -> None:
    u = np.arange(1, 42, dtype=np.float32)

    x = build_three_channel_representation(u)

    print("[TEST] Input shape:", u.shape)
    print("[TEST] Output shape:", x.shape)

    assert x.shape == (3, 5, 8), f"Unexpected output shape: {x.shape}"

    expected_ch1 = np.arange(2, 42, dtype=np.float32).reshape(5, 8)
    assert np.allclose(x[0], expected_ch1), "Channel 1 is incorrect."

    expected_diff = np.ones(40, dtype=np.float32).reshape(5, 8)
    assert np.allclose(x[1], expected_diff), "Channel 2 is incorrect."

    expected_ch3 = np.ones((5, 8), dtype=np.float32)
    assert np.allclose(x[2], expected_ch3), "Channel 3 is incorrect."

    x_alias = build_3ch_5x8_from_u41(u)
    assert np.allclose(x, x_alias), "Backward-compatible alias gives different output."

    print("[PASS] Feature builder test passed.")


if __name__ == "__main__":
    main()