# proposed_framework/data/feature_builder.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Optional

import numpy as np


def build_three_channel_representation(
    u: np.ndarray,
    c_rate_combo: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Convert U1-U41 voltage features into a three-channel representation.

    Default output:
        (3, 5, 8), using all five C-rate rows.

    If c_rate_combo is provided:
        (3, len(c_rate_combo), 8), using selected C-rate rows.

    C-rate index convention
    -----------------------
    c_rate_combo uses 1-based indices:
        1 -> first C-rate row, e.g. 0.5C
        2 -> second C-rate row, e.g. 1.0C
        3 -> third C-rate row, e.g. 1.5C
        4 -> fourth C-rate row, e.g. 2.0C
        5 -> fifth C-rate row, e.g. 2.5C

    Channel 1:
        U2-U41 reshaped into 5x8, then optionally subset by C-rate rows.

    Channel 2:
        Voltage-jump / voltage-difference features:
        U2-U1, U3-U2, ..., U41-U40, reshaped into 5x8, then optionally subset.

    Channel 3:
        Rested-voltage baseline U1 repeated into 5x8, then optionally subset.
    """
    u = np.asarray(u, dtype=np.float32)

    if u.shape[0] != 41:
        raise ValueError(f"Expected 41 U values, got {u.shape[0]}.")

    u1 = float(u[0])
    u2_41 = u[1:]

    ch1_full = u2_41.reshape(5, 8)

    diff = np.empty(40, dtype=np.float32)
    diff[0] = u[1] - u[0]
    diff[1:] = u[2:] - u[1:-1]
    ch2_full = diff.reshape(5, 8)

    ch3_full = np.full((5, 8), u1, dtype=np.float32)

    if c_rate_combo is None:
        indices = list(range(5))
    else:
        if len(c_rate_combo) == 0:
            raise ValueError("c_rate_combo cannot be empty.")

        indices = [int(i) - 1 for i in c_rate_combo]

        if min(indices) < 0 or max(indices) > 4:
            raise ValueError(
                f"c_rate_combo must contain 1-based indices from 1 to 5. "
                f"Got: {c_rate_combo}"
            )

    ch1 = ch1_full[indices, :]
    ch2 = ch2_full[indices, :]
    ch3 = ch3_full[indices, :]

    return np.stack([ch1, ch2, ch3], axis=0)


def build_3ch_5x8_from_u41(u: np.ndarray) -> np.ndarray:
    """
    Backward-compatible alias for old full-C-rate scripts.
    """
    return build_three_channel_representation(u, c_rate_combo=None)


def build_3ch_nx8_from_u41(u: np.ndarray, combo: List[int]) -> np.ndarray:
    """
    Backward-compatible alias for old C-rate-combination scripts.
    """
    return build_three_channel_representation(u, c_rate_combo=combo)