# proposed_framework/data/build_dataset.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from utils.data_loader import LoadConfig, load_pulsebat_classification


SOCSpec = Union[int, str]


def load_one_soc_one_pulse(
    data_root: str,
    soc: SOCSpec,
    pulse_ms: int,
    u_start: int,
    u_end: int,
    drop_first_class: bool,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load one SOC sheet and one pulse width.
    """
    cfg = LoadConfig(
        data_root=data_root,
        soc=soc,
        pulse_width_ms=int(pulse_ms),
        u_start=u_start,
        u_end=u_end,
        drop_first_21_only_class=drop_first_class,
        include_soc_in_X=False,
        verbose=False,
    )

    X_df, y_ser, meta = load_pulsebat_classification(cfg)

    X = X_df.to_numpy(dtype=float)
    y = y_ser.to_numpy()

    meta = meta.copy()
    meta["pulse_ms"] = int(pulse_ms)

    return X, y, meta


def build_train_mix_soc_mix_pt(
    data_root: str,
    soc_list: List[int],
    pulse_list: List[int],
    u_start: int,
    u_end: int,
    drop_first_class: bool,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build training data from multiple SOC sheets and multiple pulse widths.
    """
    Xs = []
    ys = []
    metas = []

    for soc in soc_list:
        for pulse_ms in pulse_list:
            X, y, meta = load_one_soc_one_pulse(
                data_root=data_root,
                soc=int(soc),
                pulse_ms=int(pulse_ms),
                u_start=u_start,
                u_end=u_end,
                drop_first_class=drop_first_class,
            )

            if len(y) == 0:
                continue

            Xs.append(X)
            ys.append(y)
            metas.append(meta)

    if not Xs:
        return np.zeros((0, 0)), np.array([]), pd.DataFrame()

    return (
        np.vstack(Xs),
        np.concatenate(ys),
        pd.concat(metas, axis=0, ignore_index=True),
    )


def build_test_random_mix_pt(
    data_root: str,
    pulse_list: List[int],
    u_start: int,
    u_end: int,
    drop_first_class: bool,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build test data from the random-SOC test sheet and multiple pulse widths.
    """
    Xs = []
    ys = []
    metas = []

    for pulse_ms in pulse_list:
        X, y, meta = load_one_soc_one_pulse(
            data_root=data_root,
            soc="TEST_RANDOM",
            pulse_ms=int(pulse_ms),
            u_start=u_start,
            u_end=u_end,
            drop_first_class=drop_first_class,
        )

        if len(y) == 0:
            continue

        Xs.append(X)
        ys.append(y)
        metas.append(meta)

    if not Xs:
        return np.zeros((0, 0)), np.array([]), pd.DataFrame()

    return (
        np.vstack(Xs),
        np.concatenate(ys),
        pd.concat(metas, axis=0, ignore_index=True),
    )


def pick_test_ids(
    all_ids: np.ndarray,
    test_id_frac: float = 0.2,
    test_id_count: int = 0,
    seed: int = 42,
) -> np.ndarray:
    """
    Randomly select test IDs for group-wise splitting.
    """
    ids = np.array(pd.Series(all_ids).astype(str).unique(), dtype=object)
    n_all = len(ids)

    if n_all == 0:
        raise RuntimeError("No IDs found to split.")

    rng = np.random.RandomState(seed)
    rng.shuffle(ids)

    if test_id_count and test_id_count > 0:
        n_test = int(min(max(1, test_id_count), n_all - 1))
    else:
        n_test = int(max(1, round(n_all * float(test_id_frac))))
        n_test = min(n_test, n_all - 1)

    return ids[:n_test]


def apply_id_split(
    Xtr: np.ndarray,
    ytr_str: np.ndarray,
    mtr: pd.DataFrame,
    Xte: np.ndarray,
    yte_str: np.ndarray,
    mte: pd.DataFrame,
    test_ids: np.ndarray,
):
    """
    Apply group-wise split by battery ID and check ID leakage.
    """
    if "ID" not in mtr.columns or "ID" not in mte.columns:
        raise RuntimeError("Meta must contain 'ID' column for group split.")

    test_ids = set(pd.Series(test_ids).astype(str).tolist())

    tr_ids = pd.Series(mtr["ID"]).astype(str).to_numpy()
    te_ids = pd.Series(mte["ID"]).astype(str).to_numpy()

    mask_train = ~pd.Series(tr_ids).isin(test_ids).to_numpy()
    mask_test = pd.Series(te_ids).isin(test_ids).to_numpy()

    Xtr2 = Xtr[mask_train]
    ytr2 = ytr_str[mask_train]
    mtr2 = mtr.loc[mask_train].reset_index(drop=True)

    Xte2 = Xte[mask_test]
    yte2 = yte_str[mask_test]
    mte2 = mte.loc[mask_test].reset_index(drop=True)

    overlap = set(mtr2["ID"].astype(str).unique()) & set(
        mte2["ID"].astype(str).unique()
    )

    if len(overlap) > 0:
        raise RuntimeError(
            f"ID leakage still exists after split. "
            f"Overlap examples: {list(sorted(overlap))[:10]}"
        )

    return Xtr2, ytr2, mtr2, Xte2, yte2, mte2