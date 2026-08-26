# utils/cache.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd


def ensure_dir(*paths: str) -> None:
    """
    Create directories if they do not exist.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def _hash_dict(d: dict) -> str:
    """
    Generate a short hash for a dictionary.
    """
    s = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def get_cache_tag(prefix: str, **kwargs) -> str:
    """
    Generate a cache tag from a prefix and keyword arguments.
    """
    return f"{prefix}_{_hash_dict(kwargs)}"


def save_cache(
    cache_dir: str,
    tag: str,
    X: np.ndarray,
    y_str: np.ndarray,
    meta: pd.DataFrame,
) -> None:
    """
    Save feature matrix, string labels and metadata to cache.
    """
    ensure_dir(cache_dir)

    np.savez_compressed(
        os.path.join(cache_dir, f"{tag}.npz"),
        X=X.astype(np.float32),
        y_str=y_str.astype(str),
    )

    try:
        meta.to_parquet(
            os.path.join(cache_dir, f"{tag}.parquet"),
            index=False,
        )
    except Exception:
        meta.to_csv(
            os.path.join(cache_dir, f"{tag}.csv"),
            index=False,
            encoding="utf-8-sig",
        )


def load_cache(
    cache_dir: str,
    tag: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load cached feature matrix, string labels and metadata.
    """
    obj = np.load(
        os.path.join(cache_dir, f"{tag}.npz"),
        allow_pickle=False,
    )

    X = obj["X"]
    y_str = obj["y_str"]

    parquet_path = os.path.join(cache_dir, f"{tag}.parquet")
    csv_path = os.path.join(cache_dir, f"{tag}.csv")

    if os.path.exists(parquet_path):
        meta = pd.read_parquet(parquet_path)
    else:
        meta = pd.read_csv(csv_path)

    return X, y_str, meta


def load_or_build_cache(
    cache_dir: str,
    prefix: str,
    build_fn,
    build_kwargs: dict,
):
    """
    Load cached data if available; otherwise build and cache it.
    """
    tag = get_cache_tag(prefix, **build_kwargs)

    npz_path = os.path.join(cache_dir, f"{tag}.npz")
    parquet_path = os.path.join(cache_dir, f"{tag}.parquet")
    csv_path = os.path.join(cache_dir, f"{tag}.csv")

    cache_exists = os.path.exists(npz_path) and (
        os.path.exists(parquet_path) or os.path.exists(csv_path)
    )

    if cache_exists:
        X, y_str, meta = load_cache(cache_dir, tag)
        return X, y_str, meta, tag, True

    X, y_str, meta = build_fn(**build_kwargs)
    save_cache(cache_dir, tag, X, y_str, meta)

    return X, y_str, meta, tag, False


def drop_nan_inf_rows(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    name: str = "",
):
    """
    Drop rows containing NaN or infinite values in X.
    """
    if X.size == 0:
        return X, y, meta

    bad = np.any(~np.isfinite(X), axis=1)

    if bad.any():
        if name:
            print(f"[WARN] Dropping {int(bad.sum())} invalid rows from {name}.")
        else:
            print(f"[WARN] Dropping {int(bad.sum())} invalid rows.")

        return X[~bad], y[~bad], meta.loc[~bad].reset_index(drop=True)

    return X, y, meta