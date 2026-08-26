# utils/data_loader.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

SOCSpec = Union[int, str]
# int -> "SOC85"
# str -> "TEST_RANDOM", "SOC TEST RANDOM", "ALL", "SOC ALL", etc.


@dataclass(frozen=True)
class LoadConfig:
    """
    Configuration for loading pulse-response Excel files.

    Parameters
    ----------
    data_root:
        Root directory containing material-capacity subfolders.

    soc:
        SOC sheet to load. It can be an integer, such as 85, which maps to
        "SOC85", or a string such as "TEST_RANDOM", "SOC TEST RANDOM",
        "ALL", or "SOC ALL".

    pulse_width_ms:
        Pulse width in milliseconds. Only files matching this pulse width are
        loaded.

    u_start, u_end:
        Voltage feature columns to load, from U{u_start} to U{u_end},
        inclusive. The default setting loads U1-U41.

    drop_first_21_only_class:
        Whether to skip files that only contain U1-U21 and do not contain
        U22. This is useful when one class has incomplete voltage features.

    include_soc_in_X:
        Whether to include SOC as an additional input feature. The default is
        False because the main task predicts SOC and should not use SOC as a
        direct input feature.

    verbose:
        Whether to print loading information.
    """

    data_root: str | Path
    soc: SOCSpec
    pulse_width_ms: int
    u_start: int = 1
    u_end: int = 41
    drop_first_21_only_class: bool = True
    include_soc_in_X: bool = False
    verbose: bool = True


# =============================================================================
# Internal helper functions
# =============================================================================

_FILE_RE = re.compile(
    r"^(?P<mat>[A-Za-z0-9]+)_(?P<ah>\d+(?:\.\d+)?)Ah_W_(?P<w>\d+)\.xlsx$"
)


def _u_cols(u_start: int, u_end: int) -> List[str]:
    """
    Generate voltage column names.

    Example
    -------
    _u_cols(1, 41) -> ["U1", "U2", ..., "U41"]
    """
    return [f"U{i}" for i in range(u_start, u_end + 1)]


def _sheet_name(soc: SOCSpec) -> str:
    """
    Convert a SOC specification into the corresponding Excel sheet name.

    Supported examples
    ------------------
    85              -> "SOC85"
    "TEST_RANDOM"   -> "SOC TEST RANDOM"
    "SOC TEST RANDOM" -> "SOC TEST RANDOM"
    "RANDOM"        -> "SOC TEST RANDOM"
    "ALL"           -> "SOC ALL"
    "SOC ALL"       -> "SOC ALL"
    "SOC85"         -> "SOC85"
    """
    if isinstance(soc, int):
        return f"SOC{soc}"

    s = str(soc).strip().upper().replace("_", " ")
    s = re.sub(r"\s+", " ", s)

    if s in {"TEST RANDOM", "SOC TEST RANDOM", "RANDOM", "TEST"}:
        return "SOC TEST RANDOM"

    if s in {"ALL", "SOC ALL"}:
        return "SOC ALL"

    m = re.match(r"^SOC\s*(\d+)$", s)
    if m:
        return f"SOC{int(m.group(1))}"

    return str(soc)


def _sheet_soc_value(sheet: str) -> Optional[float]:
    """
    Infer SOC value from sheet names such as "SOC85".

    Returns
    -------
    float or None
        Returns 85.0 for "SOC85"; returns None for sheets such as
        "SOC TEST RANDOM" or "SOC ALL".
    """
    s = str(sheet).strip().upper().replace("_", " ")
    s = re.sub(r"\s+", " ", s)

    m = re.match(r"^SOC\s*(\d+)$", s)
    if m:
        return float(m.group(1))

    return None


def _parse_filename(path: Path) -> Optional[dict]:
    """
    Parse material, capacity and pulse width from file names.

    Expected format
    ---------------
    LFP_35Ah_W_5000.xlsx

    Returns
    -------
    dict or None
        Example:
        {
            "mat": "LFP",
            "ah": "35",
            "w": "5000"
        }
    """
    m = _FILE_RE.match(path.name)
    return m.groupdict() if m else None


def _label_from_parsed(parsed: dict) -> str:
    """
    Build material-capacity label from parsed file name.

    Example
    -------
    {"mat": "LFP", "ah": "35"} -> "LFP_35Ah"
    """
    return f"{parsed['mat']}_{parsed['ah']}Ah"


def _safe_read_sheet(path: Path, sheet: str) -> Optional[pd.DataFrame]:
    """
    Safely read an Excel sheet.

    Returns None if the sheet does not exist or cannot be read.
    """
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception:
        return None


# =============================================================================
# Public loader
# =============================================================================

def load_pulsebat_classification(
    cfg: LoadConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load pulse-response data under a fixed SOC sheet and pulse width.

    The function scans all material-capacity subfolders under `data_root`,
    loads Excel files matching the requested pulse width, reads the requested
    SOC sheet, and returns voltage features, material-capacity labels and
    metadata.

    Parameters
    ----------
    cfg:
        Data-loading configuration.

    Returns
    -------
    X:
        Feature matrix. By default, it contains voltage columns U1-U41.
        If `include_soc_in_X=True`, SOC is inserted as the first feature column.

    y:
        Material-capacity labels, such as "LFP_35Ah" or "NMC_24Ah".

    meta:
        Sample metadata. It includes available fields such as ID, SOC, SOH,
        Qn, Q, Pt and source file information.
    """
    root = Path(cfg.data_root)

    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {root}")

    target_sheet = _sheet_name(cfg.soc)
    target_w = int(cfg.pulse_width_ms)
    ucols = _u_cols(cfg.u_start, cfg.u_end)

    X_list: List[pd.DataFrame] = []
    y_list: List[pd.Series] = []
    meta_list: List[pd.DataFrame] = []

    subdirs = [d for d in root.iterdir() if d.is_dir()]
    subdirs.sort(key=lambda p: p.name)

    for cls_dir in subdirs:
        for file_path in cls_dir.glob("*.xlsx"):
            parsed = _parse_filename(file_path)

            if parsed is None:
                continue

            if int(parsed["w"]) != target_w:
                continue

            df = _safe_read_sheet(file_path, target_sheet)

            if df is None or df.empty:
                continue

            df.columns = df.columns.astype(str)

            # Optionally skip files that only contain U1-U21.
            if cfg.drop_first_21_only_class:
                cols = set(df.columns.astype(str))
                if "U21" in cols and "U22" not in cols:
                    continue

            # Check whether all requested voltage columns exist.
            missing_cols = [c for c in ucols if c not in df.columns]
            if missing_cols:
                continue

            # -----------------------------------------------------------------
            # Features
            # -----------------------------------------------------------------
            X = df[ucols].copy()

            if cfg.include_soc_in_X:
                if "SOC" in df.columns:
                    soc_col = pd.to_numeric(df["SOC"], errors="coerce")
                else:
                    soc_value = _sheet_soc_value(target_sheet)

                    if soc_value is None:
                        # For random-SOC or all-SOC sheets, SOC cannot be
                        # inferred from the sheet name if the SOC column is
                        # missing.
                        continue

                    soc_col = pd.Series([soc_value] * len(df), index=df.index)

                if soc_col.isna().any():
                    continue

                X.insert(0, "SOC", soc_col.astype(float))

            # -----------------------------------------------------------------
            # Label
            # -----------------------------------------------------------------
            label = _label_from_parsed(parsed)
            y = pd.Series([label] * len(df), name="label")

            # -----------------------------------------------------------------
            # Metadata
            # -----------------------------------------------------------------
            meta_cols = [
                c
                for c in [
                    "File_Name",
                    "Mat",
                    "No.",
                    "No",
                    "ID",
                    "Qn",
                    "Q",
                    "SOH",
                    "Pt",
                    "SOC",
                    "SOCR",
                ]
                if c in df.columns
            ]

            if meta_cols:
                meta = df[meta_cols].copy()
            else:
                meta = pd.DataFrame(index=df.index)

            meta["source_file"] = file_path.name
            meta["source_dir"] = cls_dir.name
            meta["label"] = label
            meta["sheet"] = target_sheet
            meta["pulse_width_ms"] = target_w

            X_list.append(X)
            y_list.append(y)
            meta_list.append(meta)

    if not X_list:
        raise RuntimeError(
            f"No data loaded. Please check the loading configuration: "
            f"sheet='{target_sheet}', pulse_width_ms={target_w}, root='{root}'. "
            f"If include_soc_in_X=True, make sure the SOC column exists, especially "
            f"for random-SOC sheets."
        )

    X_all = pd.concat(X_list, axis=0, ignore_index=True)
    y_all = pd.concat(y_list, axis=0, ignore_index=True)
    meta_all = pd.concat(meta_list, axis=0, ignore_index=True)

    if cfg.verbose:
        print(f"Loaded sheet='{target_sheet}' | X: {X_all.shape}, y: {y_all.shape}")
        print("Class distribution:")
        print(y_all.value_counts())

        if cfg.include_soc_in_X:
            print("SOC is included as the first input feature.")

    return X_all, y_all, meta_all


def preview(
    X: pd.DataFrame,
    y: pd.Series,
    meta: Optional[pd.DataFrame] = None,
    n: int = 5,
) -> None:
    """
    Print a quick preview of loaded features, labels and selected metadata.

    This function is intended for quick inspection only and is not required
    by the training pipeline.
    """
    show = X.head(n).copy()
    show.insert(0, "label", y.head(n).values)

    if meta is not None and not meta.empty:
        preview_cols = [
            c
            for c in [
                "sheet",
                "source_file",
                "pulse_width_ms",
                "Pt",
                "SOC",
                "SOCR",
                "SOH",
                "ID",
                "No.",
                "No",
            ]
            if c in meta.columns
        ]

        for c in reversed(preview_cols):
            show.insert(1, c, meta[c].head(n).values)

    print(show)