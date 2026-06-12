# -*- coding: utf-8 -*-
"""
suppfig1_soh_train_test_per_material.py

Supplementary Figure 1:
SOH train-test distribution by material-capacity group.

This script uses the proposed_framework result structure.

Data:
    /data

Split:
    results/proposed_framework/splits/testIDs_seed42_frac0.2.txt

Output:
    results/figures/supp/suppfig1

Notes:
1. Raw Excel data are used because this figure shows train/test SOH distributions,
   not model prediction results.
2. The saved ID split from proposed_framework is used to ensure consistency.
3. Each material-capacity group is saved as one figure with axes, title, and legend.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# =========================================================
# Project paths
# =========================================================
PROJECT_ROOT = Path.cwd()

DATA_ROOT = PROJECT_ROOT / "data"

SAVE_DIR = PROJECT_ROOT / "results" / "figures" / "supp" / "suppfig1"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FILE = (
    PROJECT_ROOT
    / "results"
    / "proposed_framework"
    / "splits"
    / "testIDs_seed42_frac0.2.txt"
)

AUTO_REBUILD_SPLIT_IF_MISSING = True

SEED = 42
TEST_ID_FRAC = 0.2
TEST_ID_COUNT = 0


# =========================================================
# Data config
# =========================================================
PULSE_WIDTH_MS = 30

TRAIN_SOC_LIST = list(range(5, 90, 5))
TEST_SHEET_NAME = "SOC TEST RANDOM"

ID_COL = "ID"
SOH_COL = "SOH"

FOLDERS = [
    "10Ah LMO",
    "15Ah NMC",
    "21Ah NMC",
    "24Ah LMO",
    "25Ah LMO",
    "26Ah LMO",
    "35Ah LFP",
    "68Ah LFP",
]


# =========================================================
# Plot config
# =========================================================
COLOR_TRAIN_BAR = "#7FA8C9"
COLOR_TEST_BAR = "#D9A76C"

COLOR_TRAIN_KDE = "#3F6F8F"
COLOR_TEST_KDE = "#A66A2A"

GLOBAL_ALPHA = 0.42
KDE_LINEWIDTH = 6
FIGSIZE = (8, 6)
DPI = 300
N_BINS = 25


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": DPI,
})


# =========================================================
# Data loading helpers
# =========================================================
def parse_folder(folder: str) -> tuple[str, str]:
    parts = folder.split(" ")
    capacity = parts[0]
    material = parts[1]

    material_capacity = f"{material}_{capacity}"
    file_name = f"{material_capacity}_W_{PULSE_WIDTH_MS}.xlsx"

    return material_capacity, file_name


def find_soh_column(df: pd.DataFrame):
    for c in df.columns:
        if str(c).strip() == SOH_COL:
            return c

    for c in df.columns:
        c_norm = str(c).strip().lower().replace(" ", "")
        if c_norm in ["soh", "soh(%)", "soh/%", "soh_percent", "sohpercent"]:
            return c

    return None


def normalize_soh_to_percent(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) > 0 and np.nanmax(values) <= 1.5:
        values = values * 100.0

    return values


def read_one_sheet_soh(
    file_path: Path,
    sheet_name: str,
    material_capacity: str,
    split_source: str,
) -> pd.DataFrame:
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    if ID_COL not in df.columns:
        raise KeyError(
            f"{file_path} | sheet={sheet_name}: missing ID column '{ID_COL}'. "
            f"Available columns: {list(df.columns)}"
        )

    soh_col = find_soh_column(df)

    if soh_col is None:
        raise KeyError(
            f"{file_path} | sheet={sheet_name}: missing SOH column. "
            f"Available columns: {list(df.columns)}"
        )

    out = df[[ID_COL, soh_col]].copy()
    out = out.rename(columns={soh_col: SOH_COL})

    out[ID_COL] = out[ID_COL].astype(str)
    out[SOH_COL] = pd.to_numeric(out[SOH_COL], errors="coerce")
    out = out.dropna(subset=[ID_COL, SOH_COL]).copy()

    out[SOH_COL] = normalize_soh_to_percent(out[SOH_COL].to_numpy(dtype=float))

    out["material_capacity"] = material_capacity
    out["source_sheet"] = sheet_name
    out["split_source"] = split_source

    return out


def build_raw_train_meta() -> pd.DataFrame:
    all_dfs = []
    missing_files = []
    failed_sheets = 0

    for folder in FOLDERS:
        material_capacity, file_name = parse_folder(folder)
        file_path = DATA_ROOT / folder / file_name

        if not file_path.exists():
            missing_files.append(str(file_path))
            continue

        for soc in TRAIN_SOC_LIST:
            sheet = f"SOC{soc}"

            try:
                df = read_one_sheet_soh(
                    file_path=file_path,
                    sheet_name=sheet,
                    material_capacity=material_capacity,
                    split_source="raw_train",
                )

                if len(df) > 0:
                    all_dfs.append(df)

            except Exception:
                failed_sheets += 1

    if len(all_dfs) == 0:
        raise RuntimeError(
            "No raw train data were loaded. Please check DATA_ROOT and train sheet names."
        )

    if missing_files:
        print(f"[WARN] Missing train files: {len(missing_files)}")

    if failed_sheets > 0:
        print(f"[WARN] Failed train sheets: {failed_sheets}")

    return pd.concat(all_dfs, axis=0, ignore_index=True)


def build_raw_test_meta() -> pd.DataFrame:
    all_dfs = []
    missing_files = []
    failed_sheets = 0

    for folder in FOLDERS:
        material_capacity, file_name = parse_folder(folder)
        file_path = DATA_ROOT / folder / file_name

        if not file_path.exists():
            missing_files.append(str(file_path))
            continue

        try:
            df = read_one_sheet_soh(
                file_path=file_path,
                sheet_name=TEST_SHEET_NAME,
                material_capacity=material_capacity,
                split_source="raw_test",
            )

            if len(df) > 0:
                all_dfs.append(df)

        except Exception:
            failed_sheets += 1

    if len(all_dfs) == 0:
        raise RuntimeError(
            "No raw test data were loaded. Please check DATA_ROOT and test sheet name."
        )

    if missing_files:
        print(f"[WARN] Missing test files: {len(missing_files)}")

    if failed_sheets > 0:
        print(f"[WARN] Failed test sheets: {failed_sheets}")

    return pd.concat(all_dfs, axis=0, ignore_index=True)


# =========================================================
# Split helpers
# =========================================================
def pick_test_ids(
    all_ids,
    test_id_frac: float = 0.2,
    test_id_count: int = 0,
    seed: int = 42,
):
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


def load_or_create_test_ids(mtr_raw: pd.DataFrame, mte_raw: pd.DataFrame):
    if SPLIT_FILE.exists():
        with open(SPLIT_FILE, "r", encoding="utf-8") as f:
            test_ids = [line.strip() for line in f if line.strip()]
        return np.array(test_ids, dtype=object)

    if not AUTO_REBUILD_SPLIT_IF_MISSING:
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_FILE}. "
            "Please run proposed_framework first or enable AUTO_REBUILD_SPLIT_IF_MISSING."
        )

    all_ids = pd.concat([mtr_raw[ID_COL], mte_raw[ID_COL]], axis=0).astype(str).to_numpy()

    test_ids = pick_test_ids(
        all_ids=all_ids,
        test_id_frac=TEST_ID_FRAC,
        test_id_count=TEST_ID_COUNT,
        seed=SEED,
    )

    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_FILE, "w", encoding="utf-8") as f:
        for _id in test_ids:
            f.write(str(_id) + "\n")

    print(f"[WARN] Split file was missing and has been rebuilt: {SPLIT_FILE}")

    return test_ids


def apply_saved_id_split(
    mtr_raw: pd.DataFrame,
    mte_raw: pd.DataFrame,
    test_ids,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_ids = set(pd.Series(test_ids).astype(str).tolist())

    tr_ids = pd.Series(mtr_raw[ID_COL]).astype(str)
    te_ids = pd.Series(mte_raw[ID_COL]).astype(str)

    train_mask = ~tr_ids.isin(test_ids).to_numpy()
    test_mask = te_ids.isin(test_ids).to_numpy()

    mtr = mtr_raw.loc[train_mask].reset_index(drop=True)
    mte = mte_raw.loc[test_mask].reset_index(drop=True)

    mtr["split"] = "train"
    mte["split"] = "test"

    overlap = set(mtr[ID_COL].astype(str).unique()) & set(mte[ID_COL].astype(str).unique())

    if len(overlap) > 0:
        raise RuntimeError(f"ID leakage detected. Example overlap IDs: {list(sorted(overlap))[:10]}")

    if len(mtr) == 0 or len(mte) == 0:
        raise RuntimeError(
            f"Empty split after applying test IDs: train={len(mtr)}, test={len(mte)}. "
            "Please check whether the split file matches the current Excel data."
        )

    return mtr, mte


# =========================================================
# Plot helpers
# =========================================================
def safe_filename(name: str) -> str:
    return str(name).replace("/", "_").replace("\\", "_").replace(" ", "_")


def natural_group_key(name: str):
    s = str(name)
    m = re.match(r"([A-Za-z]+)[_\-\s]*(\d+(?:\.\d+)?)\s*Ah", s, re.IGNORECASE)

    if m:
        material = m.group(1).upper()
        cap = float(m.group(2))
        return material, cap, s

    return "ZZZ", 1e9, s


def plot_hist_kde_density(
    ax,
    values,
    bins,
    bar_color,
    kde_color,
    hist_label,
    kde_label,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return

    ax.hist(
        values,
        bins=bins,
        density=True,
        color=bar_color,
        alpha=GLOBAL_ALPHA,
        edgecolor="none",
        label=hist_label,
    )

    if len(values) >= 2 and np.nanstd(values) > 1e-12:
        kde = gaussian_kde(values)
        xs = np.linspace(bins[0], bins[-1], 400)

        ax.plot(
            xs,
            kde(xs),
            color=kde_color,
            linewidth=KDE_LINEWIDTH,
            label=kde_label,
            zorder=10,
        )


def _calc_ymax_for_group(df_g: pd.DataFrame, global_xlim) -> float:
    y_max = 0.0
    bins = np.linspace(global_xlim[0], global_xlim[1], N_BINS + 1)

    for split in ["train", "test"]:
        values = df_g.loc[df_g["split"] == split, SOH_COL].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        hist_density, _ = np.histogram(values, bins=bins, density=True)

        if len(hist_density) > 0 and np.isfinite(hist_density).any():
            y_max = max(y_max, np.nanmax(hist_density))

        if len(values) >= 2 and np.nanstd(values) > 1e-12:
            kde = gaussian_kde(values)
            xs = np.linspace(global_xlim[0], global_xlim[1], 400)
            y_max = max(y_max, np.nanmax(kde(xs)))

    if y_max <= 0 or not np.isfinite(y_max):
        y_max = 1.0

    return y_max


def compute_mixed_ylim(data: pd.DataFrame, global_xlim):
    groups = sorted(data["material_capacity"].unique(), key=natural_group_key)

    nmc_groups = [g for g in groups if "NMC" in g]
    other_groups = [g for g in groups if "NMC" not in g]

    y_max_global = 0.0
    for g in other_groups:
        df_g = data.loc[data["material_capacity"] == g].copy()
        y_max_global = max(y_max_global, _calc_ymax_for_group(df_g, global_xlim))

    if y_max_global <= 0 or not np.isfinite(y_max_global):
        y_max_global = 1.0

    global_ylim = (0.0, y_max_global * 1.15)

    nmc_ylim_dict = {}
    for g in nmc_groups:
        df_g = data.loc[data["material_capacity"] == g].copy()
        y_max = _calc_ymax_for_group(df_g, global_xlim)

        if y_max <= 0 or not np.isfinite(y_max):
            y_max = 1.0

        nmc_ylim_dict[g] = (0.0, y_max * 1.15)

    return global_ylim, nmc_ylim_dict


def save_plot_for_group(
    df_group: pd.DataFrame,
    material_capacity: str,
    x_limit,
    y_limit,
):
    train_values = df_group.loc[df_group["split"] == "train", SOH_COL].to_numpy(dtype=float)
    test_values = df_group.loc[df_group["split"] == "test", SOH_COL].to_numpy(dtype=float)

    train_values = train_values[np.isfinite(train_values)]
    test_values = test_values[np.isfinite(test_values)]

    if len(train_values) == 0 and len(test_values) == 0:
        return

    bins = np.linspace(x_limit[0], x_limit[1], N_BINS + 1)
    file_prefix = safe_filename(material_capacity)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI, facecolor="none")
    ax.set_facecolor("none")

    plot_hist_kde_density(
        ax=ax,
        values=train_values,
        bins=bins,
        bar_color=COLOR_TRAIN_BAR,
        kde_color=COLOR_TRAIN_KDE,
        hist_label="Train histogram",
        kde_label="Train KDE",
    )

    plot_hist_kde_density(
        ax=ax,
        values=test_values,
        bins=bins,
        bar_color=COLOR_TEST_BAR,
        kde_color=COLOR_TEST_KDE,
        hist_label="Test histogram",
        kde_label="Test KDE",
    )

    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)

    ax.set_title(f"{material_capacity} SOH Train-Test Distribution", fontsize=14)
    ax.set_xlabel("SOH (%)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)

    ax.legend(frameon=False, fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_path = SAVE_DIR / f"{file_prefix}_SOH_train_test.png"

    plt.savefig(
        save_path,
        dpi=DPI,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    print("[INFO] Generating Supplementary Figure 1 SOH distributions...")

    mtr_raw = build_raw_train_meta()
    mte_raw = build_raw_test_meta()

    test_ids = load_or_create_test_ids(mtr_raw, mte_raw)
    mtr, mte = apply_saved_id_split(mtr_raw, mte_raw, test_ids)

    data = pd.concat([mtr, mte], axis=0, ignore_index=True)

    before_n = len(data)
    data = data[np.isfinite(data[SOH_COL].to_numpy(dtype=float))].copy()
    after_n = len(data)

    if after_n < before_n:
        print(f"[WARN] Removed invalid SOH rows: {before_n - after_n}")

    all_values = data[SOH_COL].to_numpy(dtype=float)
    all_values = all_values[np.isfinite(all_values)]

    x_min, x_max = np.nanmin(all_values), np.nanmax(all_values)
    margin = (x_max - x_min) * 0.05

    if margin <= 0 or not np.isfinite(margin):
        margin = 1.0

    global_xlim = (x_min - margin, x_max + margin)

    global_ylim, nmc_ylim_dict = compute_mixed_ylim(data, global_xlim)

    raw_path = SAVE_DIR / "suppfig1_SOH_train_test_raw_values.csv"
    data.to_csv(raw_path, index=False, encoding="utf-8-sig")

    summary = (
        data.groupby(["material_capacity", "split"])[SOH_COL]
        .agg(["count", "min", "max", "mean", "median"])
        .reset_index()
    )

    summary_path = SAVE_DIR / "suppfig1_SOH_train_test_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    groups = sorted(data["material_capacity"].unique(), key=natural_group_key)

    for g in groups:
        df_g = data.loc[data["material_capacity"] == g].copy()

        if "NMC" in g:
            y_limit = nmc_ylim_dict.get(g, global_ylim)
        else:
            y_limit = global_ylim

        save_plot_for_group(df_g, g, global_xlim, y_limit)

    print("[DONE] Supplementary Figure 1 SOH distributions generated.")
    print(f"[SAVED] Figures and CSV files: {SAVE_DIR}")
    print(f"[SAVED] Raw values: {raw_path}")
    print(f"[SAVED] Summary: {summary_path}")


if __name__ == "__main__":
    main()