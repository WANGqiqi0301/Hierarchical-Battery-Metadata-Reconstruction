# -*- coding: utf-8 -*-
"""
plot_fig3e_tsne_material_raw_vs_embedded.py

Generate Figure 3e: Raw and Model Material t-SNE
Reproduces old zzz_tsne_triple_compare.py logic as closely as possible.
"""
import os
import sys
from pathlib import Path# -*- coding: utf-8 -*-
"""
plot_fig3e_tsne_material_raw_vs_embedded.py

Generate Figure 3e: Raw and Model Material t-SNE.

This version:
1) Uses proposed_framework functions only.
2) Does NOT modify proposed_framework.
3) Reorders the loaded data inside this plotting script to match the old
   zzz_tsne_triple_compare.py loading order:
      for soc in SOC_LIST:
          for pulse in PULSE_LIST:
              append data
4) Uses cache properly.
5) Fixes path issues by resolving paths relative to code/.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# =============================================================================
# Project paths
# =============================================================================
# Expected file location:
# code/figures/plot_fig3e_tsne_material_raw_vs_embedded.py
#
# PROJECT_ROOT = code/
# DATA_ROOT    = code/data
# RESULTS_ROOT = code/results
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import proposed_framework.run_proposed_framework as M


# =============================================================================
# Config
# =============================================================================
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
EXP_DIR = RESULTS_ROOT / "proposed_framework"

SAVE_DIR = RESULTS_ROOT / "figures" / "main" / "fig3e"

# Use a new cache name to avoid loading previous wrong-order cache.
CACHE_FILE = SAVE_DIR / "tsne_cache_fig3e_proposed_old_order.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SAMPLES = 3000
RANDOM_SEED = 42

SOC_LIST = list(range(5, 90, 5))
PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

SOC_COL = "SOC"
SOH_COL = "SOH"

USE_CACHE = True

CHECKPOINT_CANDIDATES = [
    EXP_DIR / "checkpoints" / "finetune" / "best.pt",
    EXP_DIR / "checkpoints" / "stage2_soh" / "best.pt",
]

SAVE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Utilities
# =============================================================================
def print_path_info() -> None:
    print(f"[PATH] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[PATH] DATA_ROOT    = {DATA_ROOT}")
    print(f"[PATH] EXP_DIR      = {EXP_DIR}")
    print(f"[PATH] SAVE_DIR     = {SAVE_DIR}")
    print(f"[PATH] CACHE_FILE   = {CACHE_FILE}")


def find_existing_checkpoint() -> Path:
    for p in CHECKPOINT_CANDIDATES:
        if p.exists():
            return p

    msg = ["Checkpoint not found. Checked paths:"]
    for p in CHECKPOINT_CANDIDATES:
        msg.append(f"  - {p}")
    raise FileNotFoundError("\n".join(msg))


def find_pulse_column(meta) -> str:
    """
    Try to find the pulse-width column in meta.

    Different versions may use slightly different names.
    """
    candidates = [
        "pulse_width_ms",
        "pulse_width",
        "PulseWidth",
        "PULSE_WIDTH_MS",
        "pulse_ms",
        "pulse",
        "pt",
        "PT",
    ]

    for c in candidates:
        if c in meta.columns:
            return c

    msg = [
        "Cannot find pulse-width column in meta.",
        "Available meta columns:",
    ]
    msg.extend([f"  - {c}" for c in meta.columns])
    msg.append("")
    msg.append("Please check which column stores pulse width in proposed_framework meta.")
    raise KeyError("\n".join(msg))


def values_match(a, b) -> np.ndarray:
    """
    Robust comparison for numeric-looking values.
    """
    try:
        return np.isclose(np.asarray(a, dtype=float), float(b))
    except Exception:
        return np.asarray(a).astype(str) == str(b)


# =============================================================================
# Feature builder
# =============================================================================
def build_3ch_5x8_from_u41(u: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert raw U1-U41 vector to 3-channel 5x8 representation.

    Channel 1:
        U2-U41 reshaped to 5x8

    Channel 2:
        voltage increments:
        d[0]  = U2 - U1
        d[1:] = U3-U2, ..., U41-U40

    Channel 3:
        repeated U1 baseline
    """
    if not np.isfinite(u).all():
        return None

    u1 = float(u[0])

    u2_41 = u[1:]
    ch1 = u2_41.reshape(5, 8)

    d = np.empty(40)
    d[0] = u[1] - u[0]
    d[1:] = u[2:] - u[1:-1]
    ch2 = d.reshape(5, 8)

    ch3 = np.full((5, 8), u1)

    x = np.stack([ch1, ch2, ch3], axis=0)

    if not np.isfinite(x).all():
        return None

    return x


# =============================================================================
# Data loading using proposed_framework, then reorder inside this script
# =============================================================================
def load_all_data_direct() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Use proposed_framework's data-building function, then reorder the output
    to match the old zzz_tsne_triple_compare.py loading order.

    Old order:
        for soc in SOC_LIST:
            for pulse in PULSE_LIST:
                append group
    """
    print("[DATA] Rebuilding raw U1-U41 using proposed_framework...")
    print(f"[DATA] DATA_ROOT = {DATA_ROOT}")

    out = M.build_train_mix_soc_mix_pt(
        data_root=str(DATA_ROOT),
        soc_list=SOC_LIST,
        pulse_list=PULSE_LIST,
        u_start=1,
        u_end=41,
        drop_first_class=True,
    )

    X, y, meta = out[0], out[1], out[2]

    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]
    meta = meta.loc[mask].reset_index(drop=True)

    pulse_col = find_pulse_column(meta)

    print(f"[DATA] SOC column   = {SOC_COL}")
    print(f"[DATA] SOH column   = {SOH_COL}")
    print(f"[DATA] Pulse column = {pulse_col}")

    # -------------------------------------------------------------------------
    # Reorder data to match old code:
    #   for soc in SOC_LIST:
    #       for pt in PULSE_LIST:
    #           append this group
    # -------------------------------------------------------------------------
    order_indices = []

    for soc in SOC_LIST:
        soc_mask = values_match(meta[SOC_COL].values, soc)

        for pt in PULSE_LIST:
            pt_mask = values_match(meta[pulse_col].values, pt)
            group_idx = np.flatnonzero(soc_mask & pt_mask)

            if len(group_idx) > 0:
                order_indices.append(group_idx)

            print(f"[ORDER] SOC={soc:>2}, pulse={pt:>4}, n={len(group_idx)}")

    if len(order_indices) == 0:
        raise RuntimeError("No data found after SOC × pulse reordering.")

    order_indices = np.concatenate(order_indices)

    X = X[order_indices]
    y = y[order_indices]
    meta = meta.iloc[order_indices].reset_index(drop=True)

    soc = meta[SOC_COL].values
    soh = meta[SOH_COL].values

    print(f"[DATA] Reordered X shape: {X.shape}, y shape: {y.shape}")

    return X, y, soc, soh


# =============================================================================
# Model loading using proposed_framework
# =============================================================================
def load_model(num_classes: int, checkpoint_path: Path) -> torch.nn.Module:
    print(f"[MODEL] Using checkpoint: {checkpoint_path}")

    model = M.Hier3HeadModel(
        num_classes=num_classes,
        width=32,
        blocks=4,
        use_pt_as_feature=True,
    ).to(DEVICE)

    ckpt = torch.load(str(checkpoint_path), map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model


# =============================================================================
# Feature extraction
# =============================================================================
def extract_raw_and_model_features(
    X: np.ndarray,
    y: np.ndarray,
    soc: np.ndarray,
    soh: np.ndarray,
    checkpoint_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample MAX_SAMPLES records, build 3-channel inputs, and extract:
    - model encoder features
    - raw U1-U41 features
    """
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(X), size=min(MAX_SAMPLES, len(X)), replace=False)

    X = X[idx]
    y = y[idx]
    soc = soc[idx]
    soh = soh[idx]

    print(f"[SAMPLE] Sampled n = {len(X)}")
    print(f"[SAMPLE] First row sum  = {np.sum(X[0]):.8f}")
    print(f"[SAMPLE] First row mean = {np.mean(X[0]):.8f}")
    print(f"[SAMPLE] First 10 labels = {y[:10]}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = load_model(
        num_classes=len(le.classes_),
        checkpoint_path=checkpoint_path,
    )

    f_model = []
    f_raw = []
    labels = []
    socs = []
    sohs = []

    for i in range(len(X)):
        x3 = build_3ch_5x8_from_u41(X[i])

        if x3 is None:
            continue

        x3_t = torch.tensor(x3, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat_m = model.encoder(x3_t).cpu().numpy().squeeze()

        if not np.isfinite(feat_m).all():
            continue

        f_model.append(feat_m)
        f_raw.append(X[i])
        labels.append(y_enc[i])
        socs.append(soc[i])
        sohs.append(soh[i])

    print(f"[TSNE] Valid samples: {len(f_model)}")

    return (
        np.array(f_model),
        np.array(f_raw),
        np.array(labels),
        np.array(socs),
        np.array(sohs),
    )


# =============================================================================
# t-SNE
# =============================================================================
def run_tsne_logic(
    features: np.ndarray,
    use_pca: bool = True,
    perp: int = 30,
    lr: int = 100,
    n_iter: int = 2000,
) -> np.ndarray:
    data = features.copy()

    if use_pca and data.shape[1] > 30:
        print("  -> PCA reduction to 30 dims...")
        # Keep this identical to old code: no explicit random_state in PCA.
        data = PCA(n_components=30).fit_transform(data)

    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            learning_rate=lr,
            n_iter=n_iter,
            random_state=RANDOM_SEED,
            init="pca",
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            learning_rate=lr,
            max_iter=n_iter,
            random_state=RANDOM_SEED,
            init="pca",
        )

    return tsne.fit_transform(data)


# =============================================================================
# Cache
# =============================================================================
def load_cache(cache_path: Path):
    print(f"[CACHE] Loading cache: {cache_path}")

    d = np.load(str(cache_path), allow_pickle=True)

    Z_model = d["Z_model"]
    Z_raw = d["Z_raw"]
    labels = d["labels"]
    socs = d["socs"]
    sohs = d["sohs"]

    return Z_model, Z_raw, labels, socs, sohs


def save_cache(
    cache_path: Path,
    Z_model: np.ndarray,
    Z_raw: np.ndarray,
    labels: np.ndarray,
    socs: np.ndarray,
    sohs: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(cache_path),
        Z_model=Z_model,
        Z_raw=Z_raw,
        labels=labels,
        socs=socs,
        sohs=sohs,
    )

    print(f"[CACHE] Saved cache: {cache_path}")


# =============================================================================
# Plotting
# =============================================================================
def plot_material_full(
    Z: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        Z[:, 0],
        Z[:, 1],
        c=labels,
        cmap="tab10",
        s=30,
        alpha=0.7,
    )
    plt.title(title)
    plt.colorbar(sc)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=600, transparent=True)
    plt.close()


def plot_material_points(
    Z: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(
        Z[:, 0],
        Z[:, 1],
        c=labels,
        cmap="tab10",
        s=30,
        alpha=0.7,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=600, bbox_inches="tight", transparent=True)
    plt.close()


def plot_material_colorbar(
    labels: np.ndarray,
    out_path: Path,
) -> None:
    norm = plt.Normalize(vmin=np.min(labels), vmax=np.max(labels))
    sm = plt.cm.ScalarMappable(cmap="tab10", norm=norm)
    sm.set_array([])

    fig_cbar, ax_cbar = plt.subplots(figsize=(1.5, 5))
    ax_cbar.axis("off")

    cbar = fig_cbar.colorbar(sm, ax=ax_cbar, orientation="vertical")
    cbar.set_label("")
    cbar.ax.yaxis.set_major_locator(plt.NullLocator())
    cbar.ax.xaxis.set_major_locator(plt.NullLocator())

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig_cbar)


def save_all_plots(
    Z_model: np.ndarray,
    Z_raw: np.ndarray,
    labels: np.ndarray,
) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    plot_material_full(
        Z_model,
        labels,
        SAVE_DIR / "model_tsne_label_full.png",
        "t-SNE Model (Material)",
    )

    plot_material_full(
        Z_raw,
        labels,
        SAVE_DIR / "raw_tsne_label_full.png",
        "t-SNE Raw (Material)",
    )

    plot_material_points(
        Z_model,
        labels,
        SAVE_DIR / "model_tsne_label_points.png",
    )

    plot_material_points(
        Z_raw,
        labels,
        SAVE_DIR / "raw_tsne_label_points.png",
    )

    plot_material_colorbar(
        labels,
        SAVE_DIR / "material_label_colorbar.png",
    )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    print_path_info()

    if USE_CACHE and CACHE_FILE.exists():
        Z_model, Z_raw, labels, socs, sohs = load_cache(CACHE_FILE)

    else:
        if USE_CACHE:
            print("[CACHE] No cache found. Rebuilding t-SNE...")
        else:
            print("[CACHE] USE_CACHE=False. Rebuilding t-SNE...")

        checkpoint_path = find_existing_checkpoint()

        X, y, soc, soh = load_all_data_direct()

        f_model, f_raw, labels, socs, sohs = extract_raw_and_model_features(
            X=X,
            y=y,
            soc=soc,
            soh=soh,
            checkpoint_path=checkpoint_path,
        )

        print(">> Running Model t-SNE (PCA 30, Perp 30)...")
        Z_model = run_tsne_logic(
            f_model,
            use_pca=True,
            perp=30,
            lr=100,
            n_iter=2000,
        )

        print(">> Running Raw t-SNE (PCA 30, Perp 30)...")
        Z_raw = run_tsne_logic(
            f_raw,
            use_pca=True,
            perp=30,
            lr=100,
            n_iter=2000,
        )

        save_cache(
            CACHE_FILE,
            Z_model=Z_model,
            Z_raw=Z_raw,
            labels=labels,
            socs=socs,
            sohs=sohs,
        )

    print("[PLOT] Saving Figure 3e images...")
    save_all_plots(Z_model, Z_raw, labels)

    print(f"✅ Done. Figure 3e images saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import proposed_framework.run_proposed_framework as M

# --- Config ---
DATA_ROOT = str(PROJECT_ROOT.parent / "data")
EXP_DIR = "results/proposed_framework"
SAVE_DIR = "results/figures/main/fig3e"
CACHE_FILE = os.path.join(SAVE_DIR, "tsne_cache_fig3e.npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SAMPLES = 3000
RANDOM_SEED = 42
SOC_LIST = list(range(5, 90, 5))
PULSE_LIST = [30,50,70,100,300,500,700,1000,3000,5000]
SOC_COL = "SOC"
SOH_COL = "SOH"

os.makedirs(SAVE_DIR, exist_ok=True)

# --- Feature builder ---
def build_3ch_5x8_from_u41(u):
    if not np.isfinite(u).all():
        return None
    u1 = float(u[0])
    u2_41 = u[1:]
    ch1 = u2_41.reshape(5, 8)
    d = np.empty(40)
    d[0] = u[1]-u[0]
    d[1:] = u[2:]-u[1:-1]
    ch2 = d.reshape(5,8)
    ch3 = np.full((5,8), u1)
    x = np.stack([ch1,ch2,ch3],axis=0)
    return x if np.isfinite(x).all() else None

# --- Load all data ---
def load_all_data_direct():
    print("[DATA] Rebuilding raw U1-U41 from original data folder...")
    out = M.build_train_mix_soc_mix_pt(
        data_root=DATA_ROOT,
        soc_list=SOC_LIST,
        pulse_list=PULSE_LIST,
        u_start=1,
        u_end=41,
        drop_first_class=True,
    )
    X, y, meta = out[0], out[1], out[2]
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]
    meta = meta.loc[mask].reset_index(drop=True)
    soc = meta[SOC_COL].values
    soh = meta[SOH_COL].values
    print(f"[DATA] X shape: {X.shape}, y shape: {y.shape}")
    return X, y, soc, soh

# --- Load model ---
def load_model(num_classes, checkpoint_path):
    model = M.Hier3HeadModel(num_classes=num_classes, width=32, blocks=4, use_pt_as_feature=True).to(DEVICE)
    ckpt = torch.load(checkpoint_path,map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

# --- Extract features ---
def extract_raw_and_model_features(X,y,soc,soh,checkpoint_path):
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(X), size=min(MAX_SAMPLES,len(X)), replace=False)
    X, y, soc, soh = X[idx], y[idx], soc[idx], soh[idx]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    model = load_model(len(le.classes_), checkpoint_path)
    f_model,f_raw,labels,socs,sohs = [],[],[],[],[]
    for i in range(len(X)):
        x3 = build_3ch_5x8_from_u41(X[i])
        if x3 is None:
            continue
        x3_t = torch.tensor(x3,dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat_m = model.encoder(x3_t).cpu().numpy().squeeze()
        if not np.isfinite(feat_m).all():
            continue
        f_model.append(feat_m)
        f_raw.append(X[i])
        labels.append(y_enc[i])
        socs.append(soc[i])
        sohs.append(soh[i])
    print(f"[TSNE] Valid samples: {len(f_model)}")
    return np.array(f_model), np.array(f_raw), np.array(labels), np.array(socs), np.array(sohs)

# --- t-SNE ---
def run_tsne_logic(features, use_pca=True, perp=30, lr=100, n_iter=2000):
    data = features.copy()

    if use_pca and data.shape[1] > 30:
        print("  -> PCA reduction to 30 dims...")
        data = PCA(n_components=30).fit_transform(data)

    # Keep the old t-SNE logic, but support both old and new sklearn versions.
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            learning_rate=lr,
            n_iter=n_iter,
            random_state=RANDOM_SEED,
            init="pca",
        )
    except TypeError:
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            learning_rate=lr,
            max_iter=n_iter,
            random_state=RANDOM_SEED,
            init="pca",
        )

    return tsne.fit_transform(data)

# --- Plot ---
def plot_material(Z,labels,out_path,title):
    plt.figure(figsize=(6,5))
    sc = plt.scatter(Z[:,0],Z[:,1],c=labels,cmap='tab10',s=30,alpha=0.7)
    plt.title(title)
    plt.colorbar(sc)
    plt.tight_layout()
    plt.savefig(out_path,dpi=600,transparent=True)
    plt.close()

# --- Main ---
def main():
    X, y, soc, soh = load_all_data_direct()
    checkpoint_path = os.path.join(EXP_DIR,"checkpoints","finetune","best.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(EXP_DIR,"checkpoints","stage2_soh","best.pt")
    print(f"[MODEL] Using checkpoint: {checkpoint_path}")
    f_model, f_raw, labels, socs, sohs = extract_raw_and_model_features(X,y,soc,soh,checkpoint_path)
    print(">> Running t-SNE for Model features...")
    Z_model = run_tsne_logic(f_model)
    print(">> Running t-SNE for Raw features...")
    Z_raw = run_tsne_logic(f_raw)
    os.makedirs(SAVE_DIR,exist_ok=True)
    plot_material(Z_model, labels, os.path.join(SAVE_DIR,"model_tsne_label_full.png"), "t-SNE Model (Material)")
    plot_material(Z_raw, labels, os.path.join(SAVE_DIR,"raw_tsne_label_full.png"), "t-SNE Raw (Material)")
    print(f"✅ Figure 3e saved to {SAVE_DIR}")

if __name__=="__main__":
    main()