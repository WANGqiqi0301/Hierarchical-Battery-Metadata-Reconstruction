# -*- coding: utf-8 -*-
"""
suppfig10_tsne_triple_compare.py

Supplementary Figure 10:
t-SNE comparison of three feature representations:
1. Model encoder features
2. Raw U1-U41 voltage features
3. Manual 3-channel 5x8x3 features flattened to 120 dimensions

Each representation is visualized by:
1. Material-capacity group
2. SOC
3. SOH

This script follows the same proposed_framework-based loading and model logic
as the organized Fig. 3e script.

Input:
    data/
    results/proposed_framework/checkpoints/finetune/best.pt
    or
    results/proposed_framework/checkpoints/stage2_soh/best.pt

Output:
    results/figures/supp/suppfig10

Generated files:
    suppfig10_model_tsne_material.png
    suppfig10_model_tsne_soc.png
    suppfig10_model_tsne_soh.png
    suppfig10_raw_tsne_material.png
    suppfig10_raw_tsne_soc.png
    suppfig10_raw_tsne_soh.png
    suppfig10_manual_tsne_material.png
    suppfig10_manual_tsne_soc.png
    suppfig10_manual_tsne_soh.png
    suppfig10_tsne_cache.npz
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


# =============================================================================
# Project paths
# =============================================================================
PROJECT_ROOT = Path.cwd().resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import proposed_framework.run_proposed_framework as M


DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
EXP_DIR = RESULTS_ROOT / "proposed_framework"

SAVE_DIR = RESULTS_ROOT / "figures" / "supp" / "suppfig10"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = SAVE_DIR / "suppfig10_tsne_cache.npz"

CHECKPOINT_CANDIDATES = [
    EXP_DIR / "checkpoints" / "finetune" / "best.pt",
    EXP_DIR / "checkpoints" / "stage2_soh" / "best.pt",
]


# =============================================================================
# Config
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SAMPLES = 3000
RANDOM_SEED = 42

SOC_LIST = list(range(5, 90, 5))
PULSE_LIST = [30, 50, 70, 100, 300, 500, 700, 1000, 3000, 5000]

SOC_COL = "SOC"
SOH_COL = "SOH"

USE_CACHE = True


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})


# =============================================================================
# Utilities
# =============================================================================
def find_existing_checkpoint() -> Path:
    for path in CHECKPOINT_CANDIDATES:
        if path.exists():
            return path

    message = ["Checkpoint not found. Checked paths:"]
    for path in CHECKPOINT_CANDIDATES:
        message.append(f"  - {path}")

    raise FileNotFoundError("\n".join(message))


def find_pulse_column(meta) -> str:
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

    for col in candidates:
        if col in meta.columns:
            return col

    raise KeyError(
        "Cannot find pulse-width column in meta. "
        f"Available columns: {list(meta.columns)}"
    )


def values_match(a, b) -> np.ndarray:
    try:
        return np.isclose(np.asarray(a, dtype=float), float(b))
    except Exception:
        return np.asarray(a).astype(str) == str(b)


# =============================================================================
# Feature builder
# =============================================================================
def build_3ch_5x8_from_u41(u: np.ndarray) -> Optional[np.ndarray]:
    u = np.asarray(u, dtype=float)

    if u.shape[0] != 41:
        return None

    if not np.isfinite(u).all():
        return None

    u1 = float(u[0])

    u2_41 = u[1:]
    ch1 = u2_41.reshape(5, 8)

    delta = np.empty(40, dtype=float)
    delta[0] = u[1] - u[0]
    delta[1:] = u[2:] - u[1:-1]
    ch2 = delta.reshape(5, 8)

    ch3 = np.full((5, 8), u1, dtype=float)

    x3 = np.stack([ch1, ch2, ch3], axis=0)

    if not np.isfinite(x3).all():
        return None

    return x3


# =============================================================================
# Data loading using proposed_framework
# =============================================================================
def load_all_data_direct() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("[DATA] Rebuilding raw U1-U41 using proposed_framework...")

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

    order_indices = []

    for soc in SOC_LIST:
        soc_mask = values_match(meta[SOC_COL].values, soc)

        for pulse in PULSE_LIST:
            pulse_mask = values_match(meta[pulse_col].values, pulse)
            group_idx = np.flatnonzero(soc_mask & pulse_mask)

            if len(group_idx) > 0:
                order_indices.append(group_idx)

    if len(order_indices) == 0:
        raise RuntimeError("No data found after SOC × pulse reordering.")

    order_indices = np.concatenate(order_indices)

    X = X[order_indices]
    y = y[order_indices]
    meta = meta.iloc[order_indices].reset_index(drop=True)

    soc = meta[SOC_COL].values
    soh = meta[SOH_COL].values

    print(f"[DATA] X shape: {X.shape}, y shape: {y.shape}")

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

    checkpoint = torch.load(str(checkpoint_path), map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model


# =============================================================================
# Feature extraction
# =============================================================================
def extract_triple_features(
    X: np.ndarray,
    y: np.ndarray,
    soc: np.ndarray,
    soh: np.ndarray,
    checkpoint_path: Path,
):
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(X), size=min(MAX_SAMPLES, len(X)), replace=False)

    X = X[idx]
    y = y[idx]
    soc = soc[idx]
    soh = soh[idx]

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(y)

    model = load_model(
        num_classes=len(label_encoder.classes_),
        checkpoint_path=checkpoint_path,
    )

    f_model = []
    f_raw = []
    f_manual = []
    valid_labels = []
    valid_soc = []
    valid_soh = []

    for i in range(len(X)):
        x3 = build_3ch_5x8_from_u41(X[i])

        if x3 is None:
            continue

        x3_tensor = torch.tensor(x3, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            model_feature = model.encoder(x3_tensor).cpu().numpy().squeeze()

        if not np.isfinite(model_feature).all():
            continue

        f_model.append(model_feature)
        f_raw.append(X[i])
        f_manual.append(x3.reshape(-1))

        valid_labels.append(labels[i])
        valid_soc.append(soc[i])
        valid_soh.append(soh[i])

    print(f"[TSNE] Valid samples: {len(f_model)}")

    if len(f_model) == 0:
        raise RuntimeError("No valid samples were extracted for t-SNE.")

    return (
        np.asarray(f_model),
        np.asarray(f_raw),
        np.asarray(f_manual),
        np.asarray(valid_labels),
        np.asarray(valid_soc),
        np.asarray(valid_soh),
        label_encoder.classes_,
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

    cache = np.load(str(cache_path), allow_pickle=True)

    return (
        cache["Z_model"],
        cache["Z_raw"],
        cache["Z_manual"],
        cache["labels"],
        cache["socs"],
        cache["sohs"],
        cache["class_names"],
    )


def save_cache(
    cache_path: Path,
    Z_model: np.ndarray,
    Z_raw: np.ndarray,
    Z_manual: np.ndarray,
    labels: np.ndarray,
    socs: np.ndarray,
    sohs: np.ndarray,
    class_names: np.ndarray,
):
    np.savez(
        str(cache_path),
        Z_model=Z_model,
        Z_raw=Z_raw,
        Z_manual=Z_manual,
        labels=labels,
        socs=socs,
        sohs=sohs,
        class_names=class_names,
    )

    print(f"[CACHE] Saved cache: {cache_path}")


# =============================================================================
# Plotting
# =============================================================================
def clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_material(
    Z: np.ndarray,
    labels: np.ndarray,
    class_names: np.ndarray,
    out_path: Path,
    title: str,
):
    fig, ax = plt.subplots(figsize=(6, 5))

    scatter = ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=labels,
        cmap="tab10",
        s=30,
        alpha=0.7,
        edgecolor="none",
    )

    ax.set_title(title, fontsize=11)
    clean_axis(ax)

    handles = []
    for idx, name in enumerate(class_names):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=5.5,
                markerfacecolor=scatter.cmap(scatter.norm(idx)),
                markeredgecolor="none",
                label=str(name),
            )
        )

    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=7.2,
        handletextpad=0.4,
    )

    fig.savefig(str(out_path), dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_continuous(
    Z: np.ndarray,
    values: np.ndarray,
    out_path: Path,
    title: str,
    colorbar_label: str,
    cmap: str,
):
    fig, ax = plt.subplots(figsize=(6, 5))

    scatter = ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=values,
        cmap=cmap,
        s=30,
        alpha=0.7,
        edgecolor="none",
    )

    ax.set_title(title, fontsize=11)
    clean_axis(ax)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(colorbar_label, fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5, length=2)
    cbar.outline.set_linewidth(0.6)

    fig.savefig(str(out_path), dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_all_plots(
    Z_model: np.ndarray,
    Z_raw: np.ndarray,
    Z_manual: np.ndarray,
    labels: np.ndarray,
    socs: np.ndarray,
    sohs: np.ndarray,
    class_names: np.ndarray,
):
    plot_items = [
        ("model", "Model features", Z_model),
        ("raw", "Raw U1-U41 features", Z_raw),
        ("manual", "Manual 3-channel features", Z_manual),
    ]

    for tag, title_name, Z in plot_items:
        plot_material(
            Z=Z,
            labels=labels,
            class_names=class_names,
            out_path=SAVE_DIR / f"suppfig10_{tag}_tsne_material.png",
            title=f"t-SNE {title_name} (Material)",
        )

        plot_continuous(
            Z=Z,
            values=socs,
            out_path=SAVE_DIR / f"suppfig10_{tag}_tsne_soc.png",
            title=f"t-SNE {title_name} (SOC)",
            colorbar_label="SOC (%)",
            cmap="viridis",
        )

        plot_continuous(
            Z=Z,
            values=sohs,
            out_path=SAVE_DIR / f"suppfig10_{tag}_tsne_soh.png",
            title=f"t-SNE {title_name} (SOH)",
            colorbar_label="SOH",
            cmap="plasma",
        )


# =============================================================================
# Main
# =============================================================================
def main():
    print("[INFO] Generating Supplementary Figure 10...")

    if USE_CACHE and CACHE_FILE.exists():
        Z_model, Z_raw, Z_manual, labels, socs, sohs, class_names = load_cache(CACHE_FILE)

    else:
        if USE_CACHE:
            print("[CACHE] No cache found. Rebuilding t-SNE...")

        checkpoint_path = find_existing_checkpoint()

        X, y, soc, soh = load_all_data_direct()

        (
            f_model,
            f_raw,
            f_manual,
            labels,
            socs,
            sohs,
            class_names,
        ) = extract_triple_features(
            X=X,
            y=y,
            soc=soc,
            soh=soh,
            checkpoint_path=checkpoint_path,
        )

        print(">> Running Model t-SNE...")
        Z_model = run_tsne_logic(
            f_model,
            use_pca=True,
            perp=30,
            lr=100,
            n_iter=2000,
        )

        print(">> Running Raw t-SNE...")
        Z_raw = run_tsne_logic(
            f_raw,
            use_pca=True,
            perp=30,
            lr=100,
            n_iter=2000,
        )

        print(">> Running Manual t-SNE...")
        Z_manual = run_tsne_logic(
            f_manual,
            use_pca=False,
            perp=30,
            lr=200,
            n_iter=1000,
        )

        save_cache(
            CACHE_FILE,
            Z_model=Z_model,
            Z_raw=Z_raw,
            Z_manual=Z_manual,
            labels=labels,
            socs=socs,
            sohs=sohs,
            class_names=class_names,
        )

    save_all_plots(
        Z_model=Z_model,
        Z_raw=Z_raw,
        Z_manual=Z_manual,
        labels=labels,
        socs=socs,
        sohs=sohs,
        class_names=class_names,
    )

    print("[DONE] Supplementary Figure 10 generated.")
    print(f"[SAVED] Output directory: {SAVE_DIR}")
    print(f"[SAVED] Cache: {CACHE_FILE}")


if __name__ == "__main__":
    main()