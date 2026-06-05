# -*- coding: utf-8 -*-
"""
plot_fig3e_tsne_material_raw_vs_embedded.py

Generate Figure 3e: Raw and Model Material t-SNE
Reproduces old zzz_tsne_triple_compare.py logic as closely as possible.
"""
import os
import sys
from pathlib import Path
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