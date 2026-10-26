"""UMAPs of zero-shot EpiAgent [CLS] embeddings with and without rank-stretch, coloured by annotated cell type.

Figures:
  figures/umap_cell_type.svg         side by side: truncated (stock, top 8,190 cCREs) | rank-stretch (all cCREs).
                                     Cell types are coloured by lineage family (one hue per lineage, shades within)
                                     and named at each type's median position.
Each arm has its own UMAP fit (15-NN graph on the raw 512-d embedding).
Also reports 5-fold kNN label-transfer accuracy / macro F1 per arm, on all cells and on cells over 8,190 cCREs.
"""
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import to_rgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = f"{ROOT}/results/rankstretch"
FIG_DIR = f"{ROOT}/figures"
EMB_PATH = f"{OUT_DIR}/epiagent_embeddings.h5ad"
MAX_CCRES = 8190
ARMS = {"truncate": f"stock EpiAgent · truncated to top {MAX_CCRES:,} cCREs",
        "stretch": "extended (stretched) rank embeddings · all cCREs"}

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
# Lineage families in rough developmental order; one hue per family, shades within it.
LINEAGES = {
    "Stem": ("#52514e", ["HSC"]),
    "Erythroid / MK": ("#e34948", ["MK/E prog", "Proerythroblast", "Erythroblast", "Normoblast"]),
    "Myeloid": ("#eb6834", ["G/M prog", "ID2-hi myeloid prog", "CD14+ Mono", "CD16+ Mono"]),
    "Dendritic": ("#6250d6", ["cDC2", "pDC"]),
    "B": ("#2a78d6", ["Lymph prog", "Transitional B", "Naive CD20+ B", "B1 B", "Plasma cell"]),
    "T": ("#008300", ["CD4+ T naive", "CD4+ T activated", "CD8+ T naive", "CD8+ T"]),
    "NK / ILC": ("#d55181", ["NK", "ILC"]),
}
CELL_TYPE_ORDER = [ct for _, cts in LINEAGES.values() for ct in cts]


def shades(hex_color, n):
    """n shades of one hue, dark to light (mixed towards black, then white)."""
    rgb = np.array(to_rgb(hex_color))
    if n == 1:
        return [rgb]
    return [rgb * (1 - t) if t > 0 else rgb + (1 - rgb) * -t for t in np.linspace(0.35, -0.45, n)]


COLORS = {ct: c for base, cts in LINEAGES.values() for ct, c in zip(cts, shades(base, len(cts)))}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK})

adata = sc.read_h5ad(EMB_PATH)
assert set(CELL_TYPE_ORDER) == set(adata.obs["cell_type"].unique())
cell_type = adata.obs["cell_type"].astype(str).values
over = adata.obs["sentence_length"].values > MAX_CCRES
print(f"{len(adata):,} cells, {over.sum():,} ({over.mean():.1%}) over {MAX_CCRES:,} cCREs")

# kNN label transfer (5-fold, stratified by cell type); same folds for both arms.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print("5-fold kNN (k=15) cell type label transfer:")
rows = []
for arm in ARMS:
    pred = cross_val_predict(KNeighborsClassifier(n_neighbors=15), adata.obsm[f"X_epiagent_{arm}"], cell_type,
                             cv=cv, n_jobs=5)
    for subset, m in {"all": np.ones_like(over), f">{MAX_CCRES}": over, f"<={MAX_CCRES}": ~over}.items():
        rows.append({"arm": arm, "cells": subset, "n": int(m.sum()),
                     "accuracy": accuracy_score(cell_type[m], pred[m]),
                     "macro_F1": f1_score(cell_type[m], pred[m], average="macro")})
    rows += [{"arm": arm, "cells": ct, "n": int((cell_type == ct).sum()), "F1": f}
             for ct, f in zip(CELL_TYPE_ORDER, f1_score(cell_type, pred, average=None, labels=CELL_TYPE_ORDER))]
metrics = pd.DataFrame(rows)
metrics.to_csv(f"{OUT_DIR}/knn_label_transfer.csv", index=False, float_format="%.4f")
print(metrics[metrics["F1"].isna()].drop(columns="F1").round(4).to_string(index=False))
print("per-cell-type F1:\n" + metrics.dropna(subset="F1").pivot(index="cells", columns="arm", values="F1")
      .loc[CELL_TYPE_ORDER].round(3).to_string())

coords = {}
for arm in ARMS:
    sc.pp.neighbors(adata, use_rep=f"X_epiagent_{arm}", n_neighbors=15, random_state=0, key_added=arm)
    sc.tl.umap(adata, random_state=0, neighbors_key=arm)
    coords[arm] = adata.obsm[f"X_umap_{arm}"] = adata.obsm.pop("X_umap")
adata.write(EMB_PATH)

order = np.random.default_rng(0).permutation(len(adata))  # no cell type drawn wholly on top
colors = np.array([COLORS[ct] for ct in cell_type])


def style(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# Side by side, coloured by cell type, names at each type's median position.
fig, axes = plt.subplots(1, 2, figsize=(17, 9.2))
for ax, (arm, title) in zip(axes, ARMS.items()):
    xy = coords[arm]
    ax.scatter(xy[order, 0], xy[order, 1], s=0.8, c=colors[order], linewidths=0, rasterized=True)
    for ct in CELL_TYPE_ORDER:
        x, y = np.median(xy[cell_type == ct], axis=0)
        ax.text(x, y, ct, ha="center", va="center", fontsize=8, color=INK,
                path_effects=[pe.withStroke(linewidth=2.5, foreground=SURFACE)])
    acc, acc_over = (metrics[(metrics["arm"] == arm) & (metrics["cells"] == c)].iloc[0]
                     for c in ("all", f">{MAX_CCRES}"))
    ax.set_title(f"{title}\nkNN acc {acc.accuracy:.3f} · macro F1 {acc.macro_F1:.3f}   "
                 f"(cells > {MAX_CCRES:,} cCREs: acc {acc_over.accuracy:.3f} · F1 {acc_over.macro_F1:.3f})",
                 loc="left", fontsize=10.5, color=INK)
    style(ax)
handles, labels = [], []
for lineage, (_, cts) in LINEAGES.items():
    handles.append(plt.Line2D([], [], linestyle="none"))
    labels.append(f"$\\bf{{{lineage.replace(' ', '~')}}}$")
    for ct in cts:
        handles.append(plt.Line2D([], [], linestyle="none", marker="o", markersize=7, color=COLORS[ct]))
        labels.append(f"{ct}  (n = {(cell_type == ct).sum():,})")
fig.legend(handles, labels, loc="center right", frameon=False, fontsize=8.5, handletextpad=0.3,
           labelcolor=INK2, borderaxespad=0.5)
fig.suptitle(f"EpiAgent zero-shot [CLS] embeddings · GSE194122 multiome BMMC · {len(adata):,} cells, "
             f"{over.sum():,} ({over.mean():.0%}) with > {MAX_CCRES:,} accessible cCREs\n"
             "coloured by annotated cell type; each embedding has its own UMAP fit",
             x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0, 0.83, 1))
os.makedirs(FIG_DIR, exist_ok=True)
fig.savefig(f"{FIG_DIR}/umap_cell_type.svg", dpi=200)  # scatter is rasterized, text and legend stay vector
plt.close(fig)
print(f"saved {FIG_DIR}/umap_cell_type.svg")
