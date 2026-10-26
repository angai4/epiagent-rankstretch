"""5-fold kNN label transfer of zero-shot EpiAgent [CLS] embeddings vs the dataset's own ATAC LSI embedding.

Reports accuracy / macro F1 per embedding and per-cell-type F1 for EpiAgent.
"""
from pathlib import Path

import h5py
import pandas as pd
import scanpy as sc
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
EMB_PATH = f"{ROOT}/results/zeroshot/epiagent_zeroshot_embeddings.h5ad"
ORIG_PATH = f"{ROOT}/data/GSE194122/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"

# Rough developmental order so related populations sit next to each other.
CELL_TYPE_ORDER = [
    "HSC", "MK/E prog", "Proerythroblast", "Erythroblast", "Normoblast",
    "G/M prog", "ID2-hi myeloid prog", "CD14+ Mono", "CD16+ Mono", "cDC2", "pDC",
    "Lymph prog", "Transitional B", "Naive CD20+ B", "B1 B", "Plasma cell",
    "CD4+ T naive", "CD4+ T activated", "CD8+ T naive", "CD8+ T", "NK", "ILC",
]

adata = sc.read_h5ad(EMB_PATH)
with h5py.File(ORIG_PATH, "r") as f:
    lsi = f["obsm/ATAC_lsi_full"][:]
    orig_names = f["obs/_index"][:].astype(str)
assert (orig_names == adata.obs_names.values).all()
adata.obsm["X_lsi"] = lsi
assert set(CELL_TYPE_ORDER) == set(adata.obs["cell_type"].unique())

# kNN label transfer (5-fold, stratified by cell type).
y = adata.obs["cell_type"].astype(str).values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print("5-fold kNN (k=15) cell type label transfer:")
for rep in ["X_epiagent", "X_lsi"]:
    pred = cross_val_predict(KNeighborsClassifier(n_neighbors=15), adata.obsm[rep], y, cv=cv, n_jobs=5)
    print(f"  {rep:11s} accuracy {accuracy_score(y, pred):.4f}  macro F1 {f1_score(y, pred, average='macro'):.4f}")
    if rep == "X_epiagent":
        per_type = pd.Series(f1_score(y, pred, average=None, labels=CELL_TYPE_ORDER), index=CELL_TYPE_ORDER)
print("per-cell-type F1 (EpiAgent zero-shot):\n" + per_type.round(3).to_string())
