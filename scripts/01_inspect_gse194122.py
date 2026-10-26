from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

pd.set_option("display.width", 200, "display.max_rows", 100, "display.max_columns", 50)
p = f"{Path(__file__).resolve().parents[1]}/data/GSE194122/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"
with h5py.File(p, "r") as f:
    obs = read_elem(f["obs"])
    var = read_elem(f["var"])
    uns = {k: read_elem(f["uns"][k]) for k in ["dataset_id", "genome", "organism"]}
    # Check X vs counts on the first cells of the ATAC block
    atac_cols = np.where(var["feature_types"].values == "ATAC")[0]
    ip = f["layers/counts/indptr"][:3]
    cnt_idx = f["layers/counts/indices"][ip[0]:ip[2]]
    cnt_dat = f["layers/counts/data"][ip[0]:ip[2]]

print("uns:", uns)
print("\nvar feature_types:\n", var["feature_types"].value_counts())
print("ATAC col range:", atac_cols.min(), atac_cols.max())
print("\nATAC var head:\n", var.iloc[atac_cols[:5]])
m = np.isin(cnt_idx, atac_cols)
print("\nATAC counts values (first 2 cells):", np.unique(cnt_dat[m], return_counts=True))
print("\nobs head:\n", obs.head(3).T)
for c in ["cell_type", "batch", "Site", "DonorNumber", "Samplename"]:
    print(f"\n--- {c}: {obs[c].nunique()} unique\n{obs[c].value_counts().to_string()}")
print("\ncell_type x batch:\n", pd.crosstab(obs["cell_type"], obs["batch"]).to_string())
