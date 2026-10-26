"""Cell sentence length stats for the GSE194122 cell-by-cCRE matrix.

Sentence length = number of non-zero cCREs per cell (global_TFIDF keeps the sparsity pattern when
document frequencies are positive, and tokenization emits one token per non-zero entry).
CellDatasetForSCA caps sentences at max_length - 2 = 8190 tokens (+[CLS], [SEP]).
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem

pd.set_option("display.width", 200, "display.max_rows", 100)
ROOT = Path(__file__).resolve().parents[1]
H5AD_PATH = f"{ROOT}/data/GSE194122/GSE194122_multiome_BMMC_cell_by_cCRE.h5ad"
DF_PATH = f"{ROOT}/EpiAgent/data/cCRE_document_frequency.npy"
CAP = 8192 - 2

with h5py.File(H5AD_PATH, "r") as f:
    obs = read_elem(f["obs"])
    lengths = np.diff(f["X/indptr"][:])
    indices = f["X/indices"][:]

# TF-IDF sanity: every covered cCRE needs a finite, positive document frequency.
doc_freq = np.load(DF_PATH)
covered = np.unique(indices)
bad = ~(np.isfinite(doc_freq[covered]) & (doc_freq[covered] > 0))
print(f"covered cCREs: {len(covered):,}; with non-positive/non-finite doc freq: {bad.sum()}")
print(f"-> sentence length == non-zero cCREs per cell: {bad.sum() == 0}")

obs["length"] = lengths
obs["truncated"] = lengths > CAP
obs["dropped"] = np.clip(lengths - CAP, 0, None)

qs = [0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1]
print("\n=== Overall sentence length (tokens, excl. [CLS]/[SEP]) ===")
print(f"cells: {len(lengths):,}  mean: {lengths.mean():.0f}  std: {lengths.std():.0f}  total tokens: {lengths.sum():,}")
print(pd.Series(np.quantile(lengths, qs), index=[f"p{int(q * 100)}" for q in qs]).round(0).astype(int).to_string())

print("\n=== Histogram ===")
bins = [0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8190, 10000, 15000, 20000, 50000]
hist = pd.cut(pd.Series(lengths), bins=bins, right=True).value_counts().sort_index()
print(pd.DataFrame({"cells": hist, "pct": (100 * hist / len(lengths)).round(2),
                    "cum_pct": (100 * hist.cumsum() / len(lengths)).round(2)}).to_string())

print(f"\n=== Truncation at {CAP} tokens (max_length=8192) ===")
t = obs["truncated"]
print(f"cells truncated: {t.sum():,} ({t.mean():.2%})")
print(f"tokens dropped: {obs['dropped'].sum():,} ({obs['dropped'].sum() / lengths.sum():.2%} of all tokens)")
if t.any():
    frac = obs.loc[t, "dropped"] / obs.loc[t, "length"]
    print(f"among truncated cells, fraction of sentence dropped: median {frac.median():.1%}, max {frac.max():.1%}")
for cap in [2048, 4096]:
    print(f"(for reference) cells > {cap - 2}: {(lengths > cap - 2).mean():.2%}")

def summarise(key):
    g = obs.groupby(key, observed=True)
    out = pd.DataFrame({
        "cells": g.size(),
        "mean": g["length"].mean().round(0),
        "median": g["length"].median().round(0),
        "p5": g["length"].quantile(0.05).round(0),
        "p95": g["length"].quantile(0.95).round(0),
        "max": g["length"].max(),
        "pct_truncated": (100 * g["truncated"].mean()).round(1),
    })
    return out.sort_values("median", ascending=False).astype({c: int for c in ["mean", "median", "p5", "p95"]})

print("\n=== By cell type ===")
print(summarise("cell_type").to_string())
print("\n=== By batch ===")
print(summarise("batch").to_string())
print("\n=== By site ===")
print(summarise("Site").to_string())

obs[["cell_type", "batch", "length"]].to_csv(f"{ROOT}/data/GSE194122/cell_sentence_lengths.csv")
