"""Convert the GSE194122 multiome BMMC cell-by-peak ATAC matrix into an EpiAgent cell-by-cCRE matrix.

Follows the EpiAgent peak -> cCRE tutorial: bedtools intersect peaks with cCRE.bed, build a
row-normalised peak x cCRE mapping matrix, multiply, then ceil.
"""
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from anndata.io import read_elem
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
H5AD_PATH = f"{ROOT}/data/GSE194122/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"
CCRE_FILE_PATH = f"{ROOT}/EpiAgent/data/cCRE.bed"
PEAK_FILE_PATH = f"{ROOT}/data/GSE194122/peak.bed"
OVERLAP_FILE_PATH = f"{ROOT}/data/GSE194122/peak_cCRE_overlap.bed"
OUT_PATH = f"{ROOT}/data/GSE194122/GSE194122_multiome_BMMC_cell_by_cCRE.h5ad"


def read_bed_file(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None)
    df['region'] = df[0].astype(str) + ':' + df[1].astype(str) + '-' + df[2].astype(str)
    return df['region']


def process_overlap(overlap_file, peak_categories, ccre_categories):
    overlap_df = pd.read_csv(overlap_file, sep='\t', header=None)
    overlap_df['peak_region'] = overlap_df.iloc[:, 0].astype(str) + ':' + overlap_df.iloc[:, 1].astype(str) + '-' + overlap_df.iloc[:, 2].astype(str)
    overlap_df['ccre_region'] = overlap_df.iloc[:, 3].astype(str) + ':' + overlap_df.iloc[:, 4].astype(str) + '-' + overlap_df.iloc[:, 5].astype(str)

    overlap_df['peak_code'] = pd.Categorical(overlap_df['peak_region'], categories=peak_categories).codes
    overlap_df['ccre_code'] = pd.Categorical(overlap_df['ccre_region'], categories=ccre_categories).codes
    assert (overlap_df['peak_code'] >= 0).all() and (overlap_df['ccre_code'] >= 0).all(), "unmatched overlap regions"

    return overlap_df[['peak_code', 'ccre_code']]


def create_mapping_matrix(peak_bed, ccre_bed, overlap_output):
    peak_regions = read_bed_file(peak_bed)
    ccre_regions = read_bed_file(ccre_bed)

    overlap_df = process_overlap(overlap_output, peak_regions, ccre_regions)

    data = [1] * len(overlap_df)
    row_indices = overlap_df['peak_code'].values
    col_indices = overlap_df['ccre_code'].values

    matrix = csr_matrix((data, (row_indices, col_indices)), shape=(len(peak_regions), len(ccre_regions)))

    row_sums = matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # avoid divide 0
    matrix = matrix.multiply(1 / row_sums)

    return csr_matrix(matrix)


# 1. Load the ATAC block of layers['counts'] plus obs, without touching X/obsm.
with h5py.File(H5AD_PATH, "r") as f:
    obs = read_elem(f["obs"])
    var = read_elem(f["var"])
    g = f["layers/counts"]
    counts = csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g.attrs["shape"]))

atac_mask = (var["feature_types"] == "ATAC").values
peak_X = counts[:, atac_mask].tocsr()
peak_names = var.index[atac_mask]
del counts
print(f"cell-by-peak: {peak_X.shape}, nnz={peak_X.nnz:,}")

# 2. Write peaks as BED (names are chr-start-end) and intersect with cCREs.
peak_df = peak_names.to_series().str.rsplit("-", n=2, expand=True)
peak_df.to_csv(PEAK_FILE_PATH, sep="\t", header=False, index=False)
command = f"bedtools intersect -a {PEAK_FILE_PATH} -b {CCRE_FILE_PATH} -wa -wb > {OVERLAP_FILE_PATH}"
subprocess.run(command, shell=True, check=True)

# 3. Mapping matrix and cell-by-cCRE matrix.
mapping_matrix = create_mapping_matrix(PEAK_FILE_PATH, CCRE_FILE_PATH, OVERLAP_FILE_PATH)
cCRE_X = peak_X.dot(mapping_matrix).tocsr()
cCRE_X.data = np.ceil(cCRE_X.data)
cCRE_X = cCRE_X.astype(np.float32)
cCRE_X.eliminate_zeros()

ccre_df = pd.read_csv(CCRE_FILE_PATH, sep='\t', header=None)
var_names = ccre_df[0].astype(str) + ':' + ccre_df[1].astype(str) + '-' + ccre_df[2].astype(str)
adata_cCRE = sc.AnnData(X=cCRE_X, obs=obs, var=pd.DataFrame(index=var_names.values))
print(adata_cCRE)

# 4. Coverage diagnostics.
peaks_hit = np.asarray(mapping_matrix.sum(axis=1)).ravel() > 0
peak_totals = np.asarray(peak_X.sum(axis=0)).ravel()
print(f"peaks overlapping >=1 cCRE: {peaks_hit.sum():,}/{len(peaks_hit):,} ({peaks_hit.mean():.1%})")
print(f"peak counts retained: {peak_totals[peaks_hit].sum() / peak_totals.sum():.1%}")
print(f"cCREs overlapped by >=1 peak: {(mapping_matrix.getnnz(axis=0) > 0).sum():,}/{mapping_matrix.shape[1]:,}")
print(f"cCREs per overlapping peak: {pd.Series(mapping_matrix.getnnz(axis=1)[peaks_hit]).describe().round(2).to_dict()}")
peak_nnz = pd.Series(peak_X.getnnz(axis=1)).describe().round(0)
ccre_nnz = pd.Series(cCRE_X.getnnz(axis=1)).describe().round(0)
print("accessible features per cell:\n", pd.DataFrame({"peaks": peak_nnz, "cCREs": ccre_nnz}))
print(f"cells with 0 accessible cCREs: {(cCRE_X.getnnz(axis=1) == 0).sum()}")

adata_cCRE.write(OUT_PATH)
print(f"saved {OUT_PATH}")
