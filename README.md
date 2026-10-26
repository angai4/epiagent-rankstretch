# epiagent-rankstretch: EpiAgent zero-shot embeddings beyond 8,190 cCREs

Zero-shot [EpiAgent](https://github.com/xy-chen16/EpiAgent) [CLS] cell embeddings for the ATAC side of the
NeurIPS 2021 multiome bone marrow dataset ([GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122);
69,249 cells, 22 annotated cell types). The main result compares stock EpiAgent with a rank-stretch variant
that encodes every accessible cCRE instead of truncating each cell to its top 8,190.

![UMAPs of EpiAgent zero-shot embeddings, stock (truncated) vs rank-stretch, coloured by cell type](figures/umap_cell_type.svg)

## Why rank-stretch

EpiAgent's rank-embedding table has 8,192 rows, so stock inference keeps only each cell's top 8,190 cCREs by
TF-IDF (plus [CLS] and [SEP]). On this dataset 14,921 cells (22%) have more accessible cCREs than that.
Rank-stretch encodes the whole sentence. In a cell with L tokens, position i reads rank row
`floor(i * 8192 / L)`. Cells that already fit the table get the identity mapping, so their embeddings do not
change.

## Results

5-fold stratified kNN (k = 15) cell type label transfer on the raw 512-d embeddings
([results/rankstretch/knn_label_transfer.csv](results/rankstretch/knn_label_transfer.csv), with per-cell-type F1):

| Embedding | All cells: acc | All cells: macro F1 | Cells > 8,190 cCREs: acc | Cells > 8,190 cCREs: macro F1 |
|---|---|---|---|---|
| Stock EpiAgent (truncated) | 0.803 | 0.708 | 0.759 | 0.613 |
| Rank-stretch (all cCREs) | **0.809** | **0.722** | **0.787** | **0.653** |
| Dataset's own ATAC LSI (reference) | 0.852 | 0.800 | | |

## Reproducing

```bash
git clone --recurse-submodules https://github.com/angai4/epiagent-rankstretch.git && cd epiagent-rankstretch
mamba env create -f environment.yml      # env "epiagent-rankstretch"; Linux x86_64, CUDA 12 GPU for steps 04 and 06
bash scripts/00_download.sh              # EpiAgent cCRE files + GEO h5ad (md5-checked)
# download pretrained_EpiAgent.pth by hand into models/ (see below), then:
bash scripts/run_all.sh                  # submits steps 02-07 as one Slurm dependency chain
```

- **Software:** [environment.yml](environment.yml) pins the versions that produced the results, including
  `epiagent==0.0.2` from PyPI. That is the package the scripts import. To run in another env, set
  `RANKSTRETCH_ENV=<env name>` before submitting.
- **EpiAgent submodule:** `EpiAgent/` is pinned to upstream commit 9c4b790 and supplies the data files
  `data/cCRE.bed` (step 02) and `data/cCRE_document_frequency.npy` (steps 03 and 04). Its Python code is newer
  than 0.0.2 and is not imported. The functions this pipeline uses behave the same in both versions.
- **Pretrained weights:** download `pretrained_EpiAgent.pth` from the
  [EpiAgent Google Drive](https://drive.google.com/drive/folders/1WlNykSCNtZGsUp2oG0dw3cDdVKYDR-iX?usp=sharing)
  to `models/pretrained_EpiAgent.pth` (md5 `3cc1e76ba06543f706fa0fdf72fa97ea`; `00_download.sh` checks it).
- **Data:** `GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz` from
  [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122)
  (md5 `b78f960ae1ebd78ab86c607d287311fc`), gunzipped into `data/GSE194122/`.
- **Cluster:** the `.sbatch` wrappers are written for QMUL Apocrita (partitions, the H200 GPU request,
  `module load miniforge` / `bedtools2`). Adjust those lines for other Slurm clusters, and submit from the
  repo root, because log and script paths are relative to it.

Re-running steps 05 and 07 gives the same kNN metrics. UMAP layouts can shift slightly with other
numba / umap-learn versions.

## Pipeline

Each step is a Python script in `scripts/` with a matching Slurm `.sbatch` wrapper, which can also be
submitted on its own, e.g. `sbatch scripts/07_rankstretch_umap.sbatch`.

| Step | Script | Output |
|---|---|---|
| 0 | `00_download.sh` | fetches the inputs (see above) |
| 1 | `01_inspect_gse194122` | prints metadata, feature layout and labels |
| 2 | `02_peak_to_ccre` | maps the 116,490 peaks to EpiAgent cCREs (bedtools) → `data/GSE194122/*_cell_by_cCRE.h5ad` |
| 3 | `03_sentence_length_stats` | accessible cCREs per cell → `data/GSE194122/cell_sentence_lengths.csv` |
| 4 | `04_zeroshot_embed` (GPU) | TF-IDF, tokenisation and stock [CLS] embeddings → `results/zeroshot/` |
| 5 | `05_zeroshot_knn` | kNN label transfer, stock EpiAgent vs the dataset's ATAC LSI |
| 6 | `06_rankstretch_embed` (GPU) | rank-stretch [CLS] embeddings → `results/rankstretch/epiagent_embeddings.h5ad` |
| 7 | `07_rankstretch_umap` | kNN metrics and `figures/umap_cell_type.svg` |

## Licence

[MIT](LICENSE). EpiAgent and its pretrained weights are distributed by their authors under
[their own licence](https://github.com/xy-chen16/EpiAgent/blob/main/LICENSE).
