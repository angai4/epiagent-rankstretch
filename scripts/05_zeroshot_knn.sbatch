#!/bin/bash
#SBATCH -J zeroshot_knn
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
#SBATCH -p compute
#SBATCH -n 8
#SBATCH --mem-per-cpu=6G
#SBATCH -t 2:0:0

# kNN label transfer for the zero-shot EpiAgent embeddings vs the dataset's ATAC LSI (after 04_zeroshot_embed).

# Submit from the repo root: sbatch scripts/05_zeroshot_knn.sbatch
set -euo pipefail

module load miniforge
mamba activate "${RANKSTRETCH_ENV:-epiagent-rankstretch}"  # env from environment.yml

python scripts/05_zeroshot_knn.py
