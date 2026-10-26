#!/bin/bash
#SBATCH -J rankstretch_umap
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
#SBATCH -p compute
#SBATCH -n 8
#SBATCH --mem-per-cpu=6G
#SBATCH -t 2:0:0

# UMAPs (with / without rank-stretch) + kNN label transfer + figures (after 06_rankstretch_embed).

# Submit from the repo root: sbatch scripts/07_rankstretch_umap.sbatch
set -euo pipefail

module load miniforge
mamba activate "${RANKSTRETCH_ENV:-epiagent-rankstretch}"  # env from environment.yml

python scripts/07_rankstretch_umap.py
