#!/bin/bash
#SBATCH -J zeroshot_embed
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
#SBATCH -p apini
#SBATCH -A pilot_apini
#SBATCH -n 1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -t 4:0:0

# Zero-shot EpiAgent [CLS] embeddings for all GSE194122 multiome BMMC cells.
# Writes data/GSE194122/cell_sentences.npz and results/zeroshot/epiagent_zeroshot_embeddings.h5ad.

# Submit from the repo root: sbatch scripts/04_zeroshot_embed.sbatch
set -euo pipefail

module load miniforge
mamba activate "${RANKSTRETCH_ENV:-epiagent-rankstretch}"  # env from environment.yml

python scripts/04_zeroshot_embed.py
