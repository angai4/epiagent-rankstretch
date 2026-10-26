#!/bin/bash
#SBATCH -J rankstretch_embed
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
#SBATCH -p apini
#SBATCH -A pilot_apini
#SBATCH -n 1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH -t 3:0:0

# Zero-shot EpiAgent [CLS] embeddings with rank-stretch (full sentences) for all GSE194122 BMMC cells.
# Needs 04_zeroshot_embed outputs. Writes results/rankstretch/epiagent_embeddings.h5ad.

# Submit from the repo root: sbatch scripts/06_rankstretch_embed.sbatch
set -euo pipefail

module load miniforge
mamba activate "${RANKSTRETCH_ENV:-epiagent-rankstretch}"  # env from environment.yml

python scripts/06_rankstretch_embed.py
