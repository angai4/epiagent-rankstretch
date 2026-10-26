#!/bin/bash
#SBATCH -J sentence_length_stats
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
#SBATCH -p computeshort
#SBATCH -n 2
#SBATCH --mem-per-cpu=8G
#SBATCH -t 0:30:0

# Cell sentence length statistics for the GSE194122 cell-by-cCRE matrix.

# Submit from the repo root: sbatch scripts/03_sentence_length_stats.sbatch
set -euo pipefail

module load miniforge
mamba activate "${RANKSTRETCH_ENV:-epiagent-rankstretch}"  # env from environment.yml

python scripts/03_sentence_length_stats.py
