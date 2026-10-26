#!/bin/bash
#SBATCH -J peak_to_ccre
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
#SBATCH -p compute
#SBATCH -n 4
#SBATCH --mem-per-cpu=16G
#SBATCH -t 2:0:0

# Build the EpiAgent cell-by-cCRE matrix from the GSE194122 cell-by-peak (ATAC counts) matrix.

# Submit from the repo root: sbatch scripts/02_peak_to_ccre.sbatch
set -euo pipefail

module load miniforge
module load bedtools2
mamba activate "${RANKSTRETCH_ENV:-epiagent-rankstretch}"  # env from environment.yml

python scripts/02_peak_to_ccre.py
