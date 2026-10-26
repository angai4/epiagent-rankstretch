#!/bin/bash
#SBATCH -J inspect_gse194122
#SBATCH -o logs/%x.o%j
#SBATCH -e logs/%x.e%j
#SBATCH -p computeshort
#SBATCH -n 2
#SBATCH --mem-per-cpu=8G
#SBATCH -t 0:30:0

# Inspect GSE194122 multiome BMMC h5ad: metadata, feature layout, cell type labels.
# Reads obs/var only (no X/layers/obsm) to keep memory small.

# Submit from the repo root: sbatch scripts/01_inspect_gse194122.sbatch
set -euo pipefail

module load miniforge
mamba activate "${RANKSTRETCH_ENV:-epiagent-rankstretch}"  # env from environment.yml

python scripts/01_inspect_gse194122.py
