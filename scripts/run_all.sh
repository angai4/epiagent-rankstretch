#!/bin/bash
# Submit steps 02-07 as one Slurm dependency chain (each step starts only if the one it needs succeeded).
# Run from the repo root after scripts/00_download.sh:  bash scripts/run_all.sh
# Step 01 only prints dataset metadata, so it is not part of the chain.

set -euo pipefail

submit() { sbatch --parsable "$@"; }

s02=$(submit scripts/02_peak_to_ccre.sbatch)
s03=$(submit --dependency=afterok:"$s02" scripts/03_sentence_length_stats.sbatch)
s04=$(submit --dependency=afterok:"$s02" scripts/04_zeroshot_embed.sbatch)
s05=$(submit --dependency=afterok:"$s04" scripts/05_zeroshot_knn.sbatch)
s06=$(submit --dependency=afterok:"$s04" scripts/06_rankstretch_embed.sbatch)
s07=$(submit --dependency=afterok:"$s06" scripts/07_rankstretch_umap.sbatch)

echo "submitted 02 $s02 · 03 $s03 · 04 $s04 · 05 $s05 · 06 $s06 · 07 $s07 (logs in logs/)"
