#!/bin/bash
# Fetch the inputs: EpiAgent's cCRE files (git submodule) and the GSE194122 processed multiome h5ad from GEO,
# checked against the md5 of the copy that produced the results. The pretrained weights are on Google Drive
# and must be downloaded by hand (see README); this script checks them if they are already in place.
# Run from the repo root:  bash scripts/00_download.sh

set -euo pipefail

git submodule update --init EpiAgent

H5AD=data/GSE194122/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad
if [ ! -f "$H5AD" ]; then
    mkdir -p data/GSE194122
    wget -c -O "$H5AD.gz" \
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/$(basename "$H5AD").gz"
    echo "b78f960ae1ebd78ab86c607d287311fc  $H5AD.gz" | md5sum -c -
    gunzip "$H5AD.gz"
fi

WEIGHTS=models/pretrained_EpiAgent.pth
if [ -f "$WEIGHTS" ]; then
    echo "3cc1e76ba06543f706fa0fdf72fa97ea  $WEIGHTS" | md5sum -c -
else
    echo "Missing $WEIGHTS: download pretrained_EpiAgent.pth from" \
         "https://drive.google.com/drive/folders/1WlNykSCNtZGsUp2oG0dw3cDdVKYDR-iX into models/"
fi
