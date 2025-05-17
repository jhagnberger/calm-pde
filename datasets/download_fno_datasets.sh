#!/bin/bash
# Downloads FNO Navier-Stokes datasets from Hugging Face.
#
# Usage:
#     sh download_fno_datasets.sh ${DATASET_NAME} ${OUTPUT_DIR}
# Example:
#     sh download_fno_datasets.sh 1e-4 .

set -e

DATASET_NAME="${1}"
OUTPUT_DIR="${2}/fno_vorticity_datasets"
DATASET_URL=""

# Get the dataset ID based on the dataset name
if [ "$DATASET_NAME" = "1e-4" ]; then
   DATASET_URL="https://huggingface.co/datasets/jhagnberger/fno-vorticity/resolve/main/1e-4/navier_stokes_v1e-4_N10000_T30_u.npy?download=true"
elif [ "$DATASET_NAME" = "1e-5" ]; then
   DATASET_URL="https://huggingface.co/datasets/jhagnberger/fno-vorticity/resolve/main/1e-5/navier_stokes_v1e-5_N1200_T20_u.npy?download=true"
else
   echo "Unknown dataset name: $DATASET_NAME"
   exit 1
fi

mkdir -p ${OUTPUT_DIR}
wget -P "${OUTPUT_DIR}" --content-disposition "${DATASET_URL}"