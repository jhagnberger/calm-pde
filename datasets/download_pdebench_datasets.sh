#!/bin/bash
# Downloads PDEBench Burgers and Navier-Stokes datasets.
#
# Usage:
#     sh download_pdebench_datasets.sh ${DATASET_NAME} ${OUTPUT_DIR}
# Example:
#     sh download_pdebench_datasets.sh 1d_burgers .

set -e

DATASET_NAME="${1}"
OUTPUT_DIR="${2}/pdebench_datasets"
BASE_URL="https://darus.uni-stuttgart.de/api/access/datafile/"
DATASET_ID=""

# Get the dataset ID based on the dataset name
if [ "$DATASET_NAME" = "1d_burgers" ]; then
   DATASET_ID="268190"
elif [ "$DATASET_NAME" = "3d_navier_stokes" ]; then
   DATASET_ID="173286"
else
   echo "Unknown dataset name: $DATASET_NAME"
   exit 1
fi

mkdir -p ${OUTPUT_DIR}
wget -P "${OUTPUT_DIR}" --content-disposition "${BASE_URL}${DATASET_ID}"