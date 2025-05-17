#!/bin/bash
# Downloads preprocessed MeshGraphNets datasets from Hugging Face.
#
# Usage:
#     sh download_preprocessed_meshgraphnets_datasets.sh ${DATASET_NAME} ${OUTPUT_DIR}
# Example:
#     sh download_preprocessed_meshgraphnets_datasets.sh cylinder_flow .

set -e

DATASET_NAME="${1}"
OUTPUT_DIR="${2}/meshgraphnets_datasets/${DATASET_NAME}"
BASE_URL="https://huggingface.co/datasets/jhagnberger/meshgraphnets/resolve/main/${DATASET_NAME}"

# Get files
if [ "$DATASET_NAME" = "airfoil" ]; then
    files=(data node_type grid cells)
elif [ "$DATASET_NAME" = "cylinder_flow" ]; then
    files=(data node_type grid cells num_nodes num_cells)
else
   echo "Unknown dataset name: $DATASET_NAME"
   exit 1
fi


mkdir -p ${OUTPUT_DIR}
for split in train test
do
    for file in ${files[@]}
    do
    wget -O "${OUTPUT_DIR}/${split}_${file}.npy" "${BASE_URL}/${split}_${file}.npy?download=true"
    done
done