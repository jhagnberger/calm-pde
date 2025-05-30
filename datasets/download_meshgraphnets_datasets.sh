#!/bin/bash
# Download script from https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/download_dataset.sh
# with adapted output dir to match the remaning structure.
#
# Copyright 2020 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Usage:
#     sh download_meshgraphnet_datasets.sh ${DATASET_NAME} ${OUTPUT_DIR}
# Example:
#     sh download_meshgraphnet_datasets.sh flag_simple /tmp/

set -e

DATASET_NAME="${1}"
OUTPUT_DIR="${2}/meshgraphnets_datasets/${DATASET_NAME}"

BASE_URL="https://storage.googleapis.com/dm-meshgraphnets/${DATASET_NAME}/"

mkdir -p ${OUTPUT_DIR}
for file in meta.json train.tfrecord valid.tfrecord test.tfrecord
do
wget -O "${OUTPUT_DIR}/${file}" "${BASE_URL}${file}"
done