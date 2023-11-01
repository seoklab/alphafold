#!/bin/bash
#
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Downloads and unzips the MGnify database for AlphaFold.
#
# Usage: bash download_mgnify.sh /path/to/download/directory
set -e

if [[ $# -eq 0 ]]; then
    echo "Error: download directory must be provided as an input argument."
    exit 1
fi

if ! command -v aria2c &> /dev/null ; then
    echo "Error: aria2c could not be found. Please install aria2c (sudo apt install aria2)."
    exit 1
fi

DOWNLOAD_DIR="$1"
ROOT_DIR="${DOWNLOAD_DIR}/mgnify"
SOURCE_FILE="$(basename "$0")/mgnify.txt"
BASENAME=mgy_clusters_2018_12.fa.gz

mkdir --parents "${ROOT_DIR}"
aria2c -x16 -j32 -i "${SOURCE_FILE}" --dir="${ROOT_DIR}"
pushd "${ROOT_DIR}"
gzip -df "${BASENAME}"
ln -fs "${BASENAME%.gz}" "mgy_clusters.fa"
popd
