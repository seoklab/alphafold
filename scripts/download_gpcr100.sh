#!/bin/bash
#
# Copyright (C) 2022 Lab of Computational Biology and Biomolecular Engineering
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
# Downloads and unzips the GPCR100 database for AlphaFold.
#
# Usage: bash download_gpcr100.sh /path/to/download/directory
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
ROOT_DIR="${DOWNLOAD_DIR}/gpcr100"
SOURCE_URL_PREFIX="https://zenodo.org/record/5745217/files/"

mkdir -p "${ROOT_DIR}"
for state in Active Inactive Intermediate; do
	aria2c "${SOURCE_URL_PREFIX}/GPCR100.${state}.tgz" --dir="${ROOT_DIR}"

	archive="${ROOT_DIR}/GPCR100.${state}.tgz"
	tar -xzf "${archive}" -C "${ROOT_DIR}"
	rm -f "${archive}"
done
