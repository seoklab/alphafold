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
# Downloads and unzips all required data for AlphaFold.
#
# Usage: bash download_all_data.sh /path/to/download/directory
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
ALPHAFOLD_HOME="$(realpath "${ALPHAFOLD_HOME-$(dirname "$SCRIPT_DIR")}")"

if [[ $# -eq 0 ]]; then
    echo "Setting download dir to $ALPHAFOLD_HOME/data"
    DOWNLOAD_DIR="$ALPHAFOLD_HOME/data"
else
    echo "Symlinking $ALPHAFOLD_HOME/data to download dir"
    DOWNLOAD_DIR="$1"
    ln -s "$(realpath "$DOWNLOAD_DIR")" "$ALPHAFOLD_HOME/data"
fi

if ! command -v aria2c &> /dev/null ; then
  echo "Error: aria2c could not be found. Please install aria2c (sudo apt install aria2)."
  exit 1
fi

echo "Downloading AlphaFold parameters..."
bash "${SCRIPT_DIR}/download_alphafold_params.sh" "${DOWNLOAD_DIR}"

echo "Downloading BFD..."
bash "${SCRIPT_DIR}/download_bfd.sh" "${DOWNLOAD_DIR}"

echo "Downloading Small BFD..."
bash "${SCRIPT_DIR}/download_small_bfd.sh" "${DOWNLOAD_DIR}"

echo "Downloading MGnify..."
bash "${SCRIPT_DIR}/download_mgnify.sh" "${DOWNLOAD_DIR}"

echo "Downloading PDB70..."
bash "${SCRIPT_DIR}/download_pdb70.sh" "${DOWNLOAD_DIR}"

echo "Downloading GPCR100..."
bash "${SCRIPT_DIR}/download_gpcr100.sh" "${DOWNLOAD_DIR}"

echo "Downloading PDB mmCIF files..."
bash "${SCRIPT_DIR}/download_pdb_mmcif.sh" "${DOWNLOAD_DIR}"

echo "Downloading Uniref30..."
bash "${SCRIPT_DIR}/download_uniref30.sh" "${DOWNLOAD_DIR}"

echo "Downloading Uniref90..."
bash "${SCRIPT_DIR}/download_uniref90.sh" "${DOWNLOAD_DIR}"

echo "Downloading UniProt..."
bash "${SCRIPT_DIR}/download_uniprot.sh" "${DOWNLOAD_DIR}"

echo "Downloading PDB SeqRes..."
bash "${SCRIPT_DIR}/download_pdb_seqres.sh" "${DOWNLOAD_DIR}"

echo "All data downloaded."
