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
# Downloads, unzips and flattens the PDB database for AlphaFold.
#
# Usage: bash download_pdb_mmcif.sh /path/to/download/directory
set -e

type parallel

if [[ $# -eq 0 ]]; then
    echo "Error: download directory must be provided as an input argument."
    exit 1
fi

if ! command -v aria2c &> /dev/null ; then
    echo "Error: aria2c could not be found. Please install aria2c (sudo apt install aria2)."
    exit 1
fi

if ! command -v rsync &> /dev/null ; then
    echo "Error: rsync could not be found. Please install rsync."
    exit 1
fi

DOWNLOAD_DIR="$1"
ROOT_DIR="${DOWNLOAD_DIR}/pdb_mmcif"
RAW_DIR="${ROOT_DIR}/raw"
MMCIF_DIR="${ROOT_DIR}/mmcif_files"

echo "Running rsync to fetch all mmCIF files (note that the rsync progress estimate might be inaccurate)..."
echo "If the download speed is too slow, try changing the mirror to:"
echo "  * rsync.ebi.ac.uk::pub/databases/pdb/data/structures/divided/mmCIF/ (Europe)"
echo "  * ftp.pdbj.org::ftp_data/structures/divided/mmCIF/ (Asia)"
echo "or see https://www.wwpdb.org/ftp/pdb-ftp-sites for more download options."
mkdir --parents "${RAW_DIR}"
# Fetch all directories from the PDB FTP site.
rsync -a --info=progress2 --delete --exclude='*.gz' \
  ftp.pdbj.org::ftp_data/structures/divided/mmCIF/ "${RAW_DIR}"

pushd "${RAW_DIR}"
# Parallel download using up to 16 connections
find -mindepth 1 -maxdepth 1 -type d -print0 \
  | parallel -0 -j16 --bar rsync -amq --delete \
      'ftp.pdbj.org::ftp_data/structures/divided/mmCIF/{/}/' '{}/'

find -type d -empty -delete  # Delete empty directories.

echo "Unzipping all mmCIF files..."
find -type f -iname "*.gz" -print0 \
  | parallel -0 -n1024 -j8 --bar '(gzip -dkq {} &>/dev/null || true)'
popd

echo "Flattening all mmCIF files..."
rm -rf "${MMCIF_DIR}"
mkdir --parents "${MMCIF_DIR}"
for dir in "${RAW_DIR}"/*; do
  mv -t "${MMCIF_DIR}/" "$dir/"*.cif
done

aria2c "ftp://ftp.wwpdb.org/pub/pdb/data/status/obsolete.dat" \
  --allow-overwrite --dir="${ROOT_DIR}"
