#!/bin/bash

# Some of the commands are adopted from:
# https://github.com/kalininalab/alphafold_non_docker

set -euvo pipefail

type wget

__alphafold_home="$(dirname "$(realpath "$0")")"
__conda_prefix="$(realpath "${CONDA_PREFIX-/opt/conda}")"
__sudo="${SUDO-y}"

if [[ "$__sudo" == 'y' ]]; then
  function maybe_sudo() {
    sudo "$@"
  }
else
  function maybe_sudo() {
    "$@"
  }
fi

if [[ -f ~/.condarc ]]; then
  mv ~/.condarc ~/condarc
fi

if [[ -d "$__conda_prefix" ]]; then
  echo "Skipping anaconda installation existing at $__conda_prefix" >&2
else
  __tmpdir="$(mktemp -d)"
  wget -P "$__tmpdir" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  maybe_sudo bash "$__tmpdir/Miniconda3-latest-Linux-x86_64.sh" -b -p "$__conda_prefix"
  rm "$__tmpdir/Miniconda3-latest-Linux-x86_64.sh"
  rmdir "$__tmpdir"
fi

if [[ "$__sudo" == 'y' ]]; then
  sudo chmod go-rwx "$__conda_prefix"
  sudo chown -R "$USER" "$__conda_prefix"

  sudo chmod go-rwx "$__alphafold_home"
  sudo chown -R "$USER" "$__alphafold_home"
fi

cd "$__alphafold_home"

wget -P alphafold/common/ https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

set +eu
set --

source "$__conda_prefix/bin/activate" base
conda update -y -c conda-forge -n base conda
conda env create -v -f environment.yml || exit 1
conda activate alphafold2 || exit 1

set -eu

pip install --upgrade 'jax>=0.3.25,<0.4' 'jaxlib==0.3.25+cuda11.cudnn82' \
            -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pushd "$CONDA_PREFIX/lib/python3.9/site-packages"
git apply "$__alphafold_home/patch/pdbfixer.patch"
popd

pip install .

if [[ "$__sudo" == 'y' ]]; then
  sudo chown -R root:root "$__conda_prefix"
  sudo chmod -R go+rX "$__conda_prefix"

  sudo chown -R root:root "$__alphafold_home"
  sudo chmod -R go+rX "$__alphafold_home"
fi

set_envar_script="export ALPHAFOLD_CONDA_PREFIX=$(printf '%q' "$__conda_prefix")
export ALPHAFOLD_HOME=$(printf '%q' "$__alphafold_home")
export PATH=\"\$PATH:\$ALPHAFOLD_HOME/bin\""

if [[ "$__sudo" == 'y' ]]; then
  sudo bash -c "echo $(printf '%q' "$set_envar_script") >>/etc/profile.d/alphafold.sh"
else
  case "$SHELL" in
    *bash* ) __shell=bash;;
    *zsh*  ) __shell=zsh;;
    *      )
      echo "Unknown shell; you must manually set the environment variables!"
      echo "Please do the following command:
echo $(printf '%q' "$set_envar_script") >> <your rc filename>"
      ;;
  esac
  if [[ -n ${__shell-} ]]; then
    echo "$set_envar_script" >> "$HOME/.${__shell}rc"
  fi
fi

if [[ -f ~/condarc ]]; then
  mv ~/condarc ~/.condarc
fi
