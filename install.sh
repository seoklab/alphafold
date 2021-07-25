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

source "$__conda_prefix/bin/activate"
conda update -y -c defaults -n base conda
conda create -y -c defaults -n alphafold2 python=3.8

conda activate alphafold2
conda install -y -c defaults joblib
conda install -y -c conda-forge openmm==7.5.1 pdbfixer==1.7 cudnn==8.2.1.32 \
                                cudatoolkit==11.0.3 cudatoolkit-dev==11.0.3
conda install -y -c bioconda hmmer==3.3.2 hhsuite==3.3.0 kalign2==2.04
conda install -y -c nvidia libcusolver=11

pip install absl-py==0.13.0 biopython==1.79 chex==0.0.7 dm-haiku==0.0.4   \
    dm-tree==0.1.6 immutabledict==2.0.0 jax==0.2.14 ml-collections==0.1.0 \
    numpy==1.19.5 scipy==1.7.0 tensorflow==2.5.0
pip install --upgrade jax jaxlib==0.1.69+cuda110 \
            -f https://storage.googleapis.com/jax-releases/jax_releases.html

pushd "$__conda_prefix/envs/alphafold2/lib/python3.8/site-packages"
patch -p0 < "$__alphafold_home/docker/openmm.patch"
popd

python setup.py install

if [[ "$__sudo" == 'y' ]]; then
  sudo chown -R root:root "$__conda_prefix"
  sudo chmod -R go+rX "$__conda_prefix"

  sudo chown -R root:root "$__alphafold_home"
  sudo chmod -R go+rX "$__alphafold_home"
fi

set_envar_script="export ALPHAFOLD_CONDA_PREFIX=$(printf %q "$__conda_prefix")
export ALPHAFOLD_HOME=$(printf %q "$__alphafold_home")
export PATH=\"\$PATH:\$ALPHAFOLD_HOME/bin\""

if [[ "$__sudo" == 'y' ]]; then
  sudo echo "$set_envar_script" >>/etc/profile.d/alphafold.sh
else
  case "$SHELL" in
    *bash* ) __shell=bash;;
    *zsh*  ) __shell=zsh;;
    *      )
      echo "Unknown shell; you must manually set the environment variables!"
      echo "Please do the following command:
echo $(printf %q "$set_envar_script") >> <your rc filename>"
      ;;
  esac
  if [[ -n ${__shell-} ]]; then
    echo "$set_envar_script" >> "$HOME/.${__shell}rc"
  fi
fi

if [[ -f ~/condarc ]]; then
  mv ~/condarc ~/.condarc
fi
