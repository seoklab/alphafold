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

"""Full AlphaFold protein structure prediction script."""
import json
import os
import pathlib
import pickle
import random
import sys
import multiprocessing
import shutil
from datetime import date
from itertools import cycle
from typing import List, Dict, Union, Optional

from absl import app
from absl import flags
from absl import logging
from absl.flags import argparse_flags
import numpy as np
import jax
import psutil
from joblib import Parallel, delayed

from alphafold.common import protein, residue_constants, devices, profiler
from alphafold.model import config, model, data
from alphafold.data import pipeline, pipeline_multimer, templates
from alphafold.data.tools import hhsearch, hmmsearch
from alphafold.relax import relax
# Internal import (7716).

#### USER CONFIGURATION ####

_conda_bin = os.path.join(os.getenv('CONDA_PREFIX'), 'bin')

# Set to target of scripts/download_all_databases.sh
DOWNLOAD_DIR = os.path.join(os.getenv('ALPHAFOLD_HOME'), 'data')

# Path to a directory that will store the results.
output_dir = os.getcwd()
nproc = int(os.getenv("NSLOTS", multiprocessing.cpu_count()))

# You can individually override the following paths if you have placed the
# data in locations other than the DOWNLOAD_DIR.

# Path to directory of supporting data, contains 'params' dir.
data_dir = DOWNLOAD_DIR

# Path to the Uniref90 database for use by JackHMMER.
uniref90_database_path = os.path.join(
    DOWNLOAD_DIR, 'uniref90', 'uniref90.fasta')

# Path to the MGnify database for use by JackHMMER.
mgnify_database_path = os.path.join(
    DOWNLOAD_DIR, 'mgnify', 'mgy_clusters.fa')

# Path to the BFD database for use by HHblits.
bfd_database_path = os.path.join(
    DOWNLOAD_DIR, 'bfd',
    'bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt')

small_bfd_database_path = os.path.join(
    DOWNLOAD_DIR, 'small_bfd',
    'bfd-first_non_consensus_sequences.fasta')

# Path to the Uniclust30 database for use by HHblits.
uniclust30_database_path = os.path.join(
    DOWNLOAD_DIR, 'uniclust30', 'uniclust30_2018_08', 'uniclust30_2018_08')

# Path to the Uniprot database for use by JackHMMER.
uniprot_database_path = os.path.join(DOWNLOAD_DIR, 'uniprot', 'uniprot.fasta')

# Path to the PDB70 database for use by HHsearch.
pdb70_database_paths = {
  "normal": os.path.join(DOWNLOAD_DIR, 'pdb70', 'pdb70'),
  "active": os.path.join(DOWNLOAD_DIR, 'gpcr100', 'GPCR100.Active'),
  "intermediate": os.path.join(DOWNLOAD_DIR, 'gpcr100', 'GPCR100.Intermediate'),
  "inactive": os.path.join(DOWNLOAD_DIR, 'gpcr100', 'GPCR100.Inactive'),
}

# Path to the PDB seqres database for use by hmmsearch.
_pdb_seqres_dir = os.path.join(DOWNLOAD_DIR, 'pdb_seqres')
pdb_seqres_database_paths = {
  "normal": os.path.join(_pdb_seqres_dir, 'pdb_seqres.txt'),
  "active": os.path.join(_pdb_seqres_dir, 'pdb_seqres_active.txt'),
  "intermediate": os.path.join(_pdb_seqres_dir, 'pdb_seqres_intermediate.txt'),
  "inactive": os.path.join(_pdb_seqres_dir, 'pdb_seqres_inactive.txt'),
}

# Path to a directory with template mmCIF structures, each named <pdb_id>.cif')
template_mmcif_dir = os.path.join(DOWNLOAD_DIR, 'pdb_mmcif', 'mmcif_files')

# Path to a file mapping obsolete PDB IDs to their replacements.
obsolete_pdbs_path = os.path.join(DOWNLOAD_DIR, 'pdb_mmcif', 'obsolete.dat')

#### END OF USER CONFIGURATION ####

# yapf: disable
flags.DEFINE_list('is_prokaryote_list', None, 'Optional for multimer system, '
                  'not used by the single chain system. This list should '
                  'contain a boolean for each fasta specifying true where the '
                  'target complex is from a prokaryote, and false where it is '
                  'not, or where the origin is unknown. These values determine '
                  'the pairing method for the MSA.')

flags.DEFINE_string('output_dir', output_dir, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_boolean('overwrite', False, 'Whether to re-build the features, '
                     'even if the result exists in the target directories.')
flags.DEFINE_integer('model_cnt', 5, 'Counts of models to use. Note that '
                     'AlphaFold provides 5 pretrained models, so setting '
                     'the count other than 5 is either redundant '
                     'or insufficient configuration.')
flags.DEFINE_integer('nproc', nproc, 'Maximum cpu count to use. Note that '
                     'the actual cpu load might be different than '
                     'the configured value.')
flags.DEFINE_boolean('jit', False,
                     'Whether to jit compile the alphafold model.')
flags.DEFINE_string('max_template_date', date.today().isoformat(),
                    'Maximum template release date to consider'
                    '(ISO-8601 format - i.e. YYYY-MM-DD). '
                    'Important if folding historical test sets.')
flags.DEFINE_integer('ensemble', 1, 'Choose ensemble count: note that '
                     'AlphaFold recommends 1 ("full_dbs"), and the casp model '
                     'have used 8 model ensemblings ("casp14").')
flags.DEFINE_boolean('small_bfd', False,
                     'Whether to use smaller genetic database config.')
flags.DEFINE_enum('model_type', 'normal',
                  tuple(config.MODEL_PRESETS) + ("casp14", ),
                  'Choose model type to use - the casp14 equivalent '
                  'model (normal), fined-tunded pTM models (ptm), '
                  'the alphafold-multimer (multimer), and normal model with '
                  '8 model ensemblings (casp14).\n'
                  'Note that the casp14 preset is just an alias of '
                  '--model_type=normal --ensemble=8 option.')
flags.DEFINE_enum('state', 'normal',
                  tuple(pdb70_database_paths),
                  'Choose state for GPCRs. Will be ignored if '
                  '--pdb70_database_path (monomer) or '
                  '--pdb_seqres_database_path (multimer) argument is set.')
flags.DEFINE_integer('num_multimer_predictions_per_model', 5, 'How many '
                     'predictions (each with a different random seed) will be '
                     'generated per model. E.g. if this is 2 and there are 5 '
                     'models then there will be 10 predictions per input. '
                     'Note: this FLAG only applies if model_preset=multimer')
flags.DEFINE_boolean('run_relax', True, 'Whether to run the final relaxation '
                     'step on the predicted models. Turning relax off might '
                     'result in predictions with distracting stereochemical '
                     'violations but might help in case you are having issues '
                     'with the relaxation stage.')
flags.DEFINE_boolean('benchmark', False, 'Run multiple JAX model evaluations '
                     'to obtain a timing that excludes the compilation time, '
                     'which should be more indicative of the time required for '
                     'inferencing many proteins.')
flags.DEFINE_boolean('debug', False, 'Whether to print debug output.')
flags.DEFINE_boolean('quiet', False, 'Whether to silence non-warning messages. '
                     'Takes precedence over --debug option.')
flags.DEFINE_string('data_dir', data_dir,
                    'Path to directory of supporting data.')
flags.DEFINE_string('jackhmmer_binary_path',
                    os.path.join(_conda_bin, 'jackhmmer'),
                    'Path to the JackHMMER executable.')
flags.DEFINE_string('hhblits_binary_path',
                    os.path.join(_conda_bin, 'hhblits'),
                    'Path to the HHblits executable.')
flags.DEFINE_string('hhsearch_binary_path',
                    os.path.join(_conda_bin, 'hhsearch'),
                    'Path to the HHsearch executable.')
flags.DEFINE_string('hmmsearch_binary_path',
                    os.path.join(_conda_bin, 'hmmsearch'),
                    'Path to the hmmsearch executable.')
flags.DEFINE_string('hmmbuild_binary_path',
                    os.path.join(_conda_bin, 'hmmbuild'),
                    'Path to the hmmbuild executable.')
flags.DEFINE_string('kalign_binary_path',
                    os.path.join(_conda_bin, 'kalign'),
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', uniref90_database_path,
                    'Path to the Uniref90 database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', mgnify_database_path,
                    'Path to the MGnify database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', bfd_database_path,
                    'Path to the BFD database for use by HHblits.')
flags.DEFINE_string('small_bfd_database_path', small_bfd_database_path,
                    'Path to the BFD database for use by HHblits.')
flags.DEFINE_string('uniclust30_database_path', uniclust30_database_path,
                    'Path to the Uniclust30 database for use by HHblits.')
flags.DEFINE_string('uniprot_database_path', uniprot_database_path,
                    'Path to the Uniprot database for use by JackHMMer.')
flags.DEFINE_string('pdb70_database_path', None,
                    'Path to the PDB70 database for use by HHsearch.')
flags.DEFINE_string('pdb_seqres_database_path', None,
                    'Path to the PDB seqres database for use by hmmsearch.')
flags.DEFINE_string('template_mmcif_dir', template_mmcif_dir,
                    'Path to a directory with template mmCIF structures, '
                    'each named <pdb_id>.cif')
flags.DEFINE_string('obsolete_pdbs_path', obsolete_pdbs_path,
                    'Path to file containing a mapping from obsolete PDB IDs '
                    'to the PDB IDs of their replacements.')
flags.DEFINE_integer('random_seed', None, 'The random seed for the '
                     'data pipeline. By default, this is randomly generated. '
                     'Note that even if this is set, Alphafold may still '
                     'not be deterministic, because processes like '
                     'GPU inference are nondeterministic.')
FLAGS = flags.FLAGS
# yapf: enable


if devices.BACKEND == "cpu":
  def check_nvidia_cache():
    pass
else:
  _NFS_CACHE = frozenset(
      os.stat(pi.mountpoint).st_dev
      for pi in psutil.disk_partitions(all=True)
      if pi.fstype == 'nfs')

  if _NFS_CACHE:
    def check_nvidia_cache():
      nvidia_cachedir = os.path.expanduser('~/.nv')
      nvidia_cachedir_non_nfs = os.path.expandvars("/tmp/$USER/nv")

      try:
        devid = os.stat(nvidia_cachedir, follow_symlinks=True).st_dev
      except FileNotFoundError:
        # Maybe broken symlink, test it
        if os.path.islink(nvidia_cachedir):
          os.makedirs(nvidia_cachedir_non_nfs, exist_ok=True)
          assert os.path.isdir(nvidia_cachedir)
          return
      else:
        if devid in _NFS_CACHE:
          shutil.rmtree(nvidia_cachedir, ignore_errors=True)
        elif os.path.exists(nvidia_cachedir):
          return

      os.makedirs(nvidia_cachedir_non_nfs, exist_ok=True)
      os.symlink(nvidia_cachedir_non_nfs, nvidia_cachedir)


def fasta_parser(args):
  parser = argparse_flags.ArgumentParser(prog="alphafold")
  parser.add_argument(
      "fasta_paths",
      nargs="+",
      help='Paths to FASTA files, each containing a prediction '
      'target that will be folded one after another. If a FASTA file contains '
      'multiple sequences, then it will be folded as a multimer. Paths should '
      'be separated by commas. All FASTA paths must have a unique basename as '
      'the basename is used to name the output directories for each prediction.'
  )
  return parser.parse_args(args[1:]).fasta_paths


MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def predict_structure_permodel(
    model_id: int,
    model_type: str,
    output_dir: str,
    data_dir: str,
    num_ensemble: int,
    feature_dict: dict,
    benchmark: bool,
    random_seed: int,
    device=None,
    overwrite: bool = False):
  model_name = f"model_{model_id + 1}"
  if model_type != "normal":
    model_name += f"_{model_type}"

  result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
  model_config = config.model_config(model_id, model_type)
  if model_type == "multimer":
    model_config.model.num_ensemble_eval = num_ensemble
  else:
    model_config.data.eval.num_ensemble = num_ensemble

  model_params = data.get_model_haiku_params(model_id=model_id,
                                             model_type=model_type,
                                             data_dir=data_dir)
  model_runner = model.RunModel(
      model_config, model_params, device=device, jit_compile=FLAGS.jit)
  with profiler(f'process_features_{model_name}'):
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed)

  if not overwrite and os.path.isfile(result_output_path):
    logging.info(f'Reusing existing model {model_name}; '
                 'pass --overwrite option to predict again')

    with open(result_output_path, 'rb') as f:
      prediction_result = pickle.load(f)
  else:
    with profiler(f'predict_and_compile_{model_name}') as p:
      prediction_result = model_runner.predict(processed_feature_dict,
                                               random_seed)

    fasta_name = os.path.basename(output_dir)
    logging.info(f'Total JAX model {model_name} on {fasta_name} predict time '
                 f'(includes compilation time, see --benchmark): {p.delta}')

    if benchmark:
      with profiler(f'predict_benchmark_{model_name}'):
        model_runner.predict(processed_feature_dict, random_seed)

    # Save the model outputs.
    result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
    with open(result_output_path, 'wb') as f:
      pickle.dump(prediction_result, f, protocol=4)

  # Get mean pLDDT confidence metric.
  plddt = prediction_result['plddt']
  plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1)

  unrelaxed_protein = protein.from_prediction(
    features=processed_feature_dict,
    result=prediction_result,
    b_factors=plddt_b_factors,
    remove_leading_feature_dimension=not model_runner.multimer_mode)
  unrelaxed_pdb = protein.to_pdb(unrelaxed_protein)

  unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
  if overwrite or not os.path.isfile(unrelaxed_pdb_path):
    with open(unrelaxed_pdb_path, 'w') as f:
      f.write(unrelaxed_pdb)

  ranking_confidence = prediction_result['ranking_confidence']
  label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
  return (label, model_name, profiler.timings, ranking_confidence,
          unrelaxed_protein)


def predict_structure_perdev(model_ids: List[int],
                             model_type: str,
                             random_seeds: List[int],
                             output_dir: str,
                             data_dir: str,
                             num_ensemble: int,
                             feature_dict: dict,
                             benchmark: bool,
                             device_id: int = None,
                             overwrite: bool = False,
                             _loglvl: int = logging.INFO):
  # Required for multiprocessing
  logging.set_verbosity(_loglvl)

  logging.info(f"device:{device_id} is assigned {model_ids}")
  device = jax.devices()[device_id] if device_id is not None else None
  return [
      predict_structure_permodel(
          mid, model_type, output_dir, data_dir, num_ensemble, feature_dict,
          benchmark, random_seed, device=device, overwrite=overwrite)
      for mid, random_seed in zip(model_ids, random_seeds)
  ]


def _chunk_list(lst, n):
  return (lst[i::n] for i in range(n))


def _preprocess(
    fasta_path: str,
    output_dir: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
    overwrite: Optional[bool]):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  # Write out features as a pickled dictionary.
  features_output_path = os.path.join(output_dir, 'features.pkl')
  if not overwrite and os.path.isfile(features_output_path):
    logging.info("Skipping feature generation; pass --overwrite option to "
                 "force re-build the feature dictionary")
    with open(features_output_path, 'rb') as f:
      feature_dict = pickle.load(f)
  else:
    with profiler("features"):
      # Get features.
      feature_dict = data_pipeline.process(
          input_fasta_path=fasta_path,
          msa_output_dir=msa_output_dir)
    # Write out features as a pickled dictionary.
    with open(features_output_path, 'wb') as f:
      pickle.dump(feature_dict, f, protocol=4)
  return feature_dict


def _predict(
    feature_dict: pipeline.FeatureDict,
    output_dir: str,
    model_ids_chunked: List[List[int]],
    model_type: str,
    num_ensemble: int,
    benchmark: bool,
    random_seeds_chunked: List[List[int]],
    n_jobs: int,
    overwrite: Optional[bool]):
  multiprocessing.set_start_method("spawn")
  dev_results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
      delayed(predict_structure_perdev)(
          ids, model_type, seeds, output_dir, FLAGS.data_dir, num_ensemble,
          feature_dict, benchmark, device_id=device_id, overwrite=overwrite,
          _loglvl=logging.get_verbosity())
      for ids, seeds, device_id
      in zip(model_ids_chunked, random_seeds_chunked, cycle(devices.DEV_POOL)))
  results = [result for dev_result in dev_results for result in dev_result]

  ranking_confidences = {}
  unrelaxed_prots = {}
  for _, model_name, model_timings, ranking_confidence, prot in results:
    profiler.timings.update(model_timings)
    ranking_confidences[model_name] = ranking_confidence
    unrelaxed_prots[model_name] = prot
  return results, ranking_confidences, unrelaxed_prots


def _relax(
    unrelaxed_prots: Dict[str, protein.Protein],
    fasta_path: str,
    output_dir: str,
    relaxer: Optional[relax.AmberRelaxation],
    overwrite: Optional[bool]):
  if relaxer:
    result_pdbs = {}
    for model_name, unrelaxed_prot in unrelaxed_prots.items():
      relaxed_output_path = os.path.join(output_dir,
                                         f'relaxed_{model_name}.pdb')

      if not overwrite and os.path.isfile(relaxed_output_path):
        logging.info(f'Reusing existing relaxed model {model_name}; '
                     'pass --overwrite option to relax again')
        with open(relaxed_output_path, 'r') as f:
          relaxed_pdb = f.read()
      else:
        # Relax the prediction.
        with profiler(f'relax_{model_name}'):
          relaxed_pdb, *_ = relaxer.process(prot=unrelaxed_prot)

        # Save the relaxed PDB.
        with open(relaxed_output_path, 'w') as f:
          f.write(relaxed_pdb)

      result_pdbs[model_name] = relaxed_pdb
  else:
    logging.info(f'Skipping relaxation for {fasta_path}')
    result_pdbs = {
        model_name: protein.to_pdb(prot)
        for model_name, prot in unrelaxed_prots.items()
    }

  return result_pdbs


def _report(
    results: List[tuple],
    ranking_confidences: dict,
    result_pdbs: Dict[str, str],
    output_dir: str,):
  """Rank by model confidence and write out relaxed PDBs in rank order."""
  ranked_order = []
  for idx, (model_name, _) in enumerate(
      sorted(ranking_confidences.items(), key=lambda x: x[1], reverse=True)):
    ranked_order.append(model_name)
    ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
    with open(ranked_output_path, 'w') as f:
      f.write(result_pdbs[model_name])

  ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
  with open(ranking_output_path, 'w') as f:
    f.write(json.dumps(
      {results[0][0]: ranking_confidences, 'order': ranked_order}, indent=4))

  timings_output_path = os.path.join(output_dir, 'timings.json')
  profiler.dump(timings_output_path)


def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
    model_ids_chunked: List[List[int]],
    model_type: str,
    num_ensemble: int,
    relaxer: Optional[relax.AmberRelaxation],
    benchmark: bool,
    random_seeds_chunked: List[List[int]],
    n_jobs: int,
    overwrite: Optional[bool] = False):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  output_dir = os.path.join(output_dir_base, fasta_name)

  # Run msa
  feature_dict = _preprocess(fasta_path, output_dir, data_pipeline, overwrite)

  # Run the models
  results, ranking_confidences, unrelaxed_prots = _predict(
    feature_dict, output_dir, model_ids_chunked, model_type, num_ensemble,
    benchmark, random_seeds_chunked, n_jobs, overwrite)

  # Run relaxation
  result_pdbs = _relax(
    unrelaxed_prots, fasta_path, output_dir, relaxer, overwrite)

  # Report results
  _report(results, ranking_confidences, result_pdbs, output_dir)


def _check_multimer(fasta_path: str):
  with open(fasta_path) as f:
    for line in f:
      if line.startswith('>'):
        break
    else:
      raise ValueError('Fasta file contains no sequences.')

    for line in f:
      if line.startswith('>'):
        return True
  return False


def main(fasta_paths: List[str]):
  # Check for duplicate FASTA file names and existence of them
  fasta_names = []
  for fasta_path in fasta_paths:
    p = pathlib.Path(fasta_path)
    fasta_names.append(p.stem)
    if not p.is_file():
      raise FileNotFoundError(repr(fasta_path))
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  if FLAGS.state != "normal":
    fasta_names = [f"{name}_{FLAGS.state}" for name in fasta_names]

  logging.set_verbosity(logging.INFO)
  if FLAGS.debug:
    logging.set_verbosity(logging.DEBUG)
  if FLAGS.quiet:
    logging.set_verbosity(logging.WARNING)

  check_nvidia_cache()

  run_multimer_system = 'multimer' in FLAGS.model_type
  if not run_multimer_system:
    if any(_check_multimer(fasta_path) for fasta_path in fasta_paths):
      logging.warning('FASTA file contains multiple sequences. '
                      'Running multimer model instead.')
      FLAGS.model_type = 'multimer'
      run_multimer_system = True
    elif FLAGS.model_type == 'casp14':
      FLAGS.model_type = 'normal'
      FLAGS.ensemble = 8

  if run_multimer_system:
    if (pdb_seqres_database_path := FLAGS.pdb_seqres_database_path) is None:
      pdb_seqres_database_path = pdb_seqres_database_paths[FLAGS.state]

    template_searcher = hmmsearch.Hmmsearch(
        binary_path=FLAGS.hmmsearch_binary_path,
        hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
        database_path=pdb_seqres_database_path,
        n_cpu=FLAGS.nproc)
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)
  else:
    if (pdb70_database_path := FLAGS.pdb70_database_path) is None:
      pdb70_database_path = pdb70_database_paths[FLAGS.state]

    template_searcher = hhsearch.HHSearch(
        binary_path=FLAGS.hhsearch_binary_path,
        databases=[pdb70_database_path],
        n_cpu=FLAGS.nproc)
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)

  monomer_data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniclust30_database_path=FLAGS.uniclust30_database_path,
      small_bfd_database_path=FLAGS.small_bfd_database_path,
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      use_small_bfd=FLAGS.small_bfd,
      n_cpu=FLAGS.nproc,
      overwrite=FLAGS.overwrite)

  if run_multimer_system:
    data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        uniprot_database_path=FLAGS.uniprot_database_path)
  else:
    data_pipeline = monomer_data_pipeline

  if FLAGS.run_relax:
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)
  else:
    amber_relaxer = None

  if FLAGS.model_cnt != 5:
    logging.warning("5 models is recommended, as AlphaFold provides 5 "
                    "pretrained model configurations")

  if run_multimer_system:
    num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
    pred_cnt = FLAGS.model_cnt * num_predictions_per_model
  else:
    num_predictions_per_model = 1
    pred_cnt = FLAGS.model_cnt

  model_ids = list(range(pred_cnt))
  logging.info('Have %d models: %s', FLAGS.model_cnt, model_ids)

  random_seed = FLAGS.random_seed
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize // pred_cnt)
  logging.info('Using random seed %d for the data pipeline', random_seed)
  random_seeds: List[int] = [
      i + random_seed * pred_cnt for i in range(pred_cnt)
  ]

  n_jobs = min(pred_cnt, devices.DEV_CNT)
  model_ids_chunked = list(_chunk_list(model_ids, n_jobs))
  random_seeds_chunked = list(_chunk_list(random_seeds, n_jobs))

  # Predict structure for each of the sequences.
  for fasta_path, fasta_name in zip(fasta_paths, fasta_names):
    with profiler(
        f"fasta name {fasta_name}", printer=logging.info, store=False):
      predict_structure(fasta_path=fasta_path,
                        fasta_name=fasta_name,
                        output_dir_base=FLAGS.output_dir,
                        data_pipeline=data_pipeline,
                        model_ids_chunked=model_ids_chunked,
                        model_type=FLAGS.model_type,
                        num_ensemble=FLAGS.ensemble,
                        relaxer=amber_relaxer,
                        benchmark=FLAGS.benchmark,
                        random_seeds_chunked=random_seeds_chunked,
                        n_jobs=n_jobs,
                        overwrite=FLAGS.overwrite)


if __name__ == '__main__':
  app.run(main, flags_parser=fasta_parser)
