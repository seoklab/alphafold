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
from datetime import date
from itertools import cycle
from typing import List

from absl import app
from absl import flags
from absl import logging
from absl.flags import argparse_flags
import numpy as np
import jax
from joblib import Parallel, delayed

from alphafold.common import protein
from alphafold.common import profiler
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.model import features
# Internal import (7716).

#### USER CONFIGURATION ####

_conda_bin = os.path.join(os.environ['CONDA_PREFIX'], 'bin')

# Set to target of scripts/download_all_databases.sh
DOWNLOAD_DIR = os.path.join(os.environ['ALPHAFOLD_HOME'], 'data')

# Path to a directory that will store the results.
output_dir = os.getcwd()
nproc = int(os.environ.get("NSLOTS", multiprocessing.cpu_count()))

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

# Path to the Uniclust30 database for use by HHblits.
uniclust30_database_path = os.path.join(
    DOWNLOAD_DIR, 'uniclust30', 'uniclust30_2018_08', 'uniclust30_2018_08')

# Path to the PDB70 database for use by HHsearch.
pdb70_database_path = os.path.join(DOWNLOAD_DIR, 'pdb70', 'pdb70')

# Path to a directory with template mmCIF structures, each named <pdb_id>.cif')
template_mmcif_dir = os.path.join(DOWNLOAD_DIR, 'pdb_mmcif', 'mmcif_files')

# Path to a file mapping obsolete PDB IDs to their replacements.
obsolete_pdbs_path = os.path.join(DOWNLOAD_DIR, 'pdb_mmcif', 'obsolete.dat')

#### END OF USER CONFIGURATION ####

flags.DEFINE_string('output_dir', output_dir, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_integer('model_cnt', 5, 'Counts of models to use. Note that '
                     'AlphaFold provides 5 pretrained models, so setting '
                     'the count other than 5 is either redundant '
                     'or insufficient configuration.')
flags.DEFINE_integer('nproc', nproc, 'Maximum cpu count to use. Note that '
                     'the actual cpu load might be different than '
                     'the configured value.')
flags.DEFINE_string('max_template_date', date.today().isoformat(),
                    'Maximum template release date to consider'
                    '(ISO-8601 format - i.e. YYYY-MM-DD). '
                    'Important if folding historical test sets.')
flags.DEFINE_integer('ensemble', 1, 'Choose ensemble count: note that '
                     'AlphaFold recommends 1 ("full_dbs"), and the casp model '
                     'have used 8 model ensemblings ("casp14").')
flags.DEFINE_enum('model_type', 'normal', ['normal', 'ptm'],
                  'Choose model type to use - the casp14 equivalent '
                  'model (normal), or fined-tunded pTM models (ptm).')
flags.DEFINE_boolean(
    'benchmark', False, 'Run multiple JAX model evaluations '
    'to obtain a timing that excludes the compilation time, '
    'which should be more indicative of the time required '
    'for inferencing many proteins.')

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
flags.DEFINE_string('kalign_binary_path',
                    os.path.join(_conda_bin, 'kalign'),
                    'Path to the Kalign executable.')
flags.DEFINE_string('uniref90_database_path', uniref90_database_path,
                    'Path to the Uniref90 database for use by JackHMMER.')
flags.DEFINE_string('mgnify_database_path', mgnify_database_path,
                    'Path to the MGnify database for use by JackHMMER.')
flags.DEFINE_string('bfd_database_path', bfd_database_path,
                    'Path to the BFD database for use by HHblits.')
flags.DEFINE_string('uniclust30_database_path', uniclust30_database_path,
                    'Path to the Uniclust30 database for use by HHblits.')
flags.DEFINE_string('pdb70_database_path', pdb70_database_path,
                    'Path to the PDB70 database for use by HHsearch.')
flags.DEFINE_string('template_mmcif_dir', template_mmcif_dir,
                    'Path to a directory with template mmCIF structures, '
                    'each named <pdb_id>.cif')
flags.DEFINE_string('obsolete_pdbs_path', obsolete_pdbs_path,
                    'Path to file containing a mapping from obsolete PDB IDs '
                    'to the PDB IDs of their replacements.')
flags.DEFINE_integer('random_seed_seed', None, 'The random seed for the '
                     'random seed for the data pipeline. By default, this is '
                     'randomly generated. Note that even if this is set,'
                     'Alphafold may still not be deterministic, because '
                     'processes like GPU inference are nondeterministic.')
flags.DEFINE_boolean('overwrite', False, 'Whether to re-build the features, '
                     'even if the result exists in the target directories.')
FLAGS = flags.FLAGS


def fasta_parser(argv):
  parser = argparse_flags.ArgumentParser()
  parser.add_argument("fasta_paths", nargs="+",
                      help='Paths to FASTA files, each containing '
                      'one sequence. All FASTA paths must have '
                      'a unique basename as the basename is used to name the '
                      'output directories for each prediction.')
  return parser.parse_args(argv[1:]).fasta_paths


MAX_TEMPLATE_HITS = 20
# RELAX_MAX_ITERATIONS = 0
# RELAX_ENERGY_TOLERANCE = 2.39
# RELAX_STIFFNESS = 10.0
# RELAX_EXCLUDE_RESIDUES = []
# RELAX_MAX_OUTER_ITERATIONS = 20


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
  model_config.data.eval.num_ensemble = num_ensemble

  if not overwrite and os.path.isfile(result_output_path):
    print('Reusing existing model %s; pass --overwrite option to predict again'
          % model_name, file=sys.stderr, flush=True)

    with profiler(f'process_features_{model_name}'):
      processed_feature_dict = features.np_example_to_features(
          feature_dict, model_config, random_seed=random_seed)

    with open(result_output_path, 'rb') as f:
      prediction_result = pickle.load(f)
  else:
    model_params = data.get_model_haiku_params(model_id=model_id,
                                               model_type=model_type,
                                               data_dir=data_dir)
    model_runner = model.RunModel(model_config, model_params, device=device)

    with profiler(f'process_features_{model_name}'):
      processed_feature_dict = model_runner.process_features(
          feature_dict, random_seed=random_seed)

    with profiler(f'predict_and_compile_{model_name}') as p:
      prediction_result = model_runner.predict(processed_feature_dict)

    print('Total JAX model %s predict time '
          '(includes compilation time, see --benchmark): %s?' %
          (model_name, p.delta),
          file=sys.stderr)

    if benchmark:
      with profiler(f'predict_benchmark_{model_name}'):
        model_runner.predict(processed_feature_dict)

    # Save the model outputs.
    result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
    with open(result_output_path, 'wb') as f:
      pickle.dump(prediction_result, f, protocol=4)

  # Get mean pLDDT confidence metric.
  plddt = np.mean(prediction_result['plddt'])

  unrelaxed_protein = protein.from_prediction(processed_feature_dict,
                                              prediction_result)
  unrelaxed_pdb = protein.to_pdb(unrelaxed_protein)

  unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
  if overwrite or not os.path.isfile(unrelaxed_pdb_path):
    with open(unrelaxed_pdb_path, 'w') as f:
      f.write(unrelaxed_pdb)

  # Skip relaxation!
  # # Relax the prediction.
  # relaxed_output_path = os.path.join(output_dir, f'relaxed_{model_name}.pdb')
  # if not overwrite and os.path.isfile(relaxed_output_path):
  #   print('Reusing existing model %s; pass --overwrite option to relax again'
  #         % model_name, file=sys.stderr, flush=True)
  #   with open(relaxed_output_path) as f:
  #     relaxed_pdb_str = f.read()
  # else:
  #   amber_relaxer = relax.AmberRelaxation(
  #       max_iterations=RELAX_MAX_ITERATIONS,
  #       tolerance=RELAX_ENERGY_TOLERANCE,
  #       stiffness=RELAX_STIFFNESS,
  #       exclude_residues=RELAX_EXCLUDE_RESIDUES,
  #       max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)

  #   with profiler(f'relax_{model_name}'):
  #     relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)

  #   # Save the relaxed PDB.
  #   with open(relaxed_output_path, 'w') as f:
  #     f.write(relaxed_pdb_str)

  return model_name, profiler.timings, float(plddt), unrelaxed_pdb


def predict_structure_perdev(model_ids: List[int],
                             model_type: str,
                             random_seeds: List[int],
                             output_dir: str,
                             data_dir: str,
                             num_ensemble: int,
                             feature_dict: dict,
                             benchmark: bool,
                             device_id: int = None,
                             overwrite: bool = False):
  print(f"device:{device_id} is assigned {model_ids}",
        file=sys.stderr, flush=True)
  device = jax.devices()[device_id] if device_id is not None else None
  return [
      predict_structure_permodel(
          mid, model_type, output_dir, data_dir, num_ensemble, feature_dict,
          benchmark, random_seed, device=device, overwrite=overwrite)
      for mid, random_seed in zip(model_ids, random_seeds)
  ]


def _chunk_list(lst, n):
  return (lst[i::n] for i in range(n))


def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: pipeline.DataPipeline,
    model_ids: List[int],
    model_type: str,
    num_ensemble: int,
    benchmark: bool,
    random_seed_seed: int,
    nproc: int = 1,
    overwrite: bool = False):
  """Predicts structure using AlphaFold for the given sequence."""
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

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

  random.seed(random_seed_seed)
  random_seeds = [random.randrange(sys.maxsize)
                  for _ in range(len(model_ids))]

  # Run the models.
  backend = jax.default_backend()
  if backend != "cpu":
    dev_cnt = len(jax.devices())
    dev_pool = range(dev_cnt)
  else:
    dev_cnt = nproc
    dev_pool = [None]

  n_jobs = min(len(model_ids), dev_cnt)
  ids_chunked = _chunk_list(model_ids, n_jobs)
  random_seeds_chunked = _chunk_list(random_seeds, n_jobs)

  if backend == "cpu":
    per_pool_nproc = str(max(nproc // n_jobs, 1))
    os.environ["MKL_NUM_THREADS"] = per_pool_nproc
    os.environ["OPENBLAS_NUM_THREADS"] = per_pool_nproc

    old_xla_flags = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = old_xla_flags + (
        f"--xla_cpu_multi_thread_eigen=true "
        f"intra_op_parallelism_threads={nproc} "
        f"inter_op_parallelism_threads={nproc}")

  multiprocessing.set_start_method("spawn")
  dev_results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
      delayed(predict_structure_perdev)(
          ids, model_type, seeds, output_dir, FLAGS.data_dir, num_ensemble,
          feature_dict, benchmark, device_id=device_id, overwrite=overwrite)
      for ids, seeds, device_id
      in zip(ids_chunked, random_seeds_chunked, cycle(dev_pool)))
  results = [result for dev_result in dev_results for result in dev_result]

  plddts = {}
  unrelaxed_pdbs = {}
  for model_name, model_timings, plddt, pdb in results:
    profiler.timings.update(model_timings)
    plddts[model_name] = plddt
    unrelaxed_pdbs[model_name] = pdb

  # Rank by pLDDT and write out unrelaxed PDBs in rank order.
  ranked_order = []
  for idx, (model_name, _) in enumerate(
      sorted(plddts.items(), key=lambda x: x[1], reverse=True)):
    ranked_order.append(model_name)
    ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
    with open(ranked_output_path, 'w') as f:
      f.write(unrelaxed_pdbs[model_name])

  ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
  with open(ranking_output_path, 'w') as f:
    f.write(json.dumps({'plddts': plddts, 'order': ranked_order}, indent=4))

  timings_output_path = os.path.join(output_dir, 'timings.json')
  profiler.dump(timings_output_path)


def main(fasta_paths):
  # Check for duplicate FASTA file names and existence of them
  fasta_names = []
  for fasta_path in fasta_paths:
    p = pathlib.Path(fasta_path)
    fasta_names.append(p.stem)
    if not p.is_file():
      raise FileNotFoundError(repr(fasta_path))
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  template_featurizer = templates.TemplateHitFeaturizer(
      mmcif_dir=FLAGS.template_mmcif_dir,
      max_template_date=FLAGS.max_template_date,
      max_hits=MAX_TEMPLATE_HITS,
      kalign_binary_path=FLAGS.kalign_binary_path,
      release_dates_path=None,
      obsolete_pdbs_path=FLAGS.obsolete_pdbs_path)

  data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      hhsearch_binary_path=FLAGS.hhsearch_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniclust30_database_path=FLAGS.uniclust30_database_path,
      pdb70_database_path=FLAGS.pdb70_database_path,
      template_featurizer=template_featurizer,
      n_cpu=FLAGS.nproc,
      overwrite=FLAGS.overwrite)

  if FLAGS.model_cnt != 5:
    logging.warning("5 models is recommended, as AlphaFold provides 5 "
                    "pretrained model configurations")

  model_ids = list(range(FLAGS.model_cnt))
  logging.info('Have %d models: %s', FLAGS.model_cnt, model_ids)

  random_seed_seed = FLAGS.random_seed_seed
  if random_seed_seed is None:
    random_seed_seed = random.randrange(sys.maxsize)
  logging.info('Using random seed-seed %d for the data pipeline',
               random_seed_seed)

  # Predict structure for each of the sequences.
  for fasta_path, fasta_name in zip(fasta_paths, fasta_names):
    with profiler(
        f"fasta name {fasta_name}", printer=logging.info, store=False):
      predict_structure(fasta_path=fasta_path,
                        fasta_name=fasta_name,
                        output_dir_base=FLAGS.output_dir,
                        data_pipeline=data_pipeline,
                        model_ids=model_ids,
                        model_type=FLAGS.model_type,
                        num_ensemble=FLAGS.ensemble,
                        benchmark=FLAGS.benchmark,
                        random_seed_seed=random_seed_seed,
                        nproc=FLAGS.nproc,
                        overwrite=FLAGS.overwrite)


if __name__ == '__main__':
  app.run(main, flags_parser=fasta_parser)
