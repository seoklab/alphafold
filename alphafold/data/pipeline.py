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

"""Functions for building the input features for the AlphaFold model."""

import os
from typing import Mapping, Optional, Sequence

import numpy as np
from absl import logging

# Internal import (7716).

from alphafold.common import residue_constants
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhsearch
from alphafold.data.tools import jackhmmer

FeatureDict = Mapping[str, np.ndarray]


def make_sequence_features(
    sequence: str, description: str, num_res: int) -> FeatureDict:
  """Constructs a feature dict of sequence features."""
  features = {}
  features['aatype'] = residue_constants.sequence_to_onehot(
      sequence=sequence,
      mapping=residue_constants.restype_order_with_x,
      map_unknown_to_x=True)
  features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
  features['domain_name'] = np.array([description.encode('utf-8')],
                                     dtype=np.object_)
  features['residue_index'] = np.array(range(num_res), dtype=np.int32)
  features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
  features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
  return features


def make_msa_features(
    msas: Sequence[Sequence[str]],
    deletion_matrices: Sequence[parsers.DeletionMatrix]) -> FeatureDict:
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')

  int_msa = []
  deletion_matrix = []
  seen_sequences = set()
  for msa_index, msa in enumerate(msas):
    if not msa:
      raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
    for sequence_index, sequence in enumerate(msa):
      if sequence in seen_sequences:
        continue
      seen_sequences.add(sequence)
      int_msa.append(
          [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
      deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

  num_res = len(msas[0][0])
  num_alignments = len(int_msa)
  features = {}
  features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
  features['msa'] = np.array(int_msa, dtype=np.int32)
  features['num_alignments'] = np.array(
      [num_alignments] * num_res, dtype=np.int32)
  return features


class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               hhsearch_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               bfd_database_path: Optional[str],
               uniclust30_database_path: Optional[str],
               small_bfd_database_path: Optional[str],
               pdb70_database_path: str,
               template_featurizer: templates.TemplateHitFeaturizer,
               use_small_bfd: bool,
               mgnify_max_hits: int = 501,
               uniref_max_hits: int = 10000,
               n_cpu: int = 1,
               overwrite: bool = False):
    """Constructs a feature dict for a given FASTA file."""
    self.n_cpu = n_cpu
    self.overwrite = overwrite
    self._use_small_bfd = use_small_bfd

    self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=uniref90_database_path,
        n_cpu=n_cpu)
    if use_small_bfd:
      self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
          binary_path=jackhmmer_binary_path,
          database_path=small_bfd_database_path,
          n_cpu=n_cpu)
    else:
      self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
          binary_path=hhblits_binary_path,
          databases=[bfd_database_path, uniclust30_database_path],
          n_cpu=n_cpu)
    self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
        binary_path=jackhmmer_binary_path,
        database_path=mgnify_database_path,
        n_cpu=n_cpu)
    self.hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path=hhsearch_binary_path,
        databases=[pdb70_database_path],
        n_cpu=n_cpu)
    self.template_featurizer = template_featurizer

    self.mgnify_max_hits = mgnify_max_hits
    self.uniref_max_hits = uniref_max_hits

  def _run(self, database: str, input_path: str, out_path: str) -> str:
    if not self.overwrite and os.path.isfile(out_path):
      logging.info(f"Skipping {database} search; pass --overwrite option to "
                   "force re-build the msa database")
      with open(out_path) as f:
        return f.read()

    if database == "uniref90":
      runner = self.jackhmmer_uniref90_runner
      fmt = "sto"
    elif database == "mgnify":
      runner = self.jackhmmer_mgnify_runner
      fmt = "sto"
    elif database == "bfd":
      runner = self.hhblits_bfd_uniclust_runner
      fmt = "a3m"
    elif database == "small_bfd":
      runner = self.jackhmmer_small_bfd_runner
      fmt = "sto"
    else:
      raise ValueError(f"unknown database {database}")

    result = runner.query(input_path)
    result_str = result[fmt]
    with open(out_path, 'w') as f:
      f.write(result_str)

    return result_str

  def process(self, input_fasta_path: str, msa_output_dir: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)

    uniref90_out_path = os.path.join(msa_output_dir, 'uniref90_hits.sto')
    jackhmmer_uniref90_sto = self._run(
        "uniref90", input_fasta_path, uniref90_out_path)
    uniref90_msa, uniref90_deletion_matrix, _ = parsers.parse_stockholm(
        jackhmmer_uniref90_sto)

    mgnify_out_path = os.path.join(msa_output_dir, 'mgnify_hits.sto')
    jackhmmer_mgnify_sto = self._run(
        "mgnify", input_fasta_path, mgnify_out_path)
    mgnify_msa, mgnify_deletion_matrix, _ = parsers.parse_stockholm(
        jackhmmer_mgnify_sto)
    mgnify_msa = mgnify_msa[:self.mgnify_max_hits]
    mgnify_deletion_matrix = mgnify_deletion_matrix[:self.mgnify_max_hits]

    hhsearch_out_path = os.path.join(msa_output_dir, 'uniref90_hits.a3m')
    if os.path.isfile(hhsearch_out_path):
      logging.info("Skipping hhsearch; pass --overwrite option to "
                   "force re-build the msa database")
      with open(hhsearch_out_path) as f:
        hhsearch_result = f.read()
    else:
      uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(
          jackhmmer_uniref90_sto, max_sequences=self.uniref_max_hits)
      hhsearch_result = self.hhsearch_pdb70_runner.query(uniref90_msa_as_a3m)
      with open(hhsearch_out_path, "w") as f:
        f.write(hhsearch_result)
    hhsearch_hits = parsers.parse_hhr(hhsearch_result)

    if self._use_small_bfd:
      bfd_out_path = os.path.join(msa_output_dir, 'small_bfd_hits.a3m')
      jackhmmer_small_bfd_sto = self._run(
          "small_bfd", input_fasta_path, bfd_out_path)
      bfd_msa, bfd_deletion_matrix, _ = parsers.parse_stockholm(
          jackhmmer_small_bfd_sto)
    else:
      bfd_out_path = os.path.join(msa_output_dir, 'bfd_uniclust_hits.a3m')
      hhblits_bfd_uniclust_a3m = self._run(
        "bfd", input_fasta_path, bfd_out_path)
      bfd_msa, bfd_deletion_matrix = parsers.parse_a3m(
        hhblits_bfd_uniclust_a3m)

    templates_result = self.template_featurizer.get_templates(
        query_sequence=input_sequence,
        query_pdb_code=None,
        query_release_date=None,
        hits=hhsearch_hits)

    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res)

    msa_features = make_msa_features(
        msas=(uniref90_msa, bfd_msa, mgnify_msa),
        deletion_matrices=(uniref90_deletion_matrix,
                           bfd_deletion_matrix,
                           mgnify_deletion_matrix))

    return {**sequence_features, **msa_features, **templates_result.features}
