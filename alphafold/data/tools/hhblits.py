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

"""Library to run HHblits from Python."""

import glob
import os
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, List, Mapping, Optional, Sequence

from absl import logging
from alphafold.data.tools import utils
# Internal import (7716).


_HHBLITS_DEFAULT_P = 20
_HHBLITS_DEFAULT_Z = 500


class HHBlits:
  """Python wrapper of the HHblits binary."""

  def __init__(self,
               *,
               binary_path: str,
               databases: Sequence[str],
               split_bfd_uniclust: bool,
               n_cpu: int = 4,
               n_iter: int = 3,
               e_value: float = 0.001,
               maxseq: int = 1_000_000,
               realign_max: int = 100_000,
               maxfilt: int = 100_000,
               min_prefilter_hits: int = 1000,
               all_seqs: bool = False,
               alt: Optional[int] = None,
               p: int = _HHBLITS_DEFAULT_P,
               z: int = _HHBLITS_DEFAULT_Z):
    """Initializes the Python HHblits wrapper.

    Args:
      binary_path: The path to the HHblits executable.
      databases: A sequence of HHblits database paths. This should be the
        common prefix for the database files (i.e. up to but not including
        _hhm.ffindex etc.)
      split_bfd_uniclust: Whether to run on the databases separately.
      n_cpu: The number of CPUs to give HHblits.
      n_iter: The number of HHblits iterations.
      e_value: The E-value, see HHblits docs for more details.
      maxseq: The maximum number of rows in an input alignment. Note that this
        parameter is only supported in HHBlits version 3.1 and higher.
      realign_max: Max number of HMM-HMM hits to realign. HHblits default: 500.
      maxfilt: Max number of hits allowed to pass the 2nd prefilter.
        HHblits default: 20000.
      min_prefilter_hits: Min number of hits to pass prefilter.
        HHblits default: 100.
      all_seqs: Return all sequences in the MSA / Do not filter the result MSA.
        HHblits default: False.
      alt: Show up to this many alternative alignments.
      p: Minimum Prob for a hit to be included in the output hhr file.
        HHblits default: 20.
      z: Hard cap on number of hits reported in the hhr file.
        HHblits default: 500. NB: The relevant HHblits flag is -Z not -z.

    Raises:
      RuntimeError: If HHblits binary not found within the path.
    """
    self.binary_path = binary_path
    self.databases = databases
    self.split_bfd_uniclust = split_bfd_uniclust

    for database_path in self.databases:
      if not glob.glob(database_path + '_*'):
        logging.error('Could not find HHBlits database %s', database_path)
        raise ValueError(f'Could not find HHBlits database {database_path}')

    self.n_cpu = n_cpu
    self.n_iter = n_iter
    self.e_value = e_value
    self.maxseq = maxseq
    self.realign_max = realign_max
    self.maxfilt = maxfilt
    self.min_prefilter_hits = min_prefilter_hits
    self.all_seqs = all_seqs
    self.alt = alt
    self.p = p
    self.z = z

  def query(self, input_fasta_path: str) -> List[Mapping[str, Any]]:
    """Queries the database using HHblits."""
    common_cmd = [
        self.binary_path,
        '-i', input_fasta_path,
        '-cpu', str(self.n_cpu),
        '-o', '/dev/null',
        '-n', str(self.n_iter),
        '-e', str(self.e_value),
        '-maxseq', str(self.maxseq),
        '-realign_max', str(self.realign_max),
        '-maxfilt', str(self.maxfilt),
        '-min_prefilter_hits', str(self.min_prefilter_hits)]
    if self.all_seqs:
      common_cmd += ['-all']
    if self.alt:
      common_cmd += ['-alt', str(self.alt)]
    if self.p != _HHBLITS_DEFAULT_P:
      common_cmd += ['-p', str(self.p)]
    if self.z != _HHBLITS_DEFAULT_Z:
      common_cmd += ['-Z', str(self.z)]

    with TemporaryDirectory() as query_tmp_dir:
      if self.split_bfd_uniclust:
        a3m = ""
        stdout = b""
        stderr = b""

        for db_path in self.databases:
          db_name = os.path.basename(db_path)
          a3m_path = os.path.join(query_tmp_dir, f'{db_name}_output.a3m')
          timing_key = f'HHblits query {db_name}'

          db_cmd = common_cmd + ['-d', db_path]
          db_a3m, db_stdout, db_stderr = self._query_hhblits(
              db_cmd, a3m_path, timing_key=timing_key)

          a3m += db_a3m
          stdout += db_stdout
          stderr += db_stderr
      else:
        for db_path in self.databases:
          common_cmd += ['-d', db_path]

        a3m, stdout, stderr = self._query_hhblits(
            common_cmd, os.path.join(query_tmp_dir, 'output.a3m'))

    raw_output = dict(
        a3m=a3m,
        output=stdout,
        stderr=stderr,
        n_iter=self.n_iter,
        e_value=self.e_value)

    return [raw_output]

  def _query_hhblits(self, cmd_prefix: list, a3m_path: str,
                     timing_key: str = "HHblits query"):
    cmd = cmd_prefix + ['-oa3m', a3m_path]

    logging.info('Launching subprocess "%s"', ' '.join(cmd))
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    with utils.timing(timing_key):
      stdout, stderr = process.communicate()
      retcode = process.wait()

    if retcode:
      # Logs have a 15k character limit, so log HHblits error line by line.
      logging.error('HHblits failed. HHblits stderr begin:')
      for error_line in stderr.decode('utf-8', errors="replace").splitlines():
        if error_line.strip():
          logging.error(error_line.strip())
      logging.error('HHblits stderr end')
      raise RuntimeError('HHblits failed\nstdout:\n%s\n\nstderr:\n%s\n' % (
          stdout.decode('utf-8', errors="replace"),
          stderr[:500_000].decode('utf-8', errors="replace")))

    with open(a3m_path) as f:
      a3m = f.read()

    return a3m, stdout, stderr
