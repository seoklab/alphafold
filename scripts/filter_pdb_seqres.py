#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import List

import pandas as pd

gpcrdb_path = Path(__file__).resolve().parent / "GPCRdb.csv"


def chain_list_from_df(df: pd.DataFrame, state: str):
    """
    Converts a dataframe with a column of chain names to a list of chain names.
    """
    state_pdbs = df[df["State"] == state]
    pdb_chains = state_pdbs[["PDB", "Preferred"]]
    return [f"{pdb.lower()}_{chain}"
            for pdb, chain in pdb_chains.itertuples(index=False, name=None)]


def filter_seqres(seqres_path: str, chains: List[str]) -> List[str]:
  chains_set = set(chains)
  ret = []
  with open(seqres_path) as f:
    for line in f:
      if line[1:7] in chains_set:
        ret += [line, next(f)]
  return ret


def main():
  seqres_path = sys.argv[1]
  seqres_dir = Path(seqres_path).resolve(True).parent

  gpcrdb = pd.read_csv(gpcrdb_path)
  for state in ["Active", "Inactive", "Intermediate"]:
    chains = chain_list_from_df(gpcrdb, state)
    filtered_seqres = filter_seqres(seqres_path, chains)
    with open(seqres_dir / f"pdb_seqres_{state.lower()}.txt", "w") as f:
      f.write("".join(filtered_seqres))


if __name__ == "__main__":
  main()
