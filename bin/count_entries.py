#!/usr/bin/env python3
"""Counts the number of events in a LArCV dataset."""

import argparse

from ROOT import TFile
from larcv import larcv


def main(tree_name, source):
    """Checks the number of entries in a file/list of files.

    Parameters
    ----------
    tree_name : str
        Name of the tree to use as a reference to count the number of entries
    source : Union[str, List[str]]
        Path or list of paths to the input files
    """
    total_entries = 0
    print(f"\nCounting entries in {len(source)} files:")
    for idx, file_path in enumerate(source):
        # Count the number of entries in this file
        f = TFile(file_path, 'r')
        num_entries = getattr(f, f'{tree_name}_tree').GetEntries()
        f.Close()

        # Dump number for this file, increment
        print(f"- Counted {num_entries} entries in {file_path}")
        total_entries += num_entries

    print(f"\nTotal number of entries: {total_entries}")


if __name__ == "__main__":
    # Parse the command-line arguments
    argparse = argparse.ArgumentParser(description="Count entries in dataset")
    argparse.add_argument('tree_name',
                          help='TTree name used to count the entries.',
                          type=str)
    argparse.add_argument('source', '-s',
                          help='Path or list of paths to data files',
                          type=str, nargs="+")
    args = argparse.parse_args()

    # Execture the main function
    main(args.tree_name, args.source)
