#!/usr/bin/env python3
"""Counts the number of events in a LArCV dataset."""

import argparse

from ROOT import TFile # pylint: disable=E0611
from larcv import larcv # pylint: disable=W0611


def main(tree_name, source, source_list):
    """Checks the number of entries in a file/list of files.

    Parameters
    ----------
    tree_name : str
        Name of the tree to use as a reference to count the number of entries
    source : Union[str, List[str]]
        Path or list of paths to the input files
    source_list : str
        Path to a text file containing a list of data file paths
    """
    # If using source list, read it in
    if source_list is not None:
        with open(source_list, 'r', encoding='utf-8') as f:
            source = f.read().splitlines()

    # Loop over the list of files in the input
    total_entries = 0
    print(f"\nCounting entries in {len(source)} file(s):")
    for file_path in source:
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
    parser = argparse.ArgumentParser(description="Count entries in dataset")

    parser.add_argument('tree_name',
                          help='TTree name used to count the entries.',
                          type=str)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source', '-s',
                       help='Path or list of paths to data files',
                       type=str, nargs="+")
    group.add_argument('--source-list', '-S',
                       help='Path to a text file of data file paths',
                       type=str)

    args = parser.parse_args()

    # Execute the main function
    main(args.tree_name, args.source, args.source_list)
