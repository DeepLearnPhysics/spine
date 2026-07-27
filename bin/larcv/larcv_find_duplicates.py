#!/usr/bin/env python3
"""Finds duplicated files."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
from larcv import larcv  # pylint: disable=W0611
from ROOT import TFile  # pylint: disable=E0611
from tqdm import tqdm
from utils import get_branch_key, get_tree, get_tree_key, resolve_sources


def main(
    source: Sequence[str] | None,
    source_list: str | None,
    output: str,
    tree_name: str | None,
) -> None:
    """Loops over a list of files and identifies files which contain the same
    set of (run, subrun, event) triplets.

    In order to save time, this script only checks if:
    1. The number of entries in the files are the same
    2. The run, subrun and event numbers in the first entry are the same

    Parameters
    ----------
    source : sequence of str, optional
        Path or list of paths to the input files
    source_list : str, optional
        Path to a text file containing a list of data file paths
    output : str
        Path to the output text file with the list of duplicates
    tree_name : str, optional
        Name of the tree to use as a reference to count the number of entries.
        If not specified, takes the first tree in the list.
    """
    source = resolve_sources(source, source_list)

    # Initialize the output text file
    out_file = open(output, "w", encoding="utf-8")

    # Loop over the list of files in the input
    print(f"\nGathering information from {len(source)} files:")
    values = np.empty((len(source), 4), dtype=int)
    for idx, file_path in enumerate(tqdm(source)):
        # Get the tree to get the number of entries from
        f = TFile(file_path, "r")
        key = get_tree_key(f, tree_name)
        branch_key = get_branch_key(key)

        # Check the number of entries in the file
        tree = get_tree(f, key)
        num_entries = tree.GetEntries()

        # Get the event information of the first entry in the file
        tree.GetEntry(0)
        branch = getattr(tree, branch_key)
        run, subrun, event = branch.run(), branch.subrun(), branch.event()

        # Set the values list
        values[idx] = [num_entries, run, subrun, event]
        f.Close()

    # Loop over non-unique files
    print(f"\nChecking for duplicates among {len(source)} files:")
    _, inverse, counts = np.unique(
        values, axis=0, return_inverse=True, return_counts=True
    )
    duplicate_files = []
    for idx in tqdm(np.where(counts > 1)[0]):
        # Build a file mask for this class of duplicates
        index = np.where(inverse == idx)[0]

        # All the files which are not the first in this class are duplicates
        tqdm.write(f"- File {source[index[0]]} is duplicated:")
        for i in range(1, len(index)):
            file_path = source[index[i]]
            duplicate_files.append(file_path)
            out_file.write(f"{file_path}\n")
            tqdm.write(f"  - Duplicate file: {file_path}")

    print(f"\nFound {len(duplicate_files)} duplicate files.")

    # Close text file
    out_file.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Count entries in dataset")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--source",
        "-s",
        help="Path or list of paths to data files",
        type=str,
        nargs="+",
    )
    group.add_argument(
        "--source-list", "-S", help="Path to a text file of data file paths", type=str
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Path to the output text file with the duplicate list",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--tree-name", help="TTree name used to count the entries.", type=str
    )

    return parser


def cli() -> None:
    """Run the command-line interface."""
    args = build_parser().parse_args()
    main(args.source, args.source_list, args.output, args.tree_name)


if __name__ == "__main__":
    cli()
