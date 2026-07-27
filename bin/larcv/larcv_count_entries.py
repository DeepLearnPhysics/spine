#!/usr/bin/env python3
"""Counts the number of events in a LArCV dataset."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from larcv import larcv  # pylint: disable=W0611
from ROOT import TFile  # pylint: disable=E0611
from tqdm import tqdm
from utils import get_tree, get_tree_key, resolve_sources


def main(
    source: Sequence[str] | None,
    source_list: str | None,
    tree_name: str | None,
) -> None:
    """Checks the number of entries in a file/list of files.

    Parameters
    ----------
    source : sequence of str, optional
        Path or list of paths to the input files
    source_list : str, optional
        Path to a text file containing a list of data file paths
    tree_name : str, optional
        Name of the tree to use as a reference to count the number of entries.
        If not specified, takes the first tree in the list.
    """
    source = resolve_sources(source, source_list)

    # Loop over the list of files in the input
    total_entries = 0
    print(f"\nCounting entries in {len(source)} file(s):")
    for file_path in tqdm(source):
        # Get the tree to get the number of entries from
        f = TFile(file_path, "r")
        key = get_tree_key(f, tree_name)

        # Count the number of entries in this file
        num_entries = get_tree(f, key).GetEntries()
        f.Close()

        # Dump number for this file, increment
        tqdm.write(f"- Counted {num_entries} entries in {file_path}")
        total_entries += num_entries

    print(f"\nTotal number of entries: {total_entries}")


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
        "--tree-name", help="TTree name used to count the entries.", type=str
    )

    return parser


def cli() -> None:
    """Run the command-line interface."""
    args = build_parser().parse_args()
    main(args.source, args.source_list, args.tree_name)


if __name__ == "__main__":
    cli()
