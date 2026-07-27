#!/usr/bin/env python3
"""Builds a list of file which make a data run."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from larcv import larcv  # pylint: disable=W0611
from ROOT import TFile  # pylint: disable=E0611
from tqdm import tqdm
from utils import get_branch_key, get_tree, get_tree_key, resolve_sources


def main(
    source: Sequence[str] | None,
    source_list: str | None,
    output: str,
    run_number: int,
    tree_name: str | None,
) -> None:
    """Loops over a list of files and finds those which belong to a certain run.

    Parameters
    ----------
    source : sequence of str, optional
        Path or list of paths to the input files
    source_list : str, optional
        Path to a text file containing a list of data file paths
    output : str
        Path to the output text file with the list of run files
    run_number : int
        Run number to look for
    tree_name : str, optional
        Name of the tree to use as a reference to get the run number from.
        If not specified, takes the first tree in the list.
    """
    source = resolve_sources(source, source_list)

    # Initialize the output text file
    out_file = open(output, "w", encoding="utf-8")

    # Loop over the list of files in the input
    print(f"\nLooking for run {run_number} in {len(source)} files:")
    run_files = []
    for file_path in tqdm(source):
        # Get the tree to get the number of entries from
        f = TFile(file_path, "r")
        key = get_tree_key(f, tree_name)
        branch_key = get_branch_key(key)

        # Check the run number of the first entry in the file
        tree = get_tree(f, key)
        tree.GetEntry(0)
        run = getattr(tree, branch_key).run()
        f.Close()

        # If the file contains entries from the correct run, append
        if run == run_number:
            tqdm.write(f"- Good file: {file_path}")
            run_files.append(file_path)
            out_file.write(f"{file_path}\n")

    print(f"\nFound {len(run_files)} run {run_number} files.")

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
        help="Path to the output text file with the run file list",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--run-number", help="Run number to look for", type=int, required=True
    )

    parser.add_argument(
        "--tree-name", help="TTree name used to count the entries.", type=str
    )

    return parser


def cli() -> None:
    """Run the command-line interface."""
    args = build_parser().parse_args()
    main(args.source, args.source_list, args.output, args.run_number, args.tree_name)


if __name__ == "__main__":
    cli()
