#!/usr/bin/env python3
"""Builds a list of file which make a data run."""

import argparse

from larcv import larcv  # pylint: disable=W0611
from ROOT import TFile  # pylint: disable=E0611
from tqdm import tqdm


def main(source, source_list, output, run_number, tree_name):
    """Loops over a list of files and finds those which belong to a certain run.

    Parameters
    ----------
    source : Union[str, List[str]]
        Path or list of paths to the input files
    source_list : str
        Path to a text file containing a list of data file paths
    output : str
        Path to the output text file with the list of run files
    run_number : int
        Run number to look for
    tree_name : str
        Name of the tree to use as a reference to get the run number from.
        If not specified, takes the first tree in the list.
    """
    # If using source list, read it in
    if source_list is not None:
        with open(source_list, "r", encoding="utf-8") as f:
            source = f.read().splitlines()

    # Initialize the output text file
    out_file = open(output, "w", encoding="utf-8")

    # Loop over the list of files in the input
    print(f"\nLooking for run {run_number} in {len(source)} files:")
    run_files = []
    for file_path in tqdm(source):
        # Get the tree to get the number of entries from
        f = TFile(file_path, "r")
        if tree_name is None:
            key = [key.GetName() for key in f.GetListOfKeys()][0]
        else:
            key = f"{tree_name}_tree"
        branch_key = key.replace("_tree", "_branch")

        # Check the run number of the first entry in the file
        tree = getattr(f, key)
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


if __name__ == "__main__":
    # Parse the command-line arguments
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

    args = parser.parse_args()

    # Execute the main function
    main(args.source, args.source_list, args.output, args.run_number, args.tree_name)
