#!/usr/bin/env python3
"""Counts the number of events in a LArCV dataset."""

import argparse
from pathlib import Path

from larcv import larcv  # pylint: disable=W0611
from ROOT import TFile  # pylint: disable=E0611
from tqdm import tqdm


def main(source, source_list, tree_names, suffix=None, replace=False):
    """Get the length of a LArCV datasets.

    Parameters
    ----------
    source : Union[str, List[str]]
        Path or list of paths to the input files
    source_list : str
        Path to a text file containing a list of data file paths
    tree_names : List[str]
        Names of the trees to measure each entry count from.
    suffix : str, optional
        Suffix for the output file(s)
    replace : bool, default False
        If True, replace existing output files
    """
    # If using source list, read it in
    if source_list is not None:
        with open(source_list, "r", encoding="utf-8") as f:
            source = f.read().splitlines()

    # Initialize a header for the output files
    header = "entry," + ",".join([f"{name}_count" for name in tree_names]) + "\n"

    # Generate tree and branch keys
    tree_keys = [f"{name}_tree" for name in tree_names]
    branch_keys = [f"{name}_branch" for name in tree_names]

    # Loop over the list of files in the input
    print(f"\nMeasuring tree entries in {len(source)} file(s):")
    for file_path in source:
        # Initialize the output file
        print(f"- Processing file: {file_path}")
        out_path = f"{Path(file_path).stem}_{suffix}.csv"
        if not replace and Path(out_path).exists():
            tqdm.write(f"- Output file {out_path} exists, skipping...")
            continue

        out_file = open(out_path, "w", encoding="utf-8")
        out_file.write(header)

        # Open the LArCV file
        f = TFile(file_path, "r")

        # Loop over the entries in the file
        num_entries = f.Get(tree_keys[0]).GetEntries()
        for entry in tqdm(range(num_entries)):
            # Write the entry number
            out_file.write(f"{entry}")

            # Loop over the tree names
            for tree_key, tree_branch in zip(tree_keys, branch_keys):
                # Get the tree entry
                tree = getattr(f, tree_key)
                tree.GetEntry(entry)

                # Get the branch size
                branch = getattr(tree, tree_branch)
                count = branch.size()

                # Write the count to the output file
                out_file.write(f",{count}")

            # End the line for this entry
            out_file.write("\n")

        # Close the input and output files
        f.Close()
        out_file.close()

        # Dump number for this file, increment
        tqdm.write(f"- Processed {num_entries} entries for file: {file_path}")


if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description="Measure LArCV tree sizes.")

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
        "--tree-names",
        help="TTree names used to measure the tree entries.",
        type=str,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--suffix",
        "-o",
        help="Suffix for the output file(s)",
        type=str,
        required=False,
        default="counts",
    )

    parser.add_argument(
        "--replace", help="Replace existing output files", action="store_true"
    )

    args = parser.parse_args()

    # Execute the main function
    main(args.source, args.source_list, args.tree_names, args.suffix, args.replace)
