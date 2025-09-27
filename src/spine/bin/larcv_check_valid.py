#!/usr/bin/env python3
"""Mark all bad LArCV ROOT files before merging them with hadd."""

import argparse

import numpy as np
from larcv import larcv  # pylint: disable=W0611
from ROOT import TFile  # pylint: disable=E0611
from tqdm import tqdm


def main(source, source_list, output):
    """Checks the validity of a LArCV root file.

    This script loops over all TTrees in a given ROOT file and check that they
    have the same, non-zero, number of entries.

    Produces a list of bad files in 'bad_files.txt' (one per line) that can then
    be used to move/remove these bad files before doing hadd. For example using:

    .. code-block:: bash

        $ for file in $(cat bad_files.txt); do mv "$file" bad_files/; done

    Parameters
    ----------
    source : List[str]
        List of paths to the input files
    source_list : str
        Path to a text file containing a list of data file paths
    output : str
        Path to the output text file with the list of bad files
    """
    # If using source list, read it in
    if source_list is not None:
        with open(source_list, "r", encoding="utf-8") as f:
            source = f.read().splitlines()

    # Initialize the output text file
    out_file = open(output, "w", encoding="utf-8")

    # Loop over the list of files in the input, count the tree entries for each
    print(f"\nCounting entries in every tree of {len(source)} files.")
    keys_list, unique_counts = [], []
    for file_path in tqdm(source):
        # Count the number of entries in each tree
        try:
            f = TFile(file_path)
        except OSError:
            keys_list.append([])
            unique_counts.append([])
            continue

        keys = [key.GetName() for key in f.GetListOfKeys()]
        trees = [f.Get(key) for key in keys]
        num_entries = [tree.GetEntries() for tree in trees]
        f.Close()

        keys_list.append(keys)
        unique_counts.append(np.unique(num_entries))

    # Get the all the unique tree names encountered in the list of files
    all_keys = np.unique(np.concatenate(keys_list))

    # Loop over the list of keys/counts for each file in the input
    print(f"\nChecking validity of {len(source)} file(s).")
    bad_files = []
    for idx, file_path in enumerate(tqdm(source)):
        # Check that there is only one entry count and it's non-zero, and
        # that the list of keys matches expectation
        if (
            len(unique_counts[idx]) != 1
            or unique_counts[idx][0] < 1
            or (set(keys_list[idx]) != set(all_keys))
        ):
            tqdm.write(f"- Bad file: {file_path}")
            out_file.write(f"{file_path}\n")
            bad_files.append(file_path)

    print(f"\nFound {len(bad_files)} bad files.")

    # Close text file
    out_file.close()


if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description="Check dataset validity")

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
        help="Path to the output text file with the bad list",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    # Execute the main function
    main(args.source, args.source_list, args.output)
