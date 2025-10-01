#!/usr/bin/env python3
"""Check that all the input files were processed and produced an output with
the expected number of events for each input file."""

import argparse
import os

import h5py
import numpy as np
from larcv import larcv  # pylint: disable=W0611
from ROOT import TFile  # pylint: disable=E0611
from tqdm import tqdm


def main(
    source, source_list, output, dest, suffix, event_list, tree_name, larcv_output
):
    """Checks the output of the SPINE process.

    The script loops over the input files, check that there is an output file
    in the expected location and further checks that the output file entry
    count matches that of the input file.

    Produces a list of input files that have no or incomplete output in a text
    file (the name of which is provided with the `-o` or `--output` flag). This
    can be used to reprocess missing/incomplete input files.

    .. code-block:: bash

        $ python3 bin/output_check_valid.py -S file_list.txt -o missing.txt
          --dest /path/to/output/files/ --suffix output_file_suffix

    Parameters
    ----------
    source : List[str]
        List of paths to the input files
    source_list : str
        Path to a text file containing a list of data file paths
    output : str
        Path to the output text file with the list of badly processed files
    dest : str
        Destination directory for the original SPINE process
    suffix : str
        Suffix added to the end of the input files by the original SPINE process
    event_list : str
        Path to a file containing a list of events to process. If provided, only
        events which appear on this list are required for in the output.
    tree_name : str
        Name of the tree to use as a reference to count the number of entries.
        If not specified, takes the first tree in the list.
    larcv_output, bool
        If `True`, the output file is also a ROOT file
    """
    # If using source list, read it in
    if source_list is not None:
        with open(source_list, "r", encoding="utf-8") as f:
            source = f.read().splitlines()

    # Initialize the output text file
    out_file = open(output, "w", encoding="utf-8")

    # If it is provided, parse the list of (run, subrun, event) triplets
    if event_list is not None:
        with open(event_list, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            line_list = [l.replace(",", " ").split() for l in lines]
            event_list = [(int(r), int(s), int(e)) for r, s, e in line_list]

    # Loop over the list of files in the input
    print("\nChecking existence and completeness of output files.")
    miss_list, inc_list = [], []
    for idx, file_path in enumerate(tqdm(source)):
        # Find the base name of the input file (without extension)
        base = os.path.basename(file_path)
        stem, _ = os.path.splitext(base)

        # Check that the output exists under the expected path
        ext = ".h5" if not larcv_output else ".root"
        out_base = f"{stem}_{suffix}{ext}"
        out_path = f"{dest}/{out_base}"
        if not os.path.isfile(out_path):
            tqdm.write(f"- Missing: {out_base}")
            out_file.write(f"{file_path}\n")
            miss_list.append(file_path)
            continue

        # If the output does exist, check that the input and output have the
        # expected number of entries. If ROOT, get the tree name first.
        larcv_input = file_path.endswith(".root")
        if larcv_input:
            f = TFile(file_path, "r")
            if tree_name is None:
                key = [key.GetName() for key in f.GetListOfKeys()][0]
            else:
                key = f"{tree_name}_tree"
            key_b = key.replace("_tree", "_branch")

        # Dispatch depending if the event list is provided or not
        if event_list is None:
            # Count the number of entries in the input file
            if larcv_input:
                num_entries = getattr(f, key).GetEntries()
                f.Close()
            else:
                with h5py.File(file_path) as f:
                    num_entries = len(f["events"])

            # Then check the number of events in the output file
            if larcv_output:
                f = TFile(out_path, "r")
                out_num_entries = getattr(f, key).GetEntries()
                f.Close()
            else:
                with h5py.File(out_path) as f:
                    out_num_entries = len(f["events"])

            if out_num_entries != num_entries:
                tqdm.write(f"- Incomplete: {out_base}")
                out_file.write(f"{file_path}\n")
                inc_list.append(file_path)

        else:
            # Fetch the list of (run, subrun, event) triplets that should appear
            check_list = []
            if larcv_input:
                tree = getattr(f, key)
                for i in range(tree.GetEntries()):
                    tree.GetEntry(i)
                    branch = getattr(tree, key_b)
                    run, subrun, event = branch.run(), branch.subrun(), branch.event()
                    if (run, subrun, event) in event_list:
                        check_list.append((run, subrun, event))
                f.Close()
            else:
                with h5py.File(file_path) as f:
                    for run, subrun, event in f["run_info"]:
                        check_list.append((run, subrun, event))

            # Check that the events which should appear are present
            with h5py.File(out_path) as f:
                if len(f["events"]) != len(check_list):
                    tqdm.write(f"- Incomplete: {out_base}")
                    out_file.write(f"{file_path}\n")
                    inc_list.append(file_path)

    num_miss = len(miss_list)
    num_inc = len(inc_list)
    print(f"\nFound {num_miss + num_inc} problematic output file(s):")
    print(f"- {num_miss} missing output file(s);")
    print(f"- {num_inc} incomplete output file(s).")

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
        help="Path to the output text file with the problematic list",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dest",
        help="Destination directory for the original SPINE process",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--suffix",
        help="Suffix added to the input files by the original SPINE process",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--event-list", help="File containing a list of events to process.", type=str
    )

    parser.add_argument(
        "--tree-name", help="TTree name used to count the entries.", type=str
    )

    parser.add_argument(
        "--larcv-output",
        help="The output of the process is another LArCV file",
        action="store_true",
    )

    args = parser.parse_args()

    # Execute the main function
    main(
        args.source,
        args.source_list,
        args.output,
        args.dest,
        args.suffix,
        args.event_list,
        args.tree_name,
        args.larcv_output,
    )
