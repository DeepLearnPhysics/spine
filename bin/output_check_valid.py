#!/usr/bin/env python3
"""Validate SPINE outputs against the list of source input files.

The script checks that each expected output file exists and then verifies that
the output is complete.

For modern HDF5 outputs, completeness is determined primarily from explicit
writer metadata:

- ``info.attrs["complete"]`` when present
- top-level ``source`` provenance when present

If those markers are absent, the script falls back to the legacy heuristic of
matching output file names and comparing input/output entry counts.
"""

import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm

try:
    from larcv import larcv  # pylint: disable=W0611
    from ROOT import TFile  # pylint: disable=E0611
except ImportError:  # pragma: no cover - exercised in test_bin without ROOT/LArCV
    larcv = None
    TFile = None


def require_root_larcv():
    """Require ROOT/LArCV support for ROOT-based validation paths.

    Raises
    ------
    ImportError
        If the current environment cannot import the ROOT/LArCV stack needed
        to inspect LArCV files.
    """
    if TFile is None or larcv is None:
        raise ImportError(
            "ROOT and larcv are required to validate ROOT/LArCV inputs or outputs."
        )


def get_num_entries(file_path, tree_name=None):
    """Return the number of entries stored in one input or output file.

    Parameters
    ----------
    file_path : str
        Path to a ROOT or HDF5 file.
    tree_name : str, optional
        Name of the ROOT tree used to count entries. Ignored for HDF5 files.

    Returns
    -------
    int
        Number of event entries stored in the file.
    """
    larcv_input = file_path.endswith(".root")
    if larcv_input:
        require_root_larcv()
        f = TFile(file_path, "r")
        if tree_name is None:
            key = [key.GetName() for key in f.GetListOfKeys()][0]
        else:
            key = f"{tree_name}_tree"
        num_entries = getattr(f, key).GetEntries()
        f.Close()
        return num_entries

    with h5py.File(file_path) as f:
        return len(f["events"])


def has_modern_hdf5_markers(out_file):
    """Check whether an HDF5 output exposes modern completeness metadata.

    Parameters
    ----------
    out_file : h5py.File
        Open output HDF5 handle.

    Returns
    -------
    bool
        `True` if the file contains either the explicit ``complete`` marker or
        top-level source provenance used by newer SPINE writers.
    """
    has_complete = "info" in out_file and "complete" in out_file["info"].attrs
    has_source = "source" in out_file
    return has_complete or has_source


def check_hdf5_source_provenance(file_path, out_file):
    """Validate top-level source provenance when the output stores it.

    Parameters
    ----------
    file_path : str
        Path to the source input file that should have produced this output.
    out_file : h5py.File
        Open output HDF5 handle.

    Returns
    -------
    bool
        `True` if the output either has no top-level ``source`` group or if
        the stored source metadata matches the input file's basename, size, and
        modification time.
    """
    if "source" not in out_file:
        return True

    source_group = out_file["source"]
    expected_name = os.path.basename(file_path)
    stat_result = os.stat(file_path)

    file_name = source_group.attrs.get("file_name")
    if isinstance(file_name, bytes):
        file_name = file_name.decode()
    file_size = source_group.attrs.get("file_size")
    file_mtime_ns = source_group.attrs.get("file_mtime_ns")

    return (
        file_name == expected_name
        and int(file_size) == int(stat_result.st_size)
        and int(file_mtime_ns) == int(stat_result.st_mtime_ns)
    )


def is_valid_modern_hdf5_output(file_path, out_path):
    """Validate a modern HDF5 output using completeness and source metadata.

    Parameters
    ----------
    file_path : str
        Path to the source input file that should have produced this output.
    out_path : str
        Path to the output HDF5 file to validate.

    Returns
    -------
    bool or None
        Returns `True`/`False` when modern validation metadata is present and
        should be trusted. Returns `None` when the file is a legacy output and
        the caller should fall back to name/count-based validation.
    """
    with h5py.File(out_path) as out_file:
        if not has_modern_hdf5_markers(out_file):
            return None

        if "info" in out_file and "complete" in out_file["info"].attrs:
            if not out_file["info"].attrs["complete"]:
                return False

        return check_hdf5_source_provenance(file_path, out_file)


def main(
    source, source_list, output, dest, suffix, event_list, tree_name, larcv_output
):
    """Check the outputs of a SPINE processing campaign.

    The script loops over the requested input files, checks that an output file
    exists in the expected location, and then validates that output using one
    of two strategies:

    1. Modern HDF5 validation:
       use explicit ``complete`` and ``source`` metadata when available.
    2. Legacy validation:
       compare the number of expected input entries to the number of entries
       present in the output.

    The script writes a text file containing the paths of source files whose
    outputs are missing or incomplete. That list can then be fed back into a
    reprocessing campaign.

    .. code-block:: bash

        $ python3 bin/output_check_valid.py -S file_list.txt -o missing.txt
          --dest /path/to/output/files/ --suffix output_file_suffix

    Parameters
    ----------
    source : list[str]
        List of paths to the input files
    source_list : str
        Path to a text file containing a list of data file paths
    output : str
        Path to the output text file with the list of problematic files
    dest : str
        Destination directory used by the original SPINE process
    suffix : str
        Suffix added to input file stems by the original SPINE process
    event_list : str
        Path to a file containing a list of events to process. If provided,
        only events appearing on this list are required in the output.
    tree_name : str
        Name of the ROOT tree to use when counting entries. If not specified,
        the first tree found in each input ROOT file is used.
    larcv_output : bool
        If `True`, the output file is also a ROOT/LArCV file. Otherwise the
        output is assumed to be HDF5.
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

        # For modern HDF5 outputs, prefer the explicit completeness flag and
        # source provenance over inferred entry-count matching.
        if not larcv_output:
            is_valid = is_valid_modern_hdf5_output(file_path, out_path)
            if is_valid is False:
                tqdm.write(f"- Incomplete: {out_base}")
                out_file.write(f"{file_path}\n")
                inc_list.append(file_path)
                continue
            if is_valid is True:
                continue

        # Legacy fallback: compare expected and actual entry counts. If ROOT,
        # fetch the tree name first and optionally filter by event list.
        larcv_input = file_path.endswith(".root")
        if larcv_input:
            require_root_larcv()
            f = TFile(file_path, "r")
            if tree_name is None:
                key = [key.GetName() for key in f.GetListOfKeys()][0]
            else:
                key = f"{tree_name}_tree"
            key_b = key.replace("_tree", "_branch")

        # Dispatch depending if the event list is provided or not
        if event_list is None:
            # Count the number of entries in the input and legacy output.
            num_entries = get_num_entries(file_path, tree_name=tree_name)
            out_num_entries = get_num_entries(out_path, tree_name=tree_name)

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
