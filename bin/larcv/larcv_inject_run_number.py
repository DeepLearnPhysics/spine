"""Script which injects a run number in every event of every tree in a file or
a list of files.
"""

import argparse
import os
import tempfile

import numpy as np
from larcv import larcv  # pylint: disable=W0611
from ROOT import TFile  # pylint: disable=E0611
from tqdm import tqdm

# LArCV IO Manager configuration string
CFG = """
IOManager: {
    Verbosity   : 4
    Name        : "OutIO"
    IOMode      : 2
    InputFiles  : [INPUT_PATH]
    OutFileName : OUTPUT_PATH
}
"""


def initialize_manager(file_path, dest, overwrite, suffix):
    """Initialize an IOManager object given a configuration.

    Parameters
    ----------
    file_path : str
        Path to the input file
    dest : str
        Destination folder to write the output file to
    overwrite : bool
        If `True`, overwrite the original file
    suffix : str
        Suffix to append to the input file name to form the output file name

    Returns
    -------
    larcv.IOManager
        IOManager object
    """
    # If the destination is provided, direct the output file there
    out_path = file_path
    if dest is not None:
        base = os.path.basename(file_path)
        out_path = f"{dest}/{base}"

    # If a suffix is provided, append
    assert (
        suffix is None or not overwrite
    ), "No point in providing a suffix if the original file is overwritten."
    if suffix is not None:
        out_path = out_path.replace(".root", f"_{suffix}.root")
    elif overwrite:
        out_path = out_path.replace(".root", "_tmp.root")

    # Check that the output file does is not the same as the original file
    if file_path == out_path:
        raise ValueError(
            "The input file name and the output file name are the same. "
            "This is not allowed by the LArCV IOManager."
        )

    # Update the configuration with the input/output file names
    cfg = CFG
    cfg = cfg.replace("INPUT_PATH", file_path)
    cfg = cfg.replace("OUTPUT_PATH", out_path)

    # Create a temporary text file with the configuration
    tmp = tempfile.NamedTemporaryFile("w")
    tmp.write(cfg)
    tmp.flush()

    # Initialize the IOManager
    manager = larcv.IOManager(tmp.name)
    manager.initialize()

    return manager, out_path


def main(source, source_list, dest, overwrite, run_number, run_list, offset, suffix):
    """Checks the output of the SPINE process.

    The script loops over the input files, fetch the list of keys in the file
    and injects a run number of each event in each file.

    .. code-block:: bash

        $ python3 bin/inject_run_number.py -S file_list.txt
          --overwrite --run_number 123

    Parameters
    ----------
    source : List[str]
        List of paths to the input files
    source_list : str
        Path to a text file containing a list of data file paths
    dest : str
        Destination folder to write the files to
    overwrite : bool
        If `True`, overwrite the original files
    run_number : int
        Run number to inject in the input file list. If it is specied as -1,
        each file is assigned a unique run number
    run_list : str
        Path to a text file containing a list of run numbers to assign to each
        input file
    offset : int
        Offset to add to the existing run number for each successive file
    suffix : str
        String to append to the end of the input file names to form the name
        of the output file with the updated run numbers
    """
    # If using source list, read it in
    if source_list is not None:
        with open(source_list, "r", encoding="utf-8") as f:
            source = f.read().splitlines()

    # If using run list, read it in
    run_numbers = None
    if run_list is not None:
        with open(run_list, "r", encoding="utf-8") as f:
            run_numbers = f.read().splitlines()
        run_numbers = [int(r) for r in run_numbers]
        if len(run_numbers) != len(source):
            raise ValueError(
                "The number of run numbers provided does not match the number "
                "of input files."
            )

    # Loop over the list of files in the input
    print("\nUpdating the run numbers of input files.")
    for idx, file_path in enumerate(tqdm(source)):
        # Initialize the input/output processes
        io, out_path = initialize_manager(file_path, dest, overwrite, suffix)

        # Loop over entries, set the run number for every data product
        num_entries = io.get_n_entries()
        for e in range(num_entries):
            # Read existing content
            io.read_entry(e)

            # Fetch the run, subrun and event numbers
            io.get_data(e)
            event_id = io.event_id()
            run, subrun, event = event_id.run(), event_id.subrun(), event_id.event()

            # Update the run number
            if run_number is not None:
                if run_number > -1:
                    io.set_id(run_number, subrun, event)
                else:
                    io.set_id(idx, subrun, event)
            elif run_numbers is not None:
                io.set_id(run_numbers[idx], subrun, event)
            else:
                io.set_id(run + offset, subrun, event)

            # Save
            io.save_entry()

        # Finalize
        io.finalize()

        # If needed move the output file to where the input file is
        if overwrite:
            os.rename(out_path, file_path)


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

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dest", help="Destination folder for the output file", type=str
    )
    group.add_argument(
        "--overwrite",
        help="Overwrite the input file with the output file",
        action="store_true",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-number",
        help="Run number to assign to every input file. If -1, each file is "
        "assigned a unique run number",
        type=int,
    )
    group.add_argument(
        "--run-list",
        help="Path to a text file containing a list of run numbers to assign "
        "to each input file",
        type=str,
    )
    group.add_argument(
        "--offset",
        help="Offset to add to the existing run number for each successive file",
        type=int,
    )

    parser.add_argument(
        "--suffix", help="Suffix to append to the input file names", type=str
    )

    args = parser.parse_args()

    # Execute the main function
    main(
        args.source,
        args.source_list,
        args.dest,
        args.overwrite,
        args.run_number,
        args.run_list,
        args.offset,
        args.suffix,
    )
