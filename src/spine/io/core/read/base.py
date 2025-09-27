"""Contains the data reader base class.

Data readers are used to extract specific entries from files and store their
data products into dictionaries to be used downstream.
"""

import glob
import os
from warnings import warn

import numpy as np

from spine.utils.logger import logger


class ReaderBase:
    """Parent reader class which provides common functions between all readers.

    This class provides these basic functions:
    1. Method to parse the requested file list or file list file into a list of
       paths to existing files (throws if nothing is found)
    2. Method to produce a list of entries in the file(s) as selected by the
       provided parameters, checks that they exist (throws if they do not)
    3. Essential `__len__` and `__getitem__` methods. Must define the
       `get` function in the inheriting class for both of them to work.

    Attributes
    ----------
    name : str
        Name of the reader, as requested in the configuration
    num_entries : int
        Total number of entries in the files provided
    entry_index : List[int]
        List of global indexes to cycle through
    file_paths : List[str]
        List of files to read data from
    file_offsets : List[int]
        Offsets between the global index and each individual file start index
    file_index : List[int]
        Index of the file each entry in entry_index lives in
    run_info : np.ndarray
        (run, subrun, event) triplets associated with each entry in the file list
    run_map : Dict[Tuple[int], int]
        Maps each available (run, subrun, event) triplet onto an entry_index index
    """

    name = ""
    num_entries = None
    entry_index = None
    file_paths = None
    file_offsets = None
    file_index = None
    run_info = None
    run_map = None

    def __len__(self):
        """Returns the number of entries in the file(s).

        Returns
        -------
        int
            Number of entries in the file
        """
        return len(self.entry_index)

    def __getitem__(self, idx):
        """Returns a specific entry in the file.

        Parameters
        ----------
        idx : int
            Integer entry ID to access

        Returns
        -------
        dict
            One entry-worth of data from the loaded files
        """
        return self.get(idx)

    def get(self, idx):
        """Placeholder to be defined by the daughter class."""
        raise NotImplementedError

    def process_file_paths(self, file_keys, limit_num_files=None, max_print_files=10):
        """Process list of files.

        Parameters
        ----------
        file_keys : list
            List of paths to the HDF5 files to be read
        limit_num_files : int, optional
            Integer limiting number of files to be taken per data directory
        max_print_files : int, default 10
            Maximum number of loaded file names to be printed
        """
        # Some basic checks
        assert file_keys is not None, "No input `file_keys` provided, abort."
        assert (
            limit_num_files is None or limit_num_files > 0
        ), "If `limit_num_files` is provided, it must be larger than 0."

        # If the file_keys points to a single text file, it must be a text
        # file containing a list of file paths. Parse it to a list.
        if isinstance(file_keys, str) and os.path.splitext(file_keys)[-1] == ".txt":
            # If the file list is a text file, extract the list of paths
            assert os.path.isfile(file_keys), (
                "If the `file_keys` are specified as a single string, "
                "it must be the path to a text file with a file list."
            )
            with open(file_keys, "r", encoding="utf-8") as f:
                file_keys = f.read().splitlines()

        # Convert the file keys to a list of file paths with glob
        self.file_paths = []
        if isinstance(file_keys, str):
            file_keys = [file_keys]
        for file_key in file_keys:
            file_paths = glob.glob(file_key)
            assert file_paths, f"File key {file_key} yielded no compatible path."
            for path in file_paths:
                if (
                    limit_num_files is not None
                    and len(self.file_paths) >= limit_num_files
                ):
                    break
                self.file_paths.append(path)

        self.file_paths = sorted(self.file_paths)

        # Print out the list of loaded files
        num_files = len(self.file_paths)
        file_list = " - " + "\n - ".join(self.file_paths[:max_print_files])
        file_list += "\n ... \n" if num_files > max_print_files else "\n"
        logger.info("Will load %d file(s):\n%s", num_files, file_list)

    def process_run_info(self):
        """Process the run information.

        Check the run information for duplicates and initialize a dictionary
        which map (run, subrun, event) triplets onto entry index.
        """
        # Check for duplicates
        if self.run_info is not None:
            assert len(self.run_info) == self.num_entries
            num_unique = len(np.unique(self.run_info, axis=0))
            assert num_unique == len(self.run_info), (
                "Cannot create a run map if (run, subrun, event) triplets "
                "are not unique in the dataset. Abort."
            )

        # If run_info is set, flip it into a map from info to entry
        self.run_map = None
        if self.run_info is not None:
            self.run_map = {tuple(v): i for i, v in enumerate(self.run_info)}

    def process_entry_list(
        self,
        n_entry=None,
        n_skip=None,
        entry_list=None,
        skip_entry_list=None,
        run_event_list=None,
        skip_run_event_list=None,
        allow_missing=False,
    ):
        """Create a list of entries that can be accessed by :meth:`__getitem__`.

        Parameters
        ----------
        n_entry : int, optional
            Maximum number of entries to load
        n_skip : int, optional
            Number of entries to skip at the beginning
        entry_list : list, optional
            List of integer entry IDs to add to the index
        skip_entry_list : list, optional
            List of integer entry IDs to skip from the index
        run_event_list: list((int, int, int)), optional
            List of (run, subrun, event) triplets to add to the index
        skip_run_event_list: list((int, int, int)), optional
            List of (run, subrun, event) triplets to skip from the index
        allow_missing : bool, default False
            If `True`, allows missing entries in the entry or event list

        Returns
        -------
        list
            List of integer entry IDs in the index
        """
        # Make sure the parameters are sensible
        if np.any(
            [
                n_entry is not None,
                n_skip is not None,
                entry_list is not None,
                skip_entry_list is not None,
                run_event_list is not None,
                skip_run_event_list is not None,
            ]
        ):
            assert (
                (n_entry is not None or n_skip is not None)
                ^ (entry_list is not None or skip_entry_list is not None)
                ^ (run_event_list is not None or skip_run_event_list is not None)
            ), (
                "Cannot specify `n_entry` or `n_skip` at the same time "
                "as `entry_list` or `skip_entry_list` or at the same time "
                "as `run_event_list` or `skip_run_event_list`."
            )

        if n_entry is not None or n_skip is not None:
            n_skip = n_skip if n_skip else 0
            n_entry = n_entry if n_entry else self.num_entries - n_skip
            assert n_skip + n_entry <= self.num_entries, (
                f"Mismatch between `n_entry` ({n_entry}), `n_skip` ({n_skip}) "
                f"and the number of entries in the files ({self.num_entries})."
            )

        assert not entry_list or not skip_entry_list, (
            "Cannot specify both `entry_list` and "
            "`skip_entry_list` at the same time."
        )

        assert not run_event_list or not skip_run_event_list, (
            "Cannot specify both `run_event_list` and "
            "`skip_run_event_list` at the same time."
        )

        # Create a list of entries to be loaded
        if n_entry or n_skip:
            entry_list = np.arange(self.num_entries)
            if n_skip > 0:
                entry_list = entry_list[n_skip:]
            if n_entry > 0:
                entry_list = entry_list[:n_entry]

        elif entry_list:
            entry_list = self.parse_entry_list(entry_list)
            assert np.all(
                entry_list < self.num_entries
            ), "Values in entry_list outside of bounds."

        elif run_event_list:
            self.process_run_info()
            run_event_list = self.parse_run_event_list(run_event_list)
            entry_list = []
            for i, (r, s, e) in enumerate(run_event_list):
                if not allow_missing or (r, s, e) in self.run_map:
                    entry_list.append(self.get_run_event_index(r, s, e))

            entry_list = np.unique(entry_list)

        elif skip_entry_list or skip_run_event_list:
            if skip_entry_list:
                skip_entry_list = self.parse_entry_list(skip_entry_list)
                assert np.all(
                    skip_entry_list < self.num_entries
                ), "Values in skip_entry_list outside of bounds."

            else:
                self.process_run_info()
                skip_run_event_list = self.parse_run_event_list(skip_run_event_list)
                skip_entry_list = []
                for i, (r, s, e) in enumerate(skip_run_event_list):
                    if not allow_missing or (r, s, e) in self.run_map:
                        skip_entry_list.append(self.get_run_event_index(r, s, e))

            entry_mask = np.ones(self.num_entries, dtype=bool)
            entry_mask[skip_entry_list] = False
            entry_list = np.where(entry_mask)[0]

        # Apply entry list to the indexes
        entry_index = np.arange(self.num_entries, dtype=np.int64)
        if entry_list is not None:
            entry_index = entry_index[entry_list]

            if self.run_info is not None:
                run_info = self.run_info[entry_list]
                self.run_map = {tuple(v): i for i, v in enumerate(run_info)}

        assert len(entry_index), "Must at least have one entry to load."

        logger.info("Total number of entries selected: %d\n", len(entry_index))

        self.entry_index = entry_index

    def get_run_event(self, run, subrun, event):
        """Returns an entry corresponding to a specific (run, subrun, event)
        triplet.

        Parameters
        ----------
        run : int
            Run number
        subrun : int
            Subrun number
        event : int
            Event number

        Returns
        -------
        data_blob : dict
            Ditionary of input data products corresponding to one event
        result_blob : dict
            Ditionary of result data products corresponding to one event
        """
        return self.get(self.get_run_event_index(run, subrun, event))

    def get_run_event_index(self, run, subrun, event):
        """Returns an entry index corresponding to a specific
        (run, subrun, event) triplet.

        Parameters
        ----------
        run : int
            Run number
        event : int
            Event number
        """
        # Get the appropriate entry index
        assert (
            self.run_map is not None
        ), "Must build a run map to get entries by (run, sunrun, event)."
        assert (
            run,
            subrun,
            event,
        ) in self.run_map, (
            f"Could not find (run={run}, subrun={subrun}, event={event})."
        )

        return self.run_map[(run, subrun, event)]

    def get_file_path(self, idx):
        """Returns the path to the file corresponding to a specific entry.

        Parameters
        ----------
        idx : int
            Integer entry ID to access

        Returns
        -------
        str
            Path to the file
        """
        return self.file_paths[self.get_file_index(idx)]

    def get_file_index(self, idx):
        """Returns the index of the file corresponding to a specific entry.

        Parameters
        ----------
        idx : int
            Integer entry ID to access

        Returns
        -------
        int
            Index of the file in the file list
        """
        return self.file_index[self.entry_index[idx]]

    def get_file_entry_index(self, idx):
        """Returns the index of an entry within the file it lives in,
        provided a global index over the list of files.

        Parameters
        ----------
        idx : int
            Integer entry ID to access

        Returns
        -------
        int
            Index of the entry in the file
        """
        file_idx = self.get_file_index(idx)
        offset = self.file_offsets[file_idx]
        return self.entry_index[idx] - offset

    @staticmethod
    def parse_entry_list(list_source):
        """Parses a list into an np.ndarray.

        The list can be passed as a simple python list or a path to a file
        which contains space or comma separated numbers (can be on multiple
        lines or not)

        Parameters
        ----------
        list_source : Union[list, str]
            List as a python list or a text file path

        Returns
        -------
        np.ndarray
            List as a numpy array
        """
        if list_source is None:
            return np.empty(0, dtype=np.int64)

        if not np.isscalar(list_source):
            return np.asarray(list_source, dtype=np.int64)

        if isinstance(list_source, str):
            assert os.path.isfile(list_source), "The list source file does not exist."
            with open(list_source, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
                line_list = [l.replace(",", " ").split() for l in lines]
                list_source = [int(w) for l in line_list for w in l]

            return np.array(list_source, dtype=np.int64)

        raise ValueError("List format not recognized.")

    @staticmethod
    def parse_run_event_list(list_source):
        """Parses a list of (run, subrun, event) triplets into an np.ndarray.

        The list can be passed as a simple python list or a path to a file
        which contains one (run, subrun, event) pair per line.

        Parameters
        ----------
        list_source : Union[list, str]
            List as a python list or a text file path

        Returns
        -------
        Tuple[Tuple[int]]
            List as a numpy array
        """
        if list_source is None:
            return ()

        if not np.isscalar(list_source):
            return tuple(tuple(val) for val in list_source)

        if isinstance(list_source, str):
            assert os.path.isfile(list_source), "The list source file does not exist."
            with open(list_source, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
                line_list = [l.replace(",", " ").split() for l in lines]
                list_source = [(int(r), int(s), int(e)) for r, s, e in line_list]

            return tuple(tuple(val) for val in list_source)

        raise ValueError("List format not recognized.")
