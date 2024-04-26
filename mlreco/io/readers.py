"""Contains data reader classes.

Data readers are used to extract specific entries from files and store their
data products into dictionaries to be used downstream.
"""
import os
import glob
from warnings import warn

import h5py
import ROOT
import numpy as np

import mlreco.data

__all__ = ['LArCVReader', 'HDF5Reader']


class Reader:
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
        Name of the reader, as defined in the configuration
    num_entries : str
        Total number of entries in the files provided
    file_paths : List[str]
        List of files to read data from
    entry_index : List[int]
        

    """
    name = ''
    num_entries = None
    file_paths = None
    entry_index = None
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

    def process_file_paths(self, file_keys, limit_num_files=None,
                           max_print_files=10):
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
        assert limit_num_files is None or limit_num_files > 0, (
                "If `limit_num_files` is provided, it must be larger than 0")

        # If the file_keys points to a single text file, it must be a text
        # file containing a list of file paths. Parse it to a list.
        if (isinstance(file_keys, str) and
            os.path.splitext(file_keys)[-1] == '.txt'):
            # If the file list is a text file, extract the list of paths
            assert os.path.isfile(file_keys), (
                    "If the `file_keys` are specified as a single string, "
                    "it must be the path to a text file with a file list")
            with open(file_keys, 'r', encoding='utf-8') as f:
                file_keys = f.read().splitlines()

        # Convert the file keys to a list of file paths with glob
        self.file_paths = []
        if isinstance(file_keys, str):
            file_keys = [file_keys]
        for file_key in file_keys:
            file_paths = glob.glob(file_key)
            assert file_paths, (
                    f"File key {file_key} yielded no compatible path")
            for path in file_paths:
                if (limit_num_files is not None and
                    len(self.file_paths) > limit_num_files):
                    break
                self.file_paths.append(path)

            if (limit_num_files is not None and
                len(self.file_paths) >= limit_num_files):
                break

        self.file_paths = sorted(self.file_paths)

        # Print out the list of loaded files
        print(f"Will load {len(self.file_paths)} file(s):")
        for i, path in enumerate(self.file_paths):
            if i < max_print_files:
                print("  -", path)
            elif i == max_print_files:
                print("  ...")
                break
        print("")

    def process_run_info(self):
        """Process the run information.

        Check the run information for duplicates and initialize a dictionary
        which map [run, event] pairs onto entry index.
        """
        # Check for duplicates
        if self.run_info is not None:
            assert len(self.run_info) == self.num_entries
            has_duplicates = not np.all(np.unique(self.run_info,
                axis = 0, return_counts=True)[-1] == 1)
            if has_duplicates:
                warn("There are duplicated [run, event] pairs")

        # If run_info is set, flip it into a map from info to entry
        self.run_map = None
        if self.run_info is not None:
            self.run_map = {tuple(v):i for i, v in enumerate(self.run_info)}

    def process_entry_list(self, n_entry=None, n_skip=None, entry_list=None,
                           skip_entry_list=None, run_event_list=None,
                           skip_run_event_list=None):
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
        run_event_list: list((int, int)), optional
            List of [run, event] pairs to add to the index
        skip_run_event_list: list((int, int)), optional
            List of [run, event] pairs to skip from the index

        Returns
        -------
        list
            List of integer entry IDs in the index
        """
        # Make sure the parameters are sensible
        if np.any([n_entry is not None, n_skip is not None,
                   entry_list is not None, skip_entry_list is not None,
                   run_event_list is not None,
                   skip_run_event_list is not None]):
            assert (bool(n_entry or n_skip) ^
                    bool(entry_list or skip_entry_list) ^
                    bool(run_event_list or skip_run_event_list)), (
                    "Cannot specify `n_entry` or `n_skip` at the same time "
                    "as `entry_list` or `skip_entry_list` or at the same time "
                    "as `run_event_list` or `skip_run_event_list`")

        if n_entry is not None or n_skip is not None:
            n_skip = n_skip if n_skip else 0
            n_entry = n_entry if n_entry else self.num_entries - n_skip
            assert n_skip + n_entry <= self.num_entries, (
                f"Mismatch between `n_entry` ({n_entry}), `n_skip` ({n_skip}) "
                f"and the number of entries in the files ({self.num_entries})")

        assert not entry_list or not skip_entry_list, (
                "Cannot specify both `entry_list` and "
                "`skip_entry_list` at the same time")

        assert not run_event_list or not skip_run_event_list, (
                "Cannot specify both `run_event_list` and "
                "`skip_run_event_list` at the same time")

        # Create a list of entries to be loaded
        if n_entry or n_skip:
            entry_list = np.arange(self.num_entries)
            if n_skip > 0:
                entry_list = entry_list[n_skip:]
            if n_entry > 0:
                entry_list = entry_list[:n_entry]

        elif entry_list:
            entry_list = self.parse_entry_list(entry_list)
            assert np.all(entry_list < self.num_entries), (
                    "Values in entry_list outside of bounds")

        elif run_event_list:
            run_event_list = self.parse_run_event_list(run_event_list)
            entry_list = np.empty(len(run_event_list), dtype=np.int64)
            for i, (r, e) in enumerate(run_event_list):
                entry_list[i] = self.get_run_event_index(r, e)

        elif skip_entry_list or skip_run_event_list:
            if skip_entry_list:
                skip_entry_list = self.parse_entry_list(skip_entry_list)
                assert np.all(skip_entry_list < self.num_entries), (
                        "Values in skip_entry_list outside of bounds")

            else:
                skip_run_event_list = self.parse_run_event_list(
                        skip_run_event_list)
                skip_entry_list = np.empty(
                        len(skip_run_event_list), dtype=np.int64)
                for i, (r, e) in enumerate(skip_run_event_list):
                    skip_entry_list[i] = self.get_run_event_index(r, e)

            entry_mask = np.ones(self.num_entries, dtype=bool)
            entry_mask[skip_entry_list] = False
            entry_list = np.where(entry_mask)[0]

        # Apply entry list to the indexes
        entry_index = np.arange(self.num_entries, dtype=np.int64)
        if entry_list is not None:
            entry_index = entry_index[entry_list]

            if self.run_info is not None:
                run_info = self.run_info[entry_list]
                self.run_map = {tuple(v):i for i, v in enumerate(run_info)}

        assert len(entry_index), "Must at least have one entry to load"

        self.entry_index = entry_index

    def get_run_event(self, run, event):
        """Returns an entry corresponding to a specific (run, event) pair.

        Parameters
        ----------
        run : int
            Run number
        event : int
            Event number

        Returns
        -------
        data_blob : dict
            Ditionary of input data products corresponding to one event
        result_blob : dict
            Ditionary of result data products corresponding to one event
        """
        return self.get(self.get_run_event_index(run, event))

    def get_run_event_index(self, run, event):
        """Returns an entry index corresponding to a specific (run, event) pair.

        Parameters
        ----------
        run : int
            Run number
        event : int
            Event number
        """
        # Get the appropriate entry index
        assert self.run_map is not None, (
                "Must build a run map to get entries by [run, event].")
        assert (run, event) in self.run_map, (
                f"Could not find (run={run}, event={event}) pair.")

        return self.run_map[(run, event)]

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
            assert os.path.isfile(list_source), (
                    "The list source file does not exist")
            with open(list_source, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
                list_source = [
                        w for l in lines for w in l.replace(',', ' ').split()]

            return np.array(list_source, dtype=np.int64)

        raise ValueError("List format not recognized")

    @staticmethod
    def parse_run_event_list(list_source):
        """Parses a list of [run, event] pairs into an np.ndarray.

        The list can be passed as a simple python list or a path to a file
        which contains one [run, event] pair per line.

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
            return np.empty((0, 2), dtype=np.int64)

        if not np.isscalar(list_source):
            return np.asarray(list_source, dtype=np.int64).reshape(-1, 2)

        if isinstance(list_source, str):
            assert os.path.isfile(list_source), (
                    "The list source file does not exist")
            with open(list_source, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
                list_source = [l.replace(',', ' ').split() for l in lines]

            return np.array(list_source, dtype=np.int64)

        raise ValueError("List format not recognized")


class LArCVReader(Reader):
    """Class which reads information stored in LArCV files.

    More documentation to come.
    """
    name = 'larcv'

    def __init__(self, file_keys, tree_keys, limit_num_files=None,
                 max_print_files=10, n_entry=None, n_skip=None,
                 entry_list=None, skip_entry_list=None, run_event_list=None,
                 skip_run_event_list=None, create_run_map=False,
                 run_info_key=None):
        """Initialize the LArCV file reader.

        Parameters
        ----------
        file_keys : list
            List of paths to the HDF5 files to be read
        tree_keys : List[str]
            List of data keys to load from the LArCV files
        limit_num_files : int, optional
            Integer limiting number of files to be taken per data directory
        max_print_files : int, default 10
            Maximum number of loaded file names to be printed
        n_entry : int, optional
            Maximum number of entries to load
        n_skip : int, optional
            Number of entries to skip at the beginning
        entry_list : list
            List of integer entry IDs to add to the index
        skip_entry_list : list
            List of integer entry IDs to skip from the index
        run_event_list: list((int, int)), optional
            List of [run, event] pairs to add to the index
        skip_run_event_list: list((int, int)), optional
            List of [run, event] pairs to skip from the index
        create_run_map : bool, default False
            Initialize a map between [run, event] pairs and entries. For large
            files, this can be quite expensive (must load every entry).
        run_info_key : str, optional
            Key of the tree in the file to get the run information from
        """
        # Process the file_paths
        self.process_file_paths(file_keys, limit_num_files, max_print_files)

        # If an entry list is requested based on run/event ID, create map
        if run_event_list is not None or skip_run_event_list is not None:
            create_run_map = True

        # Prepare TTrees and load files
        self.trees = {}
        self.trees_ready = False
        self.file_offsets = np.empty(len(self.file_paths), dtype=np.int64)
        file_counts = []
        for key in tree_keys:
            # Check data TTree exists, and entries are identical across all
            # trees. Do not register these TTrees in yet in order to support
            # > 1 workers by the DataLoader object downstrean.
            print("Loading tree", key)
            chain = ROOT.TChain(f'{key}_tree') # pylint: disable=E1101
            for i, f in enumerate(self.file_paths):
                self.file_offsets[i] = chain.GetEntries()
                chain.AddFile(f)
                if key == tree_keys[0]:
                    count = chain.GetEntries() - self.file_offsets[i]
                    file_counts.append(count)

            if self.num_entries is not None:
                assert self.num_entries == chain.GetEntries(), (
                        f"Mismatch between the number of entries for {key} "
                        f"({chain.GetEntries()}) and the number of entries "
                        f"in other data products ({self.num_entries}).")
            else:
                self.num_entries = chain.GetEntries()

            self.trees[key] = None
        print("")

        # Build a file index
        self.file_index = np.repeat(
                np.arange(len(self.file_paths)), file_counts)

        # If requested, must extract the run information for each entry
        if create_run_map:
            # Initialize the TChain object
            assert run_info_key is not None and run_info_key in tree_keys, (
                    "Must provide the `run_info_key` if a run maps is needed. "
                    "The key must appear in the list of `tree_keys`")
            chain = ROOT.TChain(f'{run_info_key}_tree') # pylint: disable=E1101
            for f in self.file_paths:
                chain.AddFile(f)

            # Loop over entries
            self.run_info = np.empty((self.num_entries, 2), dtype=np.int64)
            for i in range(self.num_entries):
                chain.GetEntry(i)
                source = getattr(chain, f'{run_info_key}_branch')
                self.run_info[i] = [source.run(), source.event()]

        # Process the run information
        self.process_run_info()

        # Process the entry list
        self.process_entry_list(
                n_entry, n_skip, entry_list, skip_entry_list,
                run_event_list, skip_run_event_list)

    def get(self, idx):
        """Returns a specific entry in the file.

        Parameters
        ----------
        idx : int
            Integer entry ID to access

        Returns
        -------
        dict
            Dictionary which maps each data product key to an entry in the tree
        """
        # Get the appropriate entry index
        assert idx < len(self.entry_index)
        entry_idx = self.entry_index[idx]

        # If this is the first data loading, instantiate chains
        if not self.trees_ready:
            for key in self.trees:
                chain = ROOT.TChain(f'{key}_tree') # pylint: disable=E1101
                for f in self.file_paths:
                    chain.AddFile(f)
                self.trees[key] = chain
            self.trees_ready = True

        # Move the entry pointer
        for tree in self.trees.values():
            tree.GetEntry(entry_idx)

        # Load the relevant data products
        output = {}
        for key, tree in self.trees.items():
            output[key] = getattr(tree, f'{key}_branch')

        return output

    @staticmethod
    def list_data(file_path):
        """Dumps top-level information about the contents of a LArCV root file.

        Parameters
        ----------
        file_path : str
            Path to the file to scan

        Returns
        -------
        dict
            Dictionary which maps data types onto a list of keys
        """
        # Load up the file
        f = ROOT.TFile.Open(file_path, 'r') # pylint: disable=E1101

        # Loop over the list of keys
        data = {'sparse3d':[], 'cluster3d':[], 'particle':[]}
        for k in f.GetListOfKeys():
            # The the key name
            name = k.GetName()

            # Only look at tree names
            if not name.endswith('_tree'):
                continue
            if len(name.split('_')) < 3:
                continue

            # Get the data type name, skip if not recognized
            key = name.split('_')[0]
            if key not in data:
                continue

            # Append this specific tree name
            data[key].append(name[:name.rfind('_')])

        return data


class HDF5Reader(Reader):
    """Class which reads information stored in HDF5 files.

    More documentation to come.
    """
    name = 'hdf5'

    def __init__(self, file_keys, limit_num_files=None, max_print_files=10,
                 n_entry=None, n_skip=None, entry_list=None,
                 skip_entry_list=None, run_event_list=None,
                 skip_run_event_list=None, create_run_map=False,
                 split_categories=True, build_classes=True,
                 run_info_key='run_info'):
        """Initalize the HDF5 file reader.

        Parameters
        ----------
        file_keys : list
            List of paths to the HDF5 files to be read
        limit_num_files : int, optional
            Integer limiting number of files to be taken per data directory
        max_print_files : int, default 10
            Maximum number of loaded file names to be printed
        n_entry : int, optional
            Maximum number of entries to load
        n_skip : int, optional
            Number of entries to skip at the beginning
        entry_list : list
            List of integer entry IDs to add to the index
        skip_entry_list : list
            List of integer entry IDs to skip from the index
        run_event_list: list((int, int)), optional
            List of [run, event] pairs to add to the index
        skip_run_event_list: list((int, int)), optional
            List of [run, event] pairs to skip from the index
        create_run_map : bool, default False
            Initialize a map between [run, event] pairs and entries. For large
            files, this can be quite expensive (must load every entry).
        split_categories : bool, default True
            Split the data into the `data` and `result` dictionaries
        build_classes : bool, default True
            If the stored object is a class, build it back
        run_info_key : str, default 'run_info'
            Name of the data product which contains the run info of the event
        """
        # Process the list of files
        self.process_file_paths(file_keys, limit_num_files, max_print_files)

        # If an entry list is requested based on run/event ID, create map
        if run_event_list is not None or skip_run_event_list is not None:
            create_run_map = True

        # Loop over the input files, build a map from index to file ID
        self.num_entries  = 0
        self.file_index   = []
        self.file_offsets = np.empty(len(self.file_paths), dtype=np.int64)
        self.run_info     = None if not create_run_map else []
        for i, path in enumerate(self.file_paths):
            with h5py.File(path, 'r') as in_file:
                # Check that there are events in the file
                assert 'events' in in_file, (
                        "File does not contain an event tree")

                # If requested, register the [run, event] information pair
                if create_run_map:
                    assert run_info_key in in_file, (
                            f"Must provide {run_info_key} to create run map")
                    run_info = in_file[run_info_key]
                    for r, e in zip(run_info['run'], run_info['event']):
                        self.run_info.append([r, e])

                # Update the total number of entries
                num_entries = len(in_file['events'])
                self.file_index.append(i*np.ones(num_entries, dtype=np.int64))
                self.file_offsets[i] = self.num_entries
                self.num_entries += num_entries

        # Concatenate the file indexes into one
        self.file_index = np.concatenate(self.file_index)

        # Turn file index and run info into np.ndarray
        if self.run_info is not None:
            self.run_info = np.vstack(self.run_info)

        # Process the run information
        self.process_run_info()

        # Process the entry list
        self.process_entry_list(
                n_entry, n_skip, entry_list, skip_entry_list,
                run_event_list, skip_run_event_list)

        # Store other attributes
        self.split_categories = split_categories
        self.build_classes = build_classes

    def get(self, idx):
        """Returns a specific entry in the file.

        Parameters
        ----------
        idx : int
            Integer entry ID to access

        Returns
        -------
        data_blob : dict
            Ditionary of input data products corresponding to one event
        result_blob : dict
            Ditionary of result data products corresponding to one event
        """
        # Get the appropriate entry index
        assert idx < len(self.entry_index)
        file_idx  = self.get_file_index(idx)
        entry_idx = self.get_file_entry_index(idx)

        # Use the event tree to find out what needs to be loaded
        data_blob, result_blob = {}, {}
        with h5py.File(self.file_paths[file_idx], 'r') as in_file:
            event = in_file['events'][entry_idx]
            for key in event.dtype.names:
                self.load_key(in_file, event, data_blob, result_blob, key)

        # Return
        if self.split_categories:
            return data_blob, result_blob

        return dict(data_blob, **result_blob)

    def load_key(self, in_file, event, data_blob, result_blob, key):
        """Fetch a specific key for a specific event.

        Parameters
        ----------
        in_file : h5py.File
            HDF5 file instance
        event : dict
            Dictionary of objects that make up one event
        data_blob : dict
            Dictionary used to store the loaded input data
        result_blob : dict
            Dictionary used to store the loaded result data
        key: str
            Name of the dataset in the entry
        """
        # The event-level information is a region reference: fetch it
        region_ref = event[key]
        cat = in_file[key].attrs['category']
        blob = data_blob if cat == 'data' else result_blob
        if isinstance(in_file[key], h5py.Dataset):
            if not in_file[key].dtype.names:
                # If the reference points at a simple dataset, return
                blob[key] = in_file[key][region_ref]
                if in_file[key].attrs['scalar']:
                    blob[key] = blob[key][0]
                if len(in_file[key].shape) > 1:
                    blob[key] = blob[key].reshape(-1, in_file[key].shape[1])
            else:
                # If the dataset has multiple attributes, it contains an object
                array = in_file[key][region_ref]
                class_name = in_file[key].attrs['class_name']
                obj_class = getattr(mlreco.data, class_name)
                names = array.dtype.names
                blob[key] = []
                for i, el in enumerate(array):
                    obj_dict = dict(zip(names, el))
                    if self.build_classes:
                        blob[key].append(obj_class(**obj_dict))
                    else:
                        blob[key].append(obj_dict)
                if in_file[key].attrs['scalar']:
                    blob[key] = blob[key][0]
        else:
            # If the reference points at a group, unpack
            el_refs = in_file[key]['index'][region_ref].flatten()
            if len(in_file[key]['index'].shape) == 1:
                ret = np.empty(len(el_refs), dtype=np.object)
                ret[:] = [in_file[key]['elements'][r] for r in el_refs]
                if len(in_file[key]['elements'].shape) > 1:
                    for i in range(len(el_refs)):
                        ret[i] = ret[i].reshape(
                                -1, in_file[key]['elements'].shape[1])
            else:
                ret = [in_file[key][f'element_{i}'][r] for i, r in enumerate(el_refs)]
                for i in range(len(el_refs)):
                    if len(in_file[key][f'element_{i}'].shape) > 1:
                        ret[i] = ret[i].reshape(
                                -1, in_file[key][f'element_{i}'].shape[1])
            blob[key] = ret
