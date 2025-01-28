"""Contains a reader class dedicated to loading data from HDF5 files."""

import h5py
import numpy as np
from dataclasses import fields

import spine.data

from spine.utils.decorators import inherit_docstring

from .base import ReaderBase

__all__ = ['HDF5Reader']


@inherit_docstring(ReaderBase)
class HDF5Reader(ReaderBase):
    """Class which reads information stored in HDF5 files.

    This class inherits from the :class:`ReaderBase` class. It provides
    methods to load HDF5 files and extract their data products. The files
    must be structured as follows:
      - An `events` dataset with all the region references
      - One dataset per data product corresponding to each region reference in
        the `events` dataset
    """
    name = 'hdf5'

    def __init__(self, file_keys, limit_num_files=None, max_print_files=10,
                 n_entry=None, n_skip=None, entry_list=None,
                 skip_entry_list=None, run_event_list=None,
                 skip_run_event_list=None, create_run_map=False,
                 build_classes=True, skip_unknown_attrs=False,
                 run_info_key='run_info', handle_duplicates=False):
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
        run_event_list: list((int, int, int)), optional
            List of (run, subrun, event) triplets to add to the index
        skip_run_event_list: list((int, int, int)), optional
            List of (run, subrun, event) triplets to skip from the index
        create_run_map : bool, default False
            Initialize a map between (run, subrun, event) triplets and entries.
            For large files, this can be quite expensive (must load every entry).
        build_classes : bool, default True
            If the stored object is a class, build it back
        skip_unknown_attrs : bool, default False
            If `True`, allow a loaded object to have unrecognized attributes.
            This allows backward compatibility with old files, but use with
            extreme caution, as this might hide a fundamental issue with your code.
        run_info_key : str, default 'run_info'
            Name of the data product which contains the run info of the event
        """
        # Process the list of files
        self.process_file_paths(file_keys, limit_num_files, max_print_files)

        # If an entry list is requested based on run/subrun/event ID, create map
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

                # If requested, register the (run, subrun, event) information
                if create_run_map:
                    assert run_info_key in in_file, (
                            f"Must provide {run_info_key} to create run map")
                    info = in_file[run_info_key]
                    for r, s, e in zip(info['run'], info['subrun'], info['event']):
                        self.run_info.append((r, s, e))

                # Update the total number of entries
                num_entries = len(in_file['events'])
                self.file_index.append(i*np.ones(num_entries, dtype=np.int64))
                self.file_offsets[i] = self.num_entries
                self.num_entries += num_entries

        # Dump the number of entries to load
        print(f"Total number of entries in the file(s): {self.num_entries}\n")

        # Concatenate the file indexes into one
        self.file_index = np.concatenate(self.file_index)

        # Turn file index and run info into np.ndarray
        if self.run_info is not None:
            self.run_info = np.vstack(self.run_info)

        # Process the run information
        self.handle_duplicates = handle_duplicates
        self.process_run_info(handle_duplicates=self.handle_duplicates)

        # Process the entry list
        self.process_entry_list(
                n_entry, n_skip, entry_list, skip_entry_list,
                run_event_list, skip_run_event_list)

        # Store other attributes
        self.build_classes = build_classes
        self.skip_unknown_attrs = skip_unknown_attrs

    def get(self, idx):
        """Returns a specific entry in the file.

        Parameters
        ----------
        idx : int
            Integer entry ID to access

        Returns
        -------
        data : dict
            Ditionary of data products corresponding to one event
        """
        # Get the appropriate entry index
        assert idx < len(self.entry_index)
        file_idx  = self.get_file_index(idx)
        entry_idx = self.get_file_entry_index(idx)

        # Use the event tree to find out what needs to be loaded
        data = {'file_index': file_idx}
        with h5py.File(self.file_paths[file_idx], 'r') as in_file:
            event = in_file['events'][entry_idx]
            for key in event.dtype.names:
                self.load_key(in_file, event, data, key)

        return data

    def load_key(self, in_file, event, data, key):
        """Fetch a specific key for a specific event.

        Parameters
        ----------
        in_file : h5py.File
            HDF5 file instance
        event : dict
            Dictionary of objects that make up one event
        data : dict
            Dictionary of data products corresponding to one event
        key: str
            Name of the dataset in the entry
        """
        # The event-level information is a region reference: fetch it
        region_ref = event[key]
        if isinstance(in_file[key], h5py.Dataset):
            if not in_file[key].dtype.names:
                # If the reference points at a simple dataset, return
                data[key] = in_file[key][region_ref]
                if in_file[key].attrs['scalar']:
                    data[key] = data[key][0]
                if len(in_file[key].shape) > 1:
                    data[key] = data[key].reshape(-1, in_file[key].shape[1])

            else:
                # If the dataset has multiple attributes, it contains an object.
                # Start by fetching the appropriate class to rebuild
                array = in_file[key][region_ref]
                class_name = in_file[key].attrs['class_name']
                obj_class = getattr(spine.data, class_name)

                # If needed, get the list of recognized attributes
                if self.skip_unknown_attrs:
                    known_attrs = [f.name for f in fields(obj_class)]

                # Load the object
                names = array.dtype.names
                data[key] = []
                for i, el in enumerate(array):
                    # Fetch the list of key/value pairs, filter if requested
                    if self.skip_unknown_attrs:
                        obj_dict = {}
                        for i, k in enumerate(names):
                            if k in known_attrs:
                                obj_dict[k] = el[i]
                    else:
                        obj_dict = dict(zip(names, el))

                    # Rebuild an instance of the object class, if requested
                    if self.build_classes:
                        data[key].append(obj_class(**obj_dict))
                    else:
                        data[key].append(obj_dict)

                if in_file[key].attrs['scalar']:
                    data[key] = data[key][0]

        else:
            # If the reference points at a group, unpack
            el_refs = in_file[key]['index'][region_ref].flatten()
            if len(in_file[key]['index'].shape) == 1:
                ret = np.empty(len(el_refs), dtype=object)
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

            data[key] = ret
