"""Contains a reader class dedicated to loading data from HDF5 files."""

from dataclasses import fields
from typing import Any, Dict, List, Optional, Union
from warnings import warn

import h5py
import numpy as np
import yaml
from yaml.parser import ParserError

import spine.data
from spine.utils.docstring import inherit_docstring
from spine.utils.logger import logger

from .base import ReaderBase

__all__ = ["HDF5Reader"]


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

    name = "hdf5"

    def __init__(
        self,
        file_keys: Optional[Union[str, List[str]]] = None,
        file_list: Optional[str] = None,
        limit_num_files: Optional[int] = None,
        max_print_files: int = 10,
        n_entry: Optional[int] = None,
        n_skip: Optional[int] = None,
        entry_list: Optional[List[int]] = None,
        skip_entry_list: Optional[List[int]] = None,
        run_event_list: Optional[List[List[int]]] = None,
        skip_run_event_list: Optional[List[List[int]]] = None,
        create_run_map: bool = False,
        build_classes: bool = True,
        skip_unknown_attrs: bool = False,
        run_info_key: str = "run_info",
        allow_missing: bool = False,
    ) -> None:
        """Initalize the HDF5 file reader.

        Parameters
        ----------
        file_keys : Union[str, List[str]], optional
            Path or list of paths to the HDF5 files to be read
        file_list : str, optional
            Path to a text file containing a list of file paths to be read
        limit_num_files : Optional[int], optional
            Integer limiting number of files to be taken per data directory
        max_print_files : int, default 10
            Maximum number of loaded file names to be printed
        n_entry : Optional[int], optional
            Maximum number of entries to load
        n_skip : Optional[int], optional
            Number of entries to skip at the beginning
        entry_list : Optional[List[int]]
            List of integer entry IDs to add to the index
        skip_entry_list : Optional[List[int]]
            List of integer entry IDs to skip from the index
        run_event_list: Optional[List[List[int]]], optional
            List of (run, subrun, event) triplets to add to the index
        skip_run_event_list: Optional[List[List[int]]], optional
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
        allow_missing : bool, default False
            If `True`, allows missing entries in the entry or event list
        """
        # Process the list of files
        self.process_file_paths(file_keys, file_list, limit_num_files, max_print_files)

        # If an entry list is requested based on run/subrun/event ID, create map
        if run_event_list is not None or skip_run_event_list is not None:
            create_run_map = True

        # Loop over the input files, build a map from index to file ID
        file_index, run_info = [], []
        self.num_entries = 0
        self.file_offsets = np.empty(len(self.file_paths), dtype=np.int64)
        for i, path in enumerate(self.file_paths):
            with h5py.File(path, "r") as in_file:
                # Check that there are events in the file
                assert "events" in in_file, "File does not contain an event tree"

                events = in_file["events"]
                assert isinstance(
                    events, h5py.Dataset
                ), "'events' is not a dataset in the HDF5 file."

                # If requested, register the (run, subrun, event) information
                if create_run_map:
                    assert (
                        run_info_key in in_file
                    ), f"Must provide {run_info_key} to create run map"

                    info = in_file[run_info_key]
                    assert isinstance(
                        info, h5py.Dataset
                    ), f"{run_info_key} is not a dataset in the HDF5 file."
                    assert all(
                        k in info.dtype.names for k in ["run", "subrun", "event"]
                    ), f"{run_info_key} dataset missing required fields."

                    for r, s, e in zip(info["run"], info["subrun"], info["event"]):
                        run_info.append((r, s, e))

                # Update the total number of entries
                num_entries = len(events)
                file_index.append(i * np.ones(num_entries, dtype=np.int64))
                self.file_offsets[i] = self.num_entries
                self.num_entries += num_entries

        # Dump the number of entries to load
        logger.info("Total number of entries in the file(s): %d\n", self.num_entries)

        # Concatenate the file indexes into one, set run info if needed
        self.file_index = np.concatenate(file_index)
        self.run_info = run_info if create_run_map else None

        # Process the run information
        self.process_run_info()

        # Process the entry list
        self.process_entry_list(
            n_entry,
            n_skip,
            entry_list,
            skip_entry_list,
            run_event_list,
            skip_run_event_list,
            allow_missing,
        )

        # Store other attributes
        self.build_classes = build_classes
        self.skip_unknown_attrs = skip_unknown_attrs

        # Process the configuration used to produce the HDF5 file
        self.cfg = self.process_cfg()

        # Process the SPINE version used to produced the HDF5 file
        self.version = self.process_version()

    def process_cfg(self) -> Optional[dict]:
        """Fetches the SPINE configuration used to produce the HDF5 file.

        Returns
        -------
        dict
            Configuration dictionary
        """
        # Fetch the string-form configuration
        with h5py.File(self.file_paths[0], "r") as in_file:
            assert "info" in in_file, "HDF5 file missing 'info' group."
            assert (
                "cfg" in in_file["info"].attrs
            ), "HDF5 file 'info' group missing 'cfg' attribute."
            cfg_str = in_file["info"].attrs["cfg"]

        # Attempt to parse it (need try for now for SPINE versions < v0.4.0)
        try:
            assert isinstance(cfg_str, str), "'cfg' attribute is not a string."
            cfg = yaml.safe_load(cfg_str)
        except ParserError:
            warn(
                "Parsing configuration failed, returning None for SPINE versions < v0.4.0"
            )
            return None

        return cfg

    def process_version(self) -> str:
        """Returns the SPINE release version used to produce the HDF5 file.

        Returns
        -------
        str
            SPINE release tag
        """
        # Fetch the string-form configuration
        with h5py.File(self.file_paths[0], "r") as in_file:
            assert "info" in in_file, "HDF5 file missing 'info' group."
            assert (
                "version" in in_file["info"].attrs
            ), "HDF5 file 'info' group missing 'version' attribute."
            version = in_file["info"].attrs["version"]

        assert isinstance(version, str), "'version' attribute is not a string."
        return version

    def get(self, idx: int) -> Dict[str, Any]:
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
        file_idx = self.get_file_index(idx)
        entry_idx = self.get_file_entry_index(idx)

        # Use the event tree to find out what needs to be loaded
        data = {"file_index": file_idx, "file_entry_index": entry_idx}
        with h5py.File(self.file_paths[file_idx], "r") as in_file:
            events = in_file["events"]
            assert isinstance(
                events, h5py.Dataset
            ), "'events' is not a dataset in the HDF5 file."

            event = events[entry_idx]
            names = getattr(getattr(event, "dtype", None), "names", None)
            if names is not None:
                for key in names:
                    self.load_key(in_file, event, data, key)
            else:
                raise ValueError("Event entry does not have named fields.")

        # Use the global index, not the one read from file
        data["index"] = idx

        return data

    def load_key(
        self, in_file: h5py.File, event: Dict[str, Any], data: Dict[str, Any], key: str
    ) -> None:
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
        dataset = in_file[key]
        if isinstance(dataset, h5py.Dataset):
            names = getattr(getattr(dataset, "dtype", None), "names", None)
            if not names:
                # If the reference points at a simple dataset, return
                data[key] = dataset[region_ref]
                if dataset.attrs["scalar"]:
                    data[key] = data[key][0]
                if len(dataset.shape) > 1:
                    data[key] = data[key].reshape(-1, dataset.shape[1])

            else:
                # If the dataset has multiple attributes, it contains an object.
                # Start by fetching the appropriate class to rebuild
                array = dataset[region_ref]
                class_name = dataset.attrs["class_name"]
                assert isinstance(
                    class_name, str
                ), "Dataset missing 'class_name' attribute."
                obj_class = getattr(spine.data, class_name)

                # If needed, get the list of recognized attributes
                known_attrs = []
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

                if in_file[key].attrs["scalar"]:
                    data[key] = data[key][0]

        elif isinstance(dataset, h5py.Group):
            # If the reference points at a group, unpack
            index = dataset["index"]
            assert isinstance(index, h5py.Dataset), "Dataset 'index' is missing."
            el_refs = index[region_ref].flatten()
            if len(index.shape) == 1:
                elements = dataset["elements"]
                assert isinstance(
                    elements, h5py.Dataset
                ), "Dataset 'elements' is missing."
                ret = np.empty(len(el_refs), dtype=object)
                ret[:] = [elements[r] for r in el_refs]
                if len(elements.shape) > 1:
                    for i in range(len(el_refs)):
                        ret[i] = ret[i].reshape(-1, elements.shape[1])

            else:
                elements = [dataset[f"element_{i}"] for i in range(len(el_refs))]
                ret = []
                for i, el in enumerate(elements):
                    assert isinstance(
                        el, h5py.Dataset
                    ), f"Dataset 'element_{i}' is missing."
                    ret.append(el[el_refs[i]])
                    if len(el.shape) > 1:
                        ret[i] = ret[i].reshape(-1, el.shape[1])

            data[key] = ret

        else:
            raise ValueError(f"Dataset for key '{key}' is neither a group nor dataset.")
