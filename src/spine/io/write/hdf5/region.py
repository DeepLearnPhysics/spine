"""Region-reference storage backend for the SPINE HDF5 writer."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from .common import DataFormat, require_dataset, require_group


class RegionReferenceBackend:
    """Write event products addressed through HDF5 region references."""

    keys: set[str] | None
    type_dict: dict[str, DataFormat] | None
    object_dtypes: list[Any]
    event_dtype: Any
    lite: bool

    def initialize_region_datasets(
        self, out_file: h5py.Group, type_dict: dict[str, DataFormat]
    ) -> None:
        """Create datasets used by the region-reference layout.

        Parameters
        ----------
        out_file : h5py.Group
            File or stage group that will own the datasets.
        type_dict : dict[str, DataFormat]
            Physical format inferred for each stored key.
        """
        # Initialize the datasets, store the general type of the event
        self.event_dtype = []
        ref_dtype = h5py.special_dtype(ref=h5py.RegionReference)
        for key, val in type_dict.items():
            # Add a dataset reference for this key to the event dtype
            self.event_dtype.append((key, ref_dtype))
            if not isinstance(val.width, list):
                # If the key contains a list of objects of identical shape
                shape = (0, val.width) if val.width else (0,)
                maxshape = (None, val.width) if val.width else (None,)
                out_file.create_dataset(key, shape, maxshape=maxshape, dtype=val.dtype)

                # Store the class name to rebuild it later, if relevant
                if val.class_name is not None:
                    out_file[key].attrs["class_name"] = val.class_name

            elif not val.merge:
                # If the elements of the list are of variable widths, refer to
                # one dataset per element. An index is stored alongside the
                # dataset to break it into individual elements.
                group = out_file.create_group(key)

                n_arrays = len(val.width)
                shape, maxshape = (0, n_arrays), (None, n_arrays)
                group.create_dataset("index", shape, maxshape=maxshape, dtype=ref_dtype)

                for i, w in enumerate(val.width):
                    shape = (0, w) if w else (0,)
                    maxshape = (None, w) if w else (None,)
                    el = f"element_{i}"
                    group.create_dataset(el, shape, maxshape=maxshape, dtype=val.dtype)

            else:
                # If the  elements of the list are of equal width, store them
                # all to one dataset. An index is stored alongside the dataset
                # to break it into individual elements downstream.
                group = out_file.create_group(key)

                shape = (0, val.width[0]) if val.width[0] else (0,)
                maxshape = (None, val.width[0]) if val.width[0] else (None,)
                group.create_dataset("index", (0,), maxshape=(None,), dtype=ref_dtype)
                group.create_dataset(
                    "elements", shape, maxshape=maxshape, dtype=val.dtype
                )

            # Give relevant attributes to the dataset
            out_file[key].attrs["scalar"] = val.scalar

        out_file.create_dataset(
            "events", (0,), maxshape=(None,), dtype=self.event_dtype
        )

    def append_region_entry(
        self, out_file: h5py.Group, data: dict[str, Any], batch_id: int
    ) -> None:
        """Append one event and its region references.

        Parameters
        ----------
        out_file : h5py.Group
            File or stage group containing initialized datasets.
        data : dict
            Batched data products.
        batch_id : int
            Batch ID to be stored
        """
        # Initialize a new event
        event = np.empty(1, self.event_dtype)

        # Initialize a dictionary of references to be passed to the
        # event dataset and store the input and result keys
        if self.keys is None:
            raise RuntimeError("Keys to be stored have not been identified.")
        for key in self.keys:
            self.append_region_key(out_file, event, data, key, batch_id)

        # Append event
        event_ds = require_dataset(out_file, "events")

        event_id = len(event_ds)
        event_ds.resize(event_id + 1, axis=0)  # pylint: disable=E1101
        event_ds[event_id] = event

    def append_region_key(
        self,
        out_file: h5py.Group,
        event: np.ndarray,
        data: dict[str, Any],
        key: str,
        batch_id: int,
    ) -> None:
        """Append one product and attach its reference to an event row.

        Parameters
        ----------
        out_file : h5py.Group
            File or stage group containing initialized datasets.
        event : np.ndarray
            Array representing the event to which the data corresponds
        data : dict
            Dictionary of data products
        key : str
            Product key.
        batch_id : int
            Batch ID to be stored
        """
        # Sanity check that the data type information for this key has been initialized
        if self.type_dict is None:
            raise RuntimeError(
                f"Cannot append key {key}: data formats are not initialized."
            )

        # Get the data type and store it
        val = self.type_dict[key]
        if not val.merge and not isinstance(val.width, list):
            # Store single arrays
            if np.isscalar(data[key]):
                # If a data product is a single scalar, use it for every entry
                array = np.asarray([data[key]])

            else:
                # Otherwise, get the data corresponding to the current entry
                array = data[key][batch_id]
                if val.scalar:
                    array = np.asarray([array])

            if val.dtype in self.object_dtypes:
                if isinstance(val.dtype, type):
                    raise TypeError(
                        f"Object dtype for `{key}` must be compound, got "
                        f"{type(val.dtype)}."
                    )
                self.store_objects(out_file, event, key, array, val.dtype, self.lite)
            else:
                self.store(out_file, event, key, array)

        elif not val.merge:
            # Store the array and its reference for each element in the list
            array_list = data[key][batch_id]
            self.store_jagged(out_file, event, key, array_list)

        else:
            # Store one array of for all in the list and a index to break them
            array_list = data[key][batch_id]
            self.store_flat(out_file, event, key, array_list)

    @staticmethod
    def store(
        out_file: h5py.Group, event: np.ndarray, key: str, array: np.ndarray
    ) -> None:
        """Append an array and store its region reference on the event.

        Parameters
        ----------
        out_file : h5py.File
            HDF5 file instance
        event : np.ndarray
            Array representing the event to which the data corresponds
        key: str
            Name of the dataset in the file
        array : np.ndarray
            Array to be stored
        """
        # Extend the dataset, store array
        dataset = require_dataset(out_file, key)

        current_id = len(dataset)
        dataset.resize(current_id + len(array), axis=0)
        dataset[current_id : current_id + len(array)] = array

        # Define region reference, store it at the event level
        region_ref = dataset.regionref[current_id : current_id + len(array)]
        event[key] = region_ref

    @staticmethod
    def store_jagged(
        out_file: h5py.Group,
        event: np.ndarray,
        key: str,
        array_list: list[np.ndarray],
    ) -> None:
        """Append differently shaped arrays and reference each element.

        Parameters
        ----------
        out_file : h5py.File
            HDF5 file instance
        event : np.ndarray
            Array representing the event to which the data corresponds
        key: str
            Name of the dataset in the file
        array_list : list(np.ndarray)
            List of arrays to be stored
        """
        # Fetch the group corresponding to this key, which contains one dataset per
        # element in the list, and check that it is indeed a group
        group = require_group(out_file, key)

        # Extend the dataset, store combined array
        region_refs = []
        for i, array in enumerate(array_list):

            dataset = require_dataset(group, f"element_{i}")

            current_id = len(dataset)
            dataset.resize(current_id + len(array), axis=0)
            dataset[current_id : current_id + len(array)] = array

            region_ref = dataset.regionref[current_id : current_id + len(array)]
            region_refs.append(region_ref)

        # Define the index which stores a list of region_refs
        index = require_dataset(group, "index")

        current_id = len(index)
        index.resize(current_id + 1, axis=0)
        index[current_id] = region_refs

        # Define a region reference to all the references,
        # store it at the event level
        region_ref = index.regionref[current_id : current_id + 1]
        event[key] = region_ref

    @staticmethod
    def store_flat(
        out_file: h5py.Group,
        event: np.ndarray,
        key: str,
        array_list: list[np.ndarray],
    ) -> None:
        """Append same-width arrays with an index of element references.

        Parameters
        ----------
        out_file : h5py.File
            HDF5 file instance
        event : np.ndarray
            Array representing the event to which the data corresponds
        key: str
            Name of the dataset in the file
        array_list : list(np.ndarray)
            List of arrays to be stored
        """
        # Fetch the group corresponding to this key, which contains one dataset for
        # the elements in the list and one for the index, and check that it is indeed
        # a group
        group = require_group(out_file, key)

        # Extend the dataset, store combined array
        dataset = require_dataset(group, "elements")

        first_id = len(dataset)
        array = np.concatenate(array_list) if len(array_list) else []
        dataset.resize(first_id + len(array), axis=0)
        dataset[first_id : first_id + len(array)] = array

        # Loop over arrays in the list, create a reference for each
        index = require_dataset(group, "index")

        current_id = len(index)
        index.resize(current_id + len(array_list), axis=0)
        last_id = first_id
        for i, el in enumerate(array_list):
            first_id = last_id
            last_id += len(el)
            el_ref = dataset.regionref[first_id:last_id]
            index[current_id + i] = el_ref

        # Define a region reference to all the references,
        # store it at the event level
        region_ref = index.regionref[current_id : current_id + len(array_list)]
        event[key] = region_ref

    @staticmethod
    def store_objects(
        out_file: h5py.Group,
        event: np.ndarray,
        key: str,
        array: np.ndarray,
        obj_dtype: list[tuple[str, type]],
        lite: bool,
    ) -> None:
        """Append compound object rows and store their event reference.

        Parameters
        ----------
        out_file : h5py.File
            HDF5 file instance
        event : np.ndarray
            Array representing the event to which the data corresponds
        key: str
            Name of the dataset in the file
        array : np.ndarray
            Array of objects or dictionaries to be stored
        obj_dtype : list
            List of (key, dtype) pairs which specify what's to store
        lite : bool
            If `True`, store the lite version of objects
        """
        # Convert list of objects to list of storable objects
        objects = np.empty(len(array), obj_dtype)
        for i, obj in enumerate(array):
            objects[i] = tuple(obj.as_dict(lite).values())

        # Extend the dataset, store array
        dataset = require_dataset(out_file, key)

        current_id = len(dataset)
        dataset.resize(current_id + len(array), axis=0)
        dataset[current_id : current_id + len(array)] = objects

        # Define region reference, store it at the event level
        region_ref = dataset.regionref[current_id : current_id + len(array)]
        event[key] = region_ref
