"""Module to write the output of the reconstruction to file."""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import h5py
import numpy as np
import yaml

import spine.data
from spine.version import __version__

__all__ = ["HDF5Writer"]


class HDF5Writer:
    """Writes data to an HDF5 file.

    Builds an HDF5 file to store the input and/or the output of the
    reconstruction chain. It can also be used to append an existing HDF5 file
    with information coming out of the analysis tools.

    Typical configuration should look like:

    .. code-block:: yaml

        io:
          ...
          writer:
            name: hdf5
            file_name: output.h5
            keys:
              - input_data
              - segmentation
              - ...
    """

    name = "hdf5"

    def __init__(
        self,
        file_name: Optional[str] = None,
        prefix: Optional[Union[str, List[str]]] = None,
        suffix: str = "spine",
        keys: Optional[List[str]] = None,
        skip_keys: Optional[List[str]] = None,
        dummy_ds: Optional[Dict[str, str]] = None,
        overwrite: bool = False,
        append: bool = False,
        split: bool = False,
        lite: bool = False,
    ) -> None:
        """Initializes the basics of the output file.

        Parameters
        ----------
        file_name : str, optional
            Name of the output HDF5 file
        prefix : str or List[str], optional
            Input file prefix. It will be use to form the output file name,
            provided that no file_name is explicitly provided. Must be a list
            with one prefix per input file when `split` is `True`.
        suffix : str, default "spine"
            Suffix to add to the output file name if it is built from the input
        keys : List[str], optional
            List of data product keys to store. If not specified, store everything
        skip_keys: List[str], optionl
            List of data product keys to skip
        dummy_ds: Dict[str, str], optional
            Keys for which to create placeholder datasets. For each key, specify
            the object type it is supposed to represent as a string.
        overwrite : bool, default False
            If `True`, overwrite the output file if it already exists
        append : bool, default False
            If `True`, add new values to the end of an existing file
        split : bool, default False
            If `True`, split the output to produce one file per input file
        lite : bool, default False
            If `True`, the lite version of objects is stored (drop point indexes)
        """
        # Build the output file name(s) from the input prefix(es) if not provided
        self.file_names = self.get_file_names(file_name, prefix, suffix, split)

        # Check that the output file(s) do(es) not already exist, if requested
        if not overwrite and not append:
            for file_name in self.file_names:
                if os.path.isfile(file_name):
                    raise FileExistsError(f"File with name {file_name} already exists.")

        # Store other persistent attributes
        self.append = append
        self.split = split
        self.lite = lite

        self.keys = set(keys) if keys is not None else None
        self.skip_keys = skip_keys

        # Initialize dummy dataset placeholders once
        self.dummy_ds = dummy_ds
        if self.dummy_ds is not None:
            for key, class_name in self.dummy_ds.items():
                self.dummy_ds[key] = getattr(spine.data, class_name)()

        # Initialize attributes to be stored when the output file is created
        self.ready = False
        self.object_dtypes = []
        self.type_dict = None
        self.event_dtype = None

    @staticmethod
    def get_file_names(
        file_name: Optional[str] = None,
        prefix: Optional[Union[str, List[str]]] = None,
        suffix: str = "spine",
        split: bool = False,
    ) -> List[str]:
        """Build output file name(s) from an explicit name or input prefix(es).

        Logic is as follows:
        - If `split` is `False` and `file_name` is provided, use `file_name`
        - If `split` is `False` and `file_name` is not provided, build the file name
          from the input `prefix` by adding a suffix
        - If `split` is `True` and `file_name` is not provided, build the file names
          from the input `prefix` by adding a suffix
        - If `split` is `True` and `file_name` is provided, build the file names from
          `file_name` by adding an index, unless there is only one input prefix,
          in which case use `file_name` as is

        Parameters
        ----------
        file_name : str, optional
            Name of the output HDF5 file. If not provided, it will be built from the
            input prefix(es).
        prefix : str or List[str], optional
            Input file prefix(es).
        suffix : str, default "spine"
            Suffix to add to the output file name if it is built from the input
        split : bool, default False
            If `True`, split the output to produce one file per input file.

        Returns
        -------
        List[str]
            List of output file names.
        """
        # If the output is not split, use the provided file name or build it from the prefix
        if not split:
            if file_name:
                return [file_name]

            assert prefix is not None and isinstance(prefix, str), (
                "If the output `file_name` is not provided, must provide "
                "the input file `prefix` to build it from."
            )
            return [f"{prefix}_{suffix}.h5"]

        # If the output is split, build the file names from the provided one by
        # adding an index, unless there is only one prefix per file,
        # in which case use the provided name as is
        assert prefix is not None and not isinstance(prefix, str), (
            "If `split` is enabled, must provide one `prefix` per input file "
            "to determine the number of output files."
        )

        if file_name and len(prefix) == 1:
            return [file_name]

        if not file_name:
            return [f"{pre}_{suffix}.h5" for pre in prefix]

        # Otherwise, build the file names from the provided one by adding an index
        dir_name = os.path.dirname(file_name)
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        return [
            os.path.join(dir_name, f"{base_name}_{i}.h5") for i in range(len(prefix))
        ]

    @dataclass
    class DataFormat:
        """Data structure to hold writing parameters.

        Attributes
        ----------
        dtype : Union[type, List[Tuple[str, type]]], optional
            Data type
        class_name : str, optional
            Name of the class the information comes from
        width : Union[int, List[int]], default 0
            Width of the tensor to store, if it is a tensor
        merge : bool, default False
            Whether to merge lists of arrays into a single dataset
        scalar : bool, default False
            Whether the data is a scalar object or not
        """

        dtype: Optional[Union[type, List[Tuple[str, type]]]] = None
        class_name: Optional[str] = None
        width: Union[int, List[int]] = 0
        merge: bool = False
        scalar: bool = False

    def create(
        self,
        data: Dict[str, Any],
        cfg: Optional[Dict[str, Any]] = None,
        append: bool = False,
    ) -> None:
        """Initialize the output file structure based on the data dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary of data products
        cfg : Dict[str, Any]
            Dictionary containing the complete SPINE configuration
        append : bool, default False
            If `True`, load existing files if present and create missing files
        """
        # Fetch the required keys to be stored and register them
        self.keys = self.get_stored_keys(data)

        # Fetch the data type information for each key and store it in a dictionary
        self.type_dict, self.object_dtypes = self.get_data_types(data, self.keys)

        # Initialize the output HDF5 file(s)
        for file_name in self.file_names:
            if append and os.path.isfile(file_name):
                with h5py.File(file_name, "r") as out_file:
                    event_obj = out_file["events"]
                    assert isinstance(event_obj, h5py.Dataset), (
                        "Expected dataset for events to be a Dataset, but got "
                        f"{type(event_obj)} instead."
                    )
                    self.event_dtype = getattr(event_obj, "dtype")
                continue

            with h5py.File(file_name, "w") as out_file:
                # Initialize the info group that stores environment parameters
                out_file.create_group("info")
                out_file["info"].attrs["version"] = __version__
                if cfg is not None:
                    out_file["info"].attrs["cfg"] = yaml.dump(cfg)

                # Initialize the event dataset and their reference array datasets
                self.initialize_datasets(out_file, self.type_dict)

        # Mark file(s) as ready for use
        self.ready = True

    def get_stored_keys(self, data: Dict[str, Any]) -> Set[str]:
        """Get the list of data product keys to store.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary of data products

        Returns
        -------
        keys : Set[str]
            List of data keys to store to file
        """
        # If the keys were already produced, nothing to do
        if self.ready and self.keys is not None:
            return self.keys

        # Check that the required/ keys make sense,
        assert (self.keys is None) | (
            self.skip_keys is None
        ), "Must not specify both `keys` or `skip_keys`."

        # Translate keys/skip_keys into a single set
        keys = {"index"}
        if self.keys is None:
            keys.update(data.keys())
            if self.skip_keys is not None:
                for key in self.skip_keys:
                    if key not in keys:
                        raise KeyError(
                            f"Key {key} appears in `skip_keys` but does not "
                            "appear in the dictionary of data products."
                        )
                    keys.remove(key)

        else:
            keys.update(self.keys)
            for key in self.keys:
                assert key in data, (
                    f"Cannot store {key} as it does not appear "
                    "in the dictionary of data products."
                )

        # Add dummy keys to the list, if requested
        if self.dummy_ds is not None:
            for key in self.dummy_ds:
                assert key not in keys, (
                    f"The requested dummy dataset {key} already exists "
                    "in the list of real datasets being stored."
                )
            keys.update(self.dummy_ds.keys())

        return keys

    def get_data_types(
        self, data: Dict[str, Any], keys: Set[str]
    ) -> Tuple[Dict[str, DataFormat], List[List[Tuple[str, type]]]]:
        """Get the data type information for each key.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary of data products

        Returns
        -------
        type_dict : Dict[str, DataFormat]
            Dictionary containing the data type information for each key
        object_dtypes : List[List[Tuple[str, type]]]
            List of composite object dtypes found in the data
        """
        # Loop over the keys and get the data type information for each of them, store it
        type_dict = {}
        object_dtypes = []
        for key in keys:
            type_dict[key] = self.get_data_type(data, key)
            if (
                type_dict[key].class_name is not None
                and type_dict[key].dtype not in object_dtypes
            ):
                object_dtypes.append(type_dict[key].dtype)

        return type_dict, object_dtypes

    def get_data_type(self, data: Dict[str, Any], key: str) -> DataFormat:
        """Identify the dtype and shape objects to be dealt with.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing the information to be stored
        key : str
            Dictionary key name

        Returns
        -------
        DataFormat
            DataFormat object containing the data type information for the key
        """
        # Initialize a type object for this output key
        data_format = self.DataFormat()

        # Store the necessary information to know how to store a key
        if np.isscalar(data[key]):
            # Single scalar for the entire batch (e.g. accuracy, loss, etc.)
            if isinstance(data[key], str):
                data_format.dtype = h5py.string_dtype()
            else:
                data_format.dtype = type(data[key])
            data_format.scalar = True

        else:
            if np.isscalar(data[key][0]):
                # List containing a single scalar per batch ID
                if isinstance(data[key][0], str):
                    data_format.dtype = h5py.string_dtype()
                else:
                    data_format.dtype = type(data[key][0])
                data_format.scalar = True

            elif not hasattr(data[key][0], "__len__"):
                # List containing one single non-standard object per batch ID
                object_dtype = self.get_object_dtype(data[key][0])
                data_format.dtype = object_dtype
                data_format.scalar = True
                data_format.class_name = data[key][0].__class__.__name__

            else:
                # List containing a list/array of objects per batch ID
                ref_obj = data[key][0]
                if isinstance(data[key][0], list):
                    # If simple list, check if it is empty
                    if len(data[key][0]):
                        # If it contains simple objects, use the first
                        if not hasattr(data[key][0][0], "__len__"):
                            ref_obj = data[key][0][0]
                    else:
                        # If it is empty, must contain a default value
                        assert hasattr(data[key][0], "default"), (
                            f"Failed to find type of {key}. Lists that can "
                            "be empty should be initialized as an "
                            "ObjectList with a default object type."
                        )
                        ref_obj = data[key][0].default

                        # If the default value is an array, unwrap as such
                        if isinstance(ref_obj, np.ndarray):
                            data_format.width = [0]
                            data_format.merge = True

                if not hasattr(ref_obj, "__len__"):
                    # List containing a single list of objects per batch ID
                    object_dtype = self.get_object_dtype(ref_obj)
                    data_format.dtype = object_dtype
                    data_format.class_name = ref_obj.__class__.__name__

                elif not isinstance(ref_obj, list) and not ref_obj.dtype == object:
                    # List containing a single ndarray of scalars per batch ID
                    data_format.dtype = ref_obj.dtype
                    if len(ref_obj.shape) == 2:
                        data_format.width = ref_obj.shape[1]

                elif isinstance(ref_obj, (list, np.ndarray)):
                    # List containing a list/array of ndarrays per batch ID
                    widths = []
                    same_width = True
                    for el in ref_obj:
                        width = 0
                        if len(el.shape) == 2:
                            width = el.shape[1]
                        widths.append(width)
                        same_width &= width == widths[0]

                    data_format.dtype = ref_obj[0].dtype
                    data_format.width = widths
                    data_format.merge = same_width

                else:
                    dtype = type(data[key][0])
                    raise TypeError(
                        f"Cannot store output of type {dtype} in key {key}."
                    )

        return data_format

    def get_object_dtype(self, obj: Any) -> List[Tuple[str, type]]:
        """Loop over the attributes of a class to figure out what to store.

        This function assumes that the class only posseses getters that return
        either a scalar, string or np.ndarrary.

        Parameters
        ----------
        object : class
            Instance of an class used to identify attribute types

        Returns
        -------
        List[Tuple[str, type]]
            List of (key, dtype) pairs
        """
        object_dtype = []
        for key, val in obj.as_dict(self.lite).items():
            # Append the relevant data type
            if isinstance(val, str):
                # String
                object_dtype.append((key, h5py.string_dtype()))

            elif hasattr(obj, "enum_attrs") and key in obj.enum_attrs:
                # Recognized enumerated list
                enum_dtype = h5py.enum_dtype(
                    dict(obj.enum_attrs[key]), basetype=np.int64
                )
                object_dtype.append((key, enum_dtype))

            elif np.isscalar(val):
                # Non-string, non-enumerated scalar
                dtype = type(val)
                object_dtype.append((key, dtype))

            elif hasattr(obj, "_fixed_length_attrs") and key in obj._fixed_length_attrs:
                # Fixed-length array of scalars
                object_dtype.append((key, val.dtype, len(val)))

            elif isinstance(val, np.ndarray):
                # Variable-length array of scalars
                object_dtype.append((key, h5py.vlen_dtype(val.dtype)))

            else:
                raise ValueError(
                    f"Attribute {key} of {obj} has unrecognized an "
                    f"unrecognized type: {type(val)}"
                )

        return object_dtype

    def initialize_datasets(
        self, out_file: h5py.File, type_dict: Dict[str, DataFormat]
    ) -> None:
        """Create place hodlers for all the datasets to be filled.

        Parameters
        ----------
        out_file : h5py.File
            HDF5 file instance
        type_dict : Dict[str, DataFormat]
            Dictionary containing the data type information for each key
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

    def __call__(
        self, data: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None
    ) -> None:
        """Append the HDF5 file with the content of a batch.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        cfg : dict
            Dictionary containing the complete SPINE configuration
        """
        # Nest data if is not already, fetch batch size
        if np.isscalar(data["index"]):
            for k in data:
                data[k] = [data[k]]
            batch_size = 1
        else:
            batch_size = len(data["index"])

        # If needed, add empty data for dummy datasets
        if self.dummy_ds is not None:
            for key, value in self.dummy_ds.items():
                data[key] = [spine.data.ObjectList([], default=value)] * batch_size

        # If this function has never been called, initialiaze the HDF5 file(s)
        if not self.ready:
            self.create(data, cfg, append=self.append)

        # Append file(s)
        if not self.split or len(self.file_names) == 1:
            with h5py.File(self.file_names[0], "a") as out_file:
                # Loop over batch IDs
                for batch_id in range(batch_size):
                    self.append_entry(out_file, data, batch_id)

        else:
            file_ids = data["file_index"]
            for file_id in np.unique(file_ids):
                with h5py.File(self.file_names[file_id], "a") as out_file:
                    for batch_id in np.where(file_ids == file_id)[0]:
                        self.append_entry(out_file, data, batch_id)

    def append_entry(
        self, out_file: h5py.File, data: Dict[str, Any], batch_id: int
    ) -> None:
        """Stores one entry.

        Parameters
        ----------
        out_file : h5py.File
            HDF5 file instance
        data : Dict[str, Any]
            Dictionary of data products
        batch_id : int
            Batch ID to be stored
        """
        # Initialize a new event
        event = np.empty(1, self.event_dtype)

        # Initialize a dictionary of references to be passed to the
        # event dataset and store the input and result keys
        assert self.keys is not None, "Keys to be stored have not been identified."
        for key in self.keys:
            self.append_key(out_file, event, data, key, batch_id)

        # Append event
        event_ds = out_file["events"]
        assert isinstance(
            event_ds, h5py.Dataset
        ), f"Expected dataset for events to be a Dataset, but got {type(event_ds)} instead."

        event_id = len(event_ds)
        event_ds.resize(event_id + 1, axis=0)  # pylint: disable=E1101
        event_ds[event_id] = event

    def append_key(
        self,
        out_file: h5py.File,
        event: np.ndarray,
        data: Dict[str, Any],
        key: str,
        batch_id: int,
    ) -> None:
        """Stores data key in a specific dataset of an HDF5 file.

        Parameters
        ----------
        out_file : h5py.File
            HDF5 file instance
        event : np.ndarray
            Array representing the event to which the data corresponds
        data : dict
            Dictionary of data products
        key : string
            Dictionary key name
        batch_id : int
            Batch ID to be stored
        """
        # Sanity check that the data type information for this key has been initialized
        assert self.type_dict is not None and self.object_dtypes is not None, (
            f"Cannot append key {key} to file as the data type information "
            "has not been initialized."
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
                assert not isinstance(val.dtype, type), (
                    f"Expected object dtype for key {key} to be a composite type, but "
                    f"got {type(val.dtype)} instead."
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
        out_file: h5py.File, event: np.ndarray, key: str, array: np.ndarray
    ) -> None:
        """Stores an `ndarray` in the file and stores its mapping in the event
        dataset.

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
        dataset = out_file[key]
        assert isinstance(dataset, h5py.Dataset), (
            f"Expected dataset for key {key} to be a Dataset, but got "
            f"{type(dataset)} instead."
        )

        current_id = len(dataset)
        dataset.resize(current_id + len(array), axis=0)
        dataset[current_id : current_id + len(array)] = array

        # Define region reference, store it at the event level
        region_ref = dataset.regionref[current_id : current_id + len(array)]
        event[key] = region_ref

    @staticmethod
    def store_jagged(
        out_file: h5py.File,
        event: np.ndarray,
        key: str,
        array_list: List[np.ndarray],
    ) -> None:
        """Stores a jagged list of arrays in the file and stores an index
        mapping for each array element in the event dataset.

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
        group = out_file[key]
        assert isinstance(group, h5py.Group), (
            f"Expected group for key {key} to be a Group, but got "
            f"{type(group)} instead."
        )

        # Extend the dataset, store combined array
        region_refs = []
        for i, array in enumerate(array_list):

            dataset = group[f"element_{i}"]
            assert isinstance(dataset, h5py.Dataset), (
                f"Expected dataset for element {i} of key {key} to be a Dataset, "
                f"but got {type(dataset)} instead."
            )

            current_id = len(dataset)
            dataset.resize(current_id + len(array), axis=0)
            dataset[current_id : current_id + len(array)] = array

            region_ref = dataset.regionref[current_id : current_id + len(array)]
            region_refs.append(region_ref)

        # Define the index which stores a list of region_refs
        index = group["index"]
        assert isinstance(index, h5py.Dataset), (
            f"Expected dataset for index of key {key} to be a Dataset, but got "
            f"{type(index)} instead."
        )

        current_id = len(index)
        index.resize(current_id + 1, axis=0)
        index[current_id] = region_refs

        # Define a region reference to all the references,
        # store it at the event level
        region_ref = index.regionref[current_id : current_id + 1]
        event[key] = region_ref

    @staticmethod
    def store_flat(
        out_file: h5py.File, event: np.ndarray, key: str, array_list: List[np.ndarray]
    ) -> None:
        """Stores a concatenated list of arrays in the file and stores its
        index mapping in the event dataset to break them.

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
        group = out_file[key]
        assert isinstance(group, h5py.Group), (
            f"Expected group for key {key} to be a Group, but got "
            f"{type(group)} instead."
        )

        # Extend the dataset, store combined array
        dataset = group["elements"]
        assert isinstance(dataset, h5py.Dataset), (
            f"Expected dataset for elements of key {key} to be a Dataset, but got "
            f"{type(dataset)} instead."
        )

        first_id = len(dataset)
        array = np.concatenate(array_list) if len(array_list) else []
        dataset.resize(first_id + len(array), axis=0)
        dataset[first_id : first_id + len(array)] = array

        # Loop over arrays in the list, create a reference for each
        index = group["index"]
        assert isinstance(index, h5py.Dataset), (
            f"Expected dataset for index of key {key} to be a Dataset, but got "
            f"{type(index)} instead."
        )

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
        out_file: h5py.File,
        event: np.ndarray,
        key: str,
        array: np.ndarray,
        obj_dtype: List[Tuple[str, type]],
        lite: bool,
    ) -> None:
        """Stores a list of objects with understandable attributes in the file
        and stores its mapping in the event dataset.

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
        dataset = out_file[key]
        assert isinstance(dataset, h5py.Dataset), (
            f"Expected dataset for key {key} to be a Dataset, but got "
            f"{type(dataset)} instead."
        )

        current_id = len(dataset)
        dataset.resize(current_id + len(array), axis=0)
        dataset[current_id : current_id + len(array)] = objects

        # Define region reference, store it at the event level
        region_ref = dataset.regionref[current_id : current_id + len(array)]
        event[key] = region_ref
