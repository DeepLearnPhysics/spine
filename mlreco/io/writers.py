"""Contains classes used to write data to files.

Supports writing the input/output of the reconstruction chain chain to
HDF5 files or to log some inforation to CSV files.
"""

import os
from dataclasses import dataclass, asdict

import yaml
import h5py
import numpy as np

from mlreco.version import __version__

__all__ = ['HDF5Writer', 'CSVWriter']


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
            input_keys:
              - input_data
              - ...
            result_keys:
              - segmentation
              - ...
    """
    name = 'hdf5'

    def __init__(self, file_name='output.h5', input_keys=None,
                 skip_input_keys=None, result_keys=None, skip_result_keys=None,
                 append_file=False):
        """Initializes the basics of the output file.

        Parameters
        ----------
        file_name : str, default 'output.h5'
            Name of the output HDF5 file
        input_keys : List[str], optional
            List of input keys to store. If not specified, stores all of the
            input keys
        skip_input_keys: List[str], optional
            List of input keys to skip
        result_keys : List[str], optional
            List of result keys to store. If not specified, stores all of the
            result keys
        skip_result_keys: List[str], optional
            List of result keys to skip
        append_file: bool, default False
            Add new values to the end of an existing file
        """
        # Store persistent attributes
        self.file_name     = file_name
        self.append_file   = append_file
        self.ready         = False
        self.object_dtypes = []

        self.input_keys       = input_keys
        self.skip_input_keys  = skip_input_keys
        self.result_keys      = result_keys
        self.skip_result_keys = skip_result_keys

        # Initialize attributes to be stored when the output file is created
        self.batch_size  = None
        self.type_dict   = None
        self.event_dtype = None

    @dataclass
    class DataFormat:
        """Data structure to hold writing parameters.

        Attributes
        ----------
        category : str, optional
            Category of data to store ('data' or 'result')
        dtype : type, optional
            Data type
        class_name : str, optional
            Name of the class the information comes from
        width : int, defaul t0
            Width of the tensor to store, if it is a tensor
        merge : bool, default False
            Whether to merge lists of arrays into a single dataset
        scalar : bool, default False
            Whether the data is a scalar object or not
        """
        category: str = None
        dtype: str = None
        class_name: str = None
        width: int = 0
        merge: bool = False
        scalar: bool = False

    def create(self, data_blob, result_blob=None, cfg=None):
        """Create the output file structure based on the data and result blobs.

        Parameters
        ----------
        data_blob : dict
            Dictionary containing the input data
        result_blob : dict
            Dictionary containing the output of the reconstruction chain
        cfg : dict
            Dictionary containing the ML chain configuration
        """
        # Initialize a dictionary to store keys and their properties
        self.type_dict = {}

        # Fetch the required keys to be stored and register them
        self.input_keys, self.result_keys = self.get_stored_keys(
                data_blob, result_blob)
        for key in self.input_keys:
            self.register_key(data_blob, key, 'data')
        for key in self.result_keys:
            self.register_key(result_blob, key, 'result')

        # Initialize the output HDF5 file
        with h5py.File(self.file_name, 'w') as out_file:
            # Initialize the info dataset that stores environment parameters
            if cfg is not None:
                out_file.create_dataset(
                        'info', (0,), maxshape=(None,), dtype=None)
                out_file['info'].attrs['cfg'] = yaml.dump(cfg)
                out_file['info'].attrs['version'] = __version__

            # Initialize the event dataset and their reference array datasets
            self.initialize_datasets(out_file)

            # Mark file as ready for use
            self.ready = True

    def get_stored_keys(self, data_blob, result_blob):
        """Get the list of data and or result keys to store.

        Parameters
        ----------
        data_blob : dict
            Dictionary containing the input data
        result_blob : dict
            Dictionary containing the output of the reconstruction chain

        Returns
        -------
        data_keys : list
            List of data keys to store to file
        result_keys : list
            List of result keys to store to file
        """
        # If the keys were already produced, nothing to do
        if self.ready:
            return self.input_keys, self.result_keys

        # Make sure there is something to store
        assert data_blob, "Must provide a non-empty `data_blob`."

        # Check that the input/result keys make sense,
        assert (self.input_keys is None) | (self.skip_input_keys is None), (
                "Must only specify one of `input_keys` or `skip_input_keys`")
        assert (self.result_keys is None) | (self.skip_result_keys is None), (
                "Must only specify one of `result_keys` or `skip_result_keys`")

        # Translate input_keys/skip_input_keys into a single list
        input_keys = {'index'}
        if self.input_keys is None:
            input_keys.update(data_blob.keys())
            if self.skip_input_keys is not None:
                for key in self.skip_input_keys:
                    if key in input_keys:
                        input_keys.remove(key)
        else:
            input_keys.update(self.input_keys)
            for k in self.input_keys:
                assert k in data_blob, (
                        f"Cannot store {k} as it does not appear "
                         "in the input dictionary")

        # Translate result_keys/skip_result_keys into a single list
        result_keys = set()
        if self.result_keys is None:
            if result_blob is not None:
                result_keys.update(result_blob.keys())
                if self.skip_result_keys is not None:
                    for key in self.skip_result_keys:
                        if key in result_keys:
                            result_keys.remove(key)
        else:
            result_keys.update(self.result_keys)
            for k in self.result_keys:
                assert k in result_blob, (
                        'it does not appear in the result dictionary'
                        f"Cannot store {k} as it does not appear "
                         "in the result dictionary")

        return input_keys, result_keys

    def register_key(self, blob, key, category):
        """Identify the dtype and shape objects to be dealt with.

        Parameters
        ----------
        blob : dict
            Dictionary containing the information to be stored
        key : string
            Dictionary key name
        category : string
            Data category: `data` or `result`
        """
        # Initialize a type object for this output key
        self.type_dict[key] = self.DataFormat(category)

        # Store the necessary information to know how to store a key
        if np.isscalar(blob[key]):
            # Single scalar for the entire batch (e.g. accuracy, loss, etc.)
            if isinstance(blob[key], str):
                self.type_dict[key] = h5py.string_dtype()
            else:
                self.type_dict[key] = type(blob[key])
            self.type_dict[key].scalar = True

        else:
            if np.isscalar(blob[key][0]):
                # List containing a single scalar per batch ID
                if isinstance(blob[key][0], str):
                    self.type_dict[key].dtype = h5py.string_dtype()
                else:
                    self.type_dict[key].dtype = type(blob[key][0])
                self.type_dict[key].scalar = True

            elif not hasattr(blob[key][0], '__len__'):
                # List containing one single non-standard object per batch ID
                object_dtype = self.get_object_dtype(blob[key][0])
                self.object_dtypes.append(object_dtype)
                self.type_dict[key].dtype = object_dtype
                self.type_dict[key].scalar = True
                self.type_dict[key].class_name = blob[key][0].__class__.__name__

            else:
                # List containing a list/array of objects per batch ID
                ref_obj = blob[key][0]
                if isinstance(blob[key][0], list):
                    # If simple list, check if it is empty
                    if len(blob[key][0]):
                        # If it contains simple objects, use the first
                        if not hasattr(blob[key][0][0], '__len__'):
                            ref_obj = blob[key][0][0]
                    else:
                        # If it is empty, must contain a default value
                        assert hasattr(blob[key][0], 'default'), (
                               f"Failed to find type of {key}. Lists that can "
                                "be empty should be initialized as an "
                                "ObjectList with a default object type.")
                        ref_obj = blob[key][0].default

                        # If the default value is an array, unwrap as such
                        if isinstance(ref_obj, np.ndarray):
                            self.type_dict[key].width = [0]
                            self.type_dict[key].merge = True

                if not hasattr(ref_obj, '__len__'):
                    # List containing a single list of objects per batch ID
                    object_dtype = self.get_object_dtype(ref_obj)
                    self.object_dtypes.append(object_dtype)
                    self.type_dict[key].dtype = object_dtype
                    self.type_dict[key].class_name = ref_obj.__class__.__name__

                elif (not isinstance(ref_obj, list) and
                      not ref_obj.dtype == object):
                    # List containing a single ndarray of scalars per batch ID
                    self.type_dict[key].dtype = ref_obj.dtype
                    if len(ref_obj.shape) == 2:
                        self.type_dict[key].width = ref_obj.shape[1]

                elif isinstance(ref_obj, (list, np.ndarray)):
                    # List containing a list/array of ndarrays per batch ID
                    widths = []
                    for el in ref_obj:
                        width, same_width = 0, 0
                        same_width = True
                        if len(el.shape) == 2:
                            width = el.shape[1]
                        widths.append(width)
                        same_width &= width == widths[0]

                    self.type_dict[key].dtype = ref_obj[0].dtype
                    self.type_dict[key].width = widths
                    self.type_dict[key].merge = same_width

                else:
                    dtype = type(blob[key][0])
                    raise TypeError(
                            f"Do not know how to store output of type {dtype} "
                            f"in key {key}")

    def get_object_dtype(self, obj):
        """Loop over the attributes of a class to figure out what to store.

        This function assumes that the class only posseses getters that return
        either a scalar, string or np.ndarrary.

        Parameters
        ----------
        object : class
            Instance of an class used to identify attribute types

        Returns
        -------
        list
            List of (key, dtype) pairs
        """
        object_dtype = []
        for key, val in asdict(obj).items():
            # Append the relevant data type
            if isinstance(val, str):
                # String
                object_dtype.append((key, h5py.string_dtype()))

            elif hasattr(obj, 'enum_attrs') and key in obj.enum_attrs:
                # Recognized enumerated list
                enum_dtype = h5py.enum_dtype(
                        obj.enum_attrs[key], basetype=type(val))
                object_dtype.append((key, enum_dtype))

            elif np.isscalar(val):
                # Non-string, non-enumerated scalar. Force bool onto shorts
                dtype = type(val) if not isinstance(val, bool) else np.uint8
                object_dtype.append((key, dtype))

            elif (hasattr(obj, 'fixed_length_attrs') and
                  key in obj.fixed_length_attrs):
                # Fixed-length array of scalars
                object_dtype.append((key, val.dtype, len(val)))

            elif isinstance(val, np.ndarray):
                # Variable-length array of scalars
                object_dtype.append((key, h5py.vlen_dtype(val.dtype)))

            else:
                raise ValueError(
                        f"Attribute {key} of {obj} has unrecognized an "
                        "unrecognized type: {type(val)}")

        return object_dtype

    def initialize_datasets(self, out_file):
        """Create place hodlers for all the datasets to be filled.

        Parameters
        ----------
        out_file : h5py.File
            HDF5 file instance
        """
        # Initialize the datasets, store the general type of the event
        self.event_dtype = []
        ref_dtype = h5py.special_dtype(ref=h5py.RegionReference)
        for key, val in self.type_dict.items():
            # Add a dataset reference for this key to the event dtype
            self.event_dtype.append((key, ref_dtype))
            if not isinstance(val.width, list):
                # If the key contains a list of objects of identical shape
                shape = (0, val.width) if val.width else (0,)
                maxshape = (None, val.width) if val.width else (None,)
                out_file.create_dataset(
                        key, shape, maxshape=maxshape, dtype=val.dtype)

                # Store the class name to rebuild it later, if relevant
                if val.class_name is not None:
                    out_file[key].attrs['class_name'] = val.class_name

            elif not val.merge:
                # If the elements of the list are of variable widths, refer to
                # one dataset per element. An index is stored alongside the
                # dataset to break.
                group = out_file.create_group(key)

                n_arrays = len(val.width)
                shape, maxshape = (0, n_arrays), (None, n_arrays)
                group.create_dataset(
                        'index', shape, maxshape=maxshape, dtype=ref_dtype)

                for i, w in enumerate(val.width):
                    shape = (0, w) if w else (0,)
                    maxshape = (None, w) if w else (None,)
                    el = f'element_{i}'
                    group.create_dataset(
                            el, shape, maxshape=maxshape, dtype=val.dtype)

            else:
                # If the  elements of the list are of equal width, store them
                # all to one dataset. An index is stored alongside the dataset
                # to break it into individual elements downstream.
                group = out_file.create_group(key)

                shape = (0, val.width[0]) if val.width[0] else (0,)
                maxshape = (None, val.width[0]) if val.width[0] else (None,)
                group.create_dataset(
                        'index', (0,), maxshape=(None,), dtype=ref_dtype)
                group.create_dataset(
                        'elements', shape, maxshape=maxshape, dtype=val.dtype)

            # Give relevant attributes to the dataset
            out_file[key].attrs['category'] = val.category
            out_file[key].attrs['scalar'] = val.scalar

        out_file.create_dataset(
                'events', (0,), maxshape=(None,), dtype=self.event_dtype)

    def append(self, data_blob=None, result_blob=None, cfg=None):
        """Append the HDF5 file with the content of a batch.

        Parameters
        ----------
        result_blob : dict
            Dictionary containing the output of the reconstruction chain
        data_blob : dict
            Dictionary containing the input data
        cfg : dict
            Dictionary containing the ML chain configuration
        """
        # If this function has never been called, initialiaze the HDF5 file
        if (not self.ready and
            (not self.append_file or os.path.isfile(self.file_name))):
            self.create(data_blob, result_blob, cfg)
            self.ready = True

        # Append file
        self.batch_size = len(data_blob['index'])
        with h5py.File(self.file_name, 'a') as out_file:
            # Loop over batch IDs
            for batch_id in range(self.batch_size):
                # Initialize a new event
                event = np.empty(1, self.event_dtype)

                # Initialize a dictionary of references to be passed to the
                # event dataset and store the input and result keys
                for key in self.input_keys:
                    self.append_key(out_file, event, data_blob, key, batch_id)
                for key in self.result_keys:
                    self.append_key(out_file, event, result_blob, key, batch_id)

                # Append event
                event_id = len(out_file['events'])
                event_ds = out_file['events']
                event_ds.resize(event_id + 1, axis=0) # pylint: disable=E1101
                event_ds[event_id] = event

    def append_key(self, out_file, event, blob, key, batch_id):
        """Stores data key in a specific dataset of an HDF5 file.

        Parameters
        ----------
        out_file : h5py.File
            HDF5 file instance
        event : dict
            Dictionary of objects that make up one event
        blob : dict
            Dictionary containing the information to be stored
        key : string
            Dictionary key name
        batch_id : int
            Batch ID to be stored
        """
        # Get the data type and store it
        val = self.type_dict[key]
        if not val.merge and not isinstance(val.width, list):
            # Store single arrays
            if np.isscalar(blob[key]):
                # If a data product is a single scalar, use it for every entry
                array = [blob[key]]

            else:
                # Otherwise, get the data corresponding to the current entry
                array = blob[key][batch_id]
                if val.scalar:
                    array = [array]

            if val.dtype in self.object_dtypes:
                self.store_objects(out_file, event, key, array, val.dtype)
            else:
                self.store(out_file, event, key, array)

        elif not val.merge:
            # Store the array and its reference for each element in the list
            array_list = blob[key][batch_id]
            self.store_jagged(out_file, event, key, array_list)

        else:
            # Store one array of for all in the list and a index to break them
            array_list = blob[key][batch_id]
            self.store_flat(out_file, event, key, array_list)

    @staticmethod
    def store(out_file, event, key, array):
        """Stores an `ndarray` in the file and stores its mapping in the event
        dataset.

        Parameters
        ----------
        out_file : h5py.File
            HDF5 file instance
        event : dict
            Dictionary of objects that make up one event
        key: str
            Name of the dataset in the file
        array : np.ndarray
            Array to be stored
        """
        # Extend the dataset, store array
        dataset = out_file[key]
        current_id = len(dataset)
        dataset.resize(current_id + len(array), axis=0)
        dataset[current_id:current_id + len(array)] = array

        # Define region reference, store it at the event level
        region_ref = dataset.regionref[current_id:current_id + len(array)]
        event[key] = region_ref

    @staticmethod
    def store_jagged(out_file, event, key, array_list):
        """Stores a jagged list of arrays in the file and stores an index
        mapping for each array element in the event dataset.

        Parameters
        ----------
        out_file : h5py.File
            HDF5 file instance
        event : dict
            Dictionary of objects that make up one event
        key: str
            Name of the dataset in the file
        array_list : list(np.ndarray)
            List of arrays to be stored
        """
        # Extend the dataset, store combined array
        region_refs = []
        for i, array in enumerate(array_list):
            dataset = out_file[key][f'element_{i}']
            current_id = len(dataset)
            dataset.resize(current_id + len(array), axis=0)
            dataset[current_id:current_id + len(array)] = array

            region_ref = dataset.regionref[current_id:current_id + len(array)]
            region_refs.append(region_ref)

        # Define the index which stores a list of region_refs
        index = out_file[key]['index']
        current_id = len(dataset)
        index.resize(current_id+1, axis=0)
        index[current_id] = region_refs

        # Define a region reference to all the references,
        # store it at the event level
        region_ref = index.regionref[current_id:current_id+1]
        event[key] = region_ref

    @staticmethod
    def store_flat(out_file, event, key, array_list):
        """Stores a concatenated list of arrays in the file and stores its
        index mapping in the event dataset to break them.

        Parameters
        ----------
        out_file : h5py.File
            HDF5 file instance
        event : dict
            Dictionary of objects that make up one event
        key: str
            Name of the dataset in the file
        array_list : list(np.ndarray)
            List of arrays to be stored
        """
        # Extend the dataset, store combined array
        array = np.concatenate(array_list) if len(array_list) else []
        dataset = out_file[key]['elements']
        first_id = len(dataset)
        dataset.resize(first_id + len(array), axis=0)
        dataset[first_id:first_id + len(array)] = array

        # Loop over arrays in the list, create a reference for each
        index = out_file[key]['index']
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
        region_ref = index.regionref[current_id:current_id + len(array_list)]
        event[key] = region_ref

    @staticmethod
    def store_objects(out_file, event, key, array, obj_dtype):
        """Stores a list of objects with understandable attributes in the file
        and stores its mapping in the event dataset.

        Parameters
        ----------
        out_file : h5py.File
            HDF5 file instance
        event : dict
            Dictionary of objects that make up one event
        key: str
            Name of the dataset in the file
        array : np.ndarray
            Array of objects or dictionaries to be stored
        obj_dtype : list
            List of (key, dtype) pairs which specify what's to store
        """
        # Convert list of objects to list of storable objects
        objects = np.empty(len(array), obj_dtype)
        for i, obj in enumerate(array):
            objects[i] = tuple(asdict(obj).values())

        # Extend the dataset, store array
        dataset = out_file[key]
        current_id = len(dataset)
        dataset.resize(current_id + len(array), axis=0)
        dataset[current_id:current_id + len(array)] = objects

        # Define region reference, store it at the event level
        region_ref = dataset.regionref[current_id:current_id + len(array)]
        event[key] = region_ref


class CSVWriter:
    """Writes data to a CSV file.

    Builds a CSV file to store the output of the analysis tools. It can only be
    used to store relatively basic quantities (scalars, strings, etc.).

    Typical configuration should look like:

    .. code-block:: yaml

        io:
          ...
          writer:
            name: csv
            file_name: output.csv
    """
    name = 'csv'

    def __init__(self, file_name='output.csv', append_file=False,
                 accept_missing=False):
        """Initialize the basics of the output file.

        Parameters
        ----------
        file_name : str, default 'output.csv'
            Name of the output CSV file
        append_file : bool, default False
            Add more rows to an existing CSV file
        accept_missing : bool, default True
            Tolerate missing keys
        """
        self.file_name      = file_name
        self.append_file    = append_file
        self.accept_missing = accept_missing
        self.result_keys    = None
        if self.append_file:
            if not os.path.isfile(file_name):
                raise FileNotFoundError(
                        f"File not found at path: {file_name}. When using "
                         "`append=True` in CSVWriter, the file must exist at "
                         "the prescribed path before data is written to it.")

            with open(self.file_name, 'r', encoding='utf-8') as out_file:
                self.result_keys = out_file.readline().split(',')

    def create(self, result_blob):
        """Initialize the header of the CSV file, record the keys to be stored.

        Parameters
        ----------
        result_blob : dict
            Dictionary containing the output of the reconstruction chain
        """
        # Save the list of keys to store
        self.result_keys = list(result_blob.keys())

        # Create a header and write it to file
        with open(self.file_name, 'w', encoding='utf-8') as out_file:
            header_str = ','.join(self.result_keys)
            out_file.write(header_str + '\n')

    def append(self, result_blob):
        """Append the CSV file with the output.

        Parameters
        ----------
        result_blob : dict
            Dictionary containing the output of the reconstruction chain
        """
        # Fetch the values to store
        if self.result_keys is None:
            # If this function has never been called, initialiaze the CSV file
            self.create(result_blob)

        else:
            # If it has, check that the list of keys is identical
            if list(result_blob.keys()) != self.result_keys:
                # If it is not identical, check the discrepancies
                missing = self.array_diff(self.result_keys, result_blob.keys())
                excess  = self.array_diff(result_blob.keys(), self.result_keys)
                if len(excess):
                    raise AssertionError(
                             "There are keys in this entry which were not "
                             "present when the CSV file was initialized. "
                            f"New keys: {list(excess)}")

                if not self.accept_missing:
                    raise AssertionError(
                             "There are keys missing in this entry which were "
                             "present when the CSV file was initialized. "
                            f"Missing keys: {list(missing)}")

                new_result_blob = {k:-1 for k in self.result_keys}
                for k, v in result_blob.items():
                    new_result_blob[k] = v
                result_blob = new_result_blob

        # Append file
        with open(self.file_name, 'a', encoding='utf-8') as out_file:
            result_str = ','.join(
                    [str(result_blob[k]) for k in self.result_keys])
            out_file.write(result_str + '\n')

    @staticmethod
    def array_diff(array_x, array_y):
        """Compare the content of two arrays.

        This functions returns the elemnts of the first array that
        do not appear in the second array.

        Parameters
        ----------
        array_x : List[str]
            First array of strings
        array_y : List[str]
            Second array of strings

        Returns
        -------
        Set[str]
            Set of keys that appear in `array_x` but not in `array_y`.
        """
        return set(array_x).difference(set(array_y))
