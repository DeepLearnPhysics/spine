"""Logical product selection and physical dtype discovery for HDF5 writers."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np

from .common import DataFormat


class SchemaDiscoveryBackend:
    """Infer storage formats independently of the selected HDF5 layout."""

    ready: bool
    keys: set[str] | None
    skip_keys: list[str] | None
    dummy_ds: dict[str, Any] | None
    lite: bool
    source_index_keys: dict[str, str]

    def get_stored_keys(self, data: dict[str, Any]) -> set[str]:
        """Resolve the complete set of logical products to store.

        Parameters
        ----------
        data : dict
            Available batched data products.

        Returns
        -------
        set[str]
            Validated product keys, including administrative provenance.
        """
        # If the keys were already produced, nothing to do
        if self.ready and self.keys is not None:
            return self.keys

        if self.keys is not None and self.skip_keys is not None:
            raise ValueError("Must not specify both `keys` and `skip_keys`.")

        # Translate keys/skip_keys into a single set
        keys = {"index"}
        if self.keys is None:
            keys.update(data.keys())
        else:
            keys.update(self.keys)
            for key in self.keys:
                if key not in data:
                    raise KeyError(
                        f"Cannot store `{key}` because it is not present in data."
                    )

        # Persist the original source entry provenance under explicit names.
        for key, source_key in self.source_index_keys.items():
            if key in data:
                keys.add(source_key)

        if self.skip_keys is not None:
            for key in self.skip_keys:
                if key not in keys:
                    raise KeyError(
                        f"Key {key} appears in `skip_keys` but does not "
                        "appear in the dictionary of data products."
                    )
                keys.remove(key)

        # Add dummy keys to the list, if requested
        if self.dummy_ds is not None:
            for key in self.dummy_ds:
                if key in keys:
                    raise KeyError(
                        f"Dummy dataset `{key}` conflicts with a real product."
                    )
            keys.update(self.dummy_ds.keys())

        return keys

    def get_data_formats(
        self, data: dict[str, Any], keys: set[str]
    ) -> tuple[dict[str, DataFormat], list[list[tuple[str, type]]]]:
        """Infer physical formats for all selected products.

        Parameters
        ----------
        data : dict
            Available batched data products.
        keys : set[str]
            Product keys selected for storage.

        Returns
        -------
        tuple
            Per-key formats and the unique compound object dtypes.
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

    def get_data_type(self, data: dict[str, Any], key: str) -> DataFormat:
        """Infer the physical format of one logical product.

        Parameters
        ----------
        data : dict
            Available batched data products.
        key : str
            Product key.

        Returns
        -------
        DataFormat
            DataFormat object containing the data type information for the key
        """
        # Initialize a type object for this output key
        data_format = DataFormat()

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

    def get_object_dtype(self, obj: Any) -> list[tuple[str, type]]:
        """Build a compound dtype description from a SPINE data object.

        Stored object fields may be strings, enumerations, scalar values, or
        fixed- and variable-length NumPy arrays.

        Parameters
        ----------
        obj : object
            Object exposing ``as_dict`` and optional enum/fixed-field metadata.

        Returns
        -------
        list[tuple[str, type]]
            Compound field names and their HDF5-compatible dtypes.
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

            elif key in getattr(obj, "fixed_length_attrs", ()):
                # Fixed-length array of scalars
                object_dtype.append((key, val.dtype, len(val)))

            elif isinstance(val, np.ndarray):
                # Variable-length array of scalars
                object_dtype.append((key, h5py.vlen_dtype(val.dtype)))

            else:
                raise ValueError(
                    f"Attribute `{key}` of {obj} has unrecognized type " f"{type(val)}."
                )

        return object_dtype
