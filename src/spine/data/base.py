"""Module with a parent class of all data structures."""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(eq=False)
class DataBase:
    """Base class of all data structures.

    Defines basic methods shared by all data structures.
    """

    # Enumerated attributes
    _enum_attrs = ()

    # Fixed-length attributes as (key, size) or (key, (size, dtype)) pairs
    _fixed_length_attrs = ()

    # Variable-length attributes as (key, dtype) or (key, (width, dtype)) pairs
    _var_length_attrs = ()

    # Attributes specifying coordinates
    _pos_attrs = ()

    # Attributes specifying vector components
    _vec_attrs = ()

    # String attributes
    _str_attrs = ()

    # Boolean attributes
    _bool_attrs = ()

    # Index attributes
    _index_attrs = ()

    # Attributes to concatenate when merging objects
    _cat_attrs = ()

    # Attributes that must never be stored to file
    _skip_attrs = ()

    # Attributes that must not be stored to file when storing lite files
    _lite_skip_attrs = ()

    # Euclidean axis labels
    _axes = ("x", "y", "z")

    def __post_init__(self):
        """Immediately called after building the class attributes.

        Provides two functions:
        - Gives default values to array-like attributes. If a default value was
          provided in the attribute definition, all instances of this class
          would point to the same memory location.
        - Casts strings when they are provided as binary objects, which is the
          format one gets when loading string from HDF5 files.
        """
        # Provide default values to the variable-length array attributes
        for attr, dtype in self._var_length_attrs:
            if getattr(self, attr) is None:
                if not isinstance(dtype, tuple):
                    setattr(self, attr, np.empty(0, dtype=dtype))
                else:
                    width, dtype = dtype
                    setattr(self, attr, np.empty((0, width), dtype=dtype))

        # Provide default values to the fixed-length array attributes
        for attr, size in self._fixed_length_attrs:
            if getattr(self, attr) is None:
                if not isinstance(size, tuple):
                    dtype = np.float32
                else:
                    size, dtype = size
                setattr(self, attr, np.full(size, -np.inf, dtype=dtype))

        # Cast stored binary strings back to regular strings
        for attr in self._str_attrs:
            if isinstance(getattr(self, attr), bytes):
                setattr(self, attr, getattr(self, attr).decode())

        # Cast stored 8-bit unsigned integers back to booleans
        for attr in self._bool_attrs:
            if isinstance(getattr(self, attr), np.uint8):
                setattr(self, attr, bool(getattr(self, attr)))

    def __eq__(self, other):
        """Checks that all attributes of two class instances are the same.

        This overloads the default dataclass `__eq__` method to include an
        appopriate check for vector (numpy) attributes.

        Parameters
        ----------
        other : obj
            Other instance of the same object class

        Returns
        -------
        bool
            `True` if all attributes of both objects are identical
        """
        # Check that the two objects belong to the same class
        if self.__class__ != other.__class__:
            return False

        # Check that all base attributes are identical
        for k, v in self.__dict__.items():
            if np.isscalar(v):
                # For scalars, regular comparison will do
                if getattr(other, k) != v:
                    return False

            else:
                # For vectors, compare all elements
                v_other = getattr(other, k)
                if v.shape != v_other.shape or (v_other != v).any():
                    return False

        return True

    def set_precision(self, precision):
        """Casts all the vector attributes to a different precision.

        Parameters
        ----------
        int : default 4
            Precision in number of bytes (half=2, single=4, double=8)
        """
        assert precision in [
            2,
            4,
            8,
        ], "Set the vector attribute precision for this object."
        for attr in self.fixed_length_attrs + self.variable_length_attrs:
            val = getattr(self, attr)
            dtype = f"{val.dtype.str[:-1]}{precision}"
            setattr(self, attr, val.astype(dtype))

    def shift_indexes(self, shifts):
        """Apply offsets to index attributes in place.

        Invalid indexes (-1) are not offset to prevent making them valid.

        Parameters
        ----------
        shifts : Union[int, Dict[str, int]]
            Shift(s) to apply to the index attributes. If provided as a
            dictionary, the shifts are tied to a specific attribute.
        """
        # Dispatch
        for attr in self._index_attrs:
            value = getattr(self, attr)
            if not np.isscalar(value) or value > -1:
                if not isinstance(shifts, dict):
                    setattr(self, attr, value + shifts)
                else:
                    setattr(self, attr, value + shifts[attr])

    def as_dict(self, lite=False):
        """Returns the data class as dictionary of (key, value) pairs.

        Parameters
        ----------
        lite : bool, default False
            If `True`, the `_lite_skip_attrs` are dropped

        Returns
        -------
        dict
            Dictionary of attribute names and their values
        """
        # Build a list of attributes to skip
        if not lite:
            skip_attrs = self._skip_attrs
        else:
            skip_attrs = (*self._skip_attrs, *self._lite_skip_attrs)

        return {k: v for k, v in asdict(self).items() if not k in skip_attrs}

    def scalar_dict(self, attrs=None, lengths=None, lite=False):
        """Returns the data class attributes as a dictionary of scalars.

        This is useful when storing data classes in CSV files, which expect
        a single scalar per column in the table.

        Parameters
        ----------
        attrs : List[str], optional
            List of attribute names to include in the dictionary. If not
            specified, all the keys are included.
        lengths : Dict[str, int], optional
            Specifies the length of variable-length attributes
        lite : bool, default False
            If `True`, the `_lite_skip_attrs` are dropped
        """
        # Loop over the attributes of the data class
        lengths = lengths or {}
        scalar_dict, found = {}, []
        for attr, value in self.as_dict(lite).items():
            # If the attribute is not requested, skip
            if attrs is not None and attr not in attrs:
                continue
            else:
                found.append(attr)

            # Dispatch
            if np.isscalar(value):
                # If the attribute is a scalar, store as is
                scalar_dict[attr] = value

            elif attr in (self._pos_attrs + self._vec_attrs):
                # If the attribute is a position or vector, expand with axis
                for i, v in enumerate(value):
                    scalar_dict[f"{attr}_{self._axes[i]}"] = v

            elif attr in self.fixed_length_attrs:
                # If the attribute is a fixed-length array, expand with index
                for i, v in enumerate(value):
                    scalar_dict[f"{attr}_{i}"] = v

            elif attr in self.var_length_attrs:
                if attr in lengths:
                    # If the attribute is a variable-length array with a length
                    # provided, resize it to match that length and store it
                    for i in range(lengths[attr]):
                        if i < len(value):
                            scalar_dict[f"{attr}_{i}"] = value[i]
                        else:
                            scalar_dict[f"{attr}_{i}"] = None

                else:
                    # If the attribute is a variable-length array of
                    # indeterminate length, do not store it
                    assert attrs is None or attr not in attrs, (
                        f"Cannot cast {attr} to scalars. To cast a variable-"
                        "length array, must provide a fixed length."
                    )
                    continue

            else:
                raise ValueError(
                    f"Cannot expand the `{attr}` attribute of "
                    f"`{self.__class__.__name__}` to scalar values."
                )

        if attrs is not None and len(attrs) != len(found):
            class_name = self.__class__.__name__
            miss = list(set(attrs).difference(set(found)))
            raise AttributeError(
                f"Attribute(s) {miss} do(es) not appear in {class_name}."
            )

        return scalar_dict

    @property
    def fixed_length_attrs(self):
        """Fetches the dictionary of fixed-length array attributes as a dictionary.

        Returns
        -------
        Dict[str, int]
            Dictionary which maps fixed-length attributes onto their length
        """
        return dict(self._fixed_length_attrs)

    @property
    def var_length_attrs(self):
        """Fetches the list of variable-length array attributes as a dictionary.

        Returns
        -------
        Dict[str, type]
            Dictionary which maps variable-length attributes onto their type
        """
        return dict(self._var_length_attrs)

    @property
    def enum_attrs(self):
        """Fetches the list of enumerated attributes as a dictionary.

        Returns
        -------
        Dict[int, Dict[int, str]]
            Dictionary which maps names onto enumerator descriptors
        """
        return {k: dict(v) for k, v in self._enum_attrs}

    @property
    def index_attrs(self):
        """Fetches the list of attributes that correspond to indexes.

        Returns
        -------
        List[str]
            List of attributes that specificy indexes
        """
        return self._index_attrs

    @property
    def skip_attrs(self):
        """Fetches the list of attributes to not store to file.

        Returns
        -------
        List[str]
            List of attributes to exclude from the storage process
        """
        return self._skip_attrs

    @property
    def lite_skip_attrs(self):
        """Fetches the list of attributes to not store to lite file.

        Returns
        -------
        List[str]
            List of attributes to exclude from the storage process
        """
        return self._lite_skip_attrs


@dataclass(eq=False)
class PosDataBase(DataBase):
    """Base class of for data structures with positional attributes.

    Includes method to convert positional attributes

    Attributes
    ----------
    units : str
        Units in which the position attributes are expressed
    """

    units = "cm"

    def __post_init__(self):
        """Immediately called after building the class attributes.

        Makes sure the units are not binary and that they are recognized.
        """
        # Call the main post initialization function
        super().__post_init__()

        # Parse the units
        if isinstance(self.units, bytes):
            self.units = self.units.decode()

        assert self.units in ["cm", "px"], "Units can only be `cm` or `px`."

    def to_cm(self, meta):
        """Converts the coordinates of the positional attributes to cm.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        assert self.units != "cm", "Units already expressed in cm"
        self.units = "cm"
        for attr in self._pos_attrs:
            setattr(self, attr, meta.to_cm(getattr(self, attr)))

    def to_px(self, meta):
        """Converts the coordinates of the positional attributes to pixel.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        assert self.units != "px", "Units already expressed in pixels"
        self.units = "px"
        for attr in self._pos_attrs:
            setattr(self, attr, meta.to_px(getattr(self, attr)))
