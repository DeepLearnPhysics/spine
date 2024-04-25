"""Module with a parent class of all data structures."""

from dataclasses import dataclass

import numpy as np


@dataclass
class DataStructBase:
    """Base class of all data structures.

    Defines basic methods shared by all data structures.
    """

    # Enumerated attributes
    _enum_attrs = {}

    # Fixed-length attributes as (key, size) pairs
    _fixed_length_attrs = {}

    # Variable-length attributes as (key, dtype) pairs
    _var_length_attrs = {}

    # Attributes specifying coordinates
    _pos_attrs = []

    # String attributes
    _str_attrs = []

    # Attributes that should not be stored to file
    _skip_attrs = []

    def __post_init__(self):
        """Immediately called after building the class attributes.

        Provides two functions:
        - Gives default values to array-like attributes. If a default value was
          provided in the attribute definition, all instances of this class
          would point to the same memory location.
        - Casts strings when they are provided as binary objects, which is the
          format one gets when loading string from HDF5 files.
        """
        # Provide default values to the fixed-length array attributes
        for attr, size in self._fixed_length_attrs.items():
            if getattr(self, attr) is None:
                setattr(self, attr, np.full(size, -np.inf, dtype=np.float32))

        # Provide default values to the variable-length array attributes
        for attr, dtype in self._var_length_attrs.items():
            if getattr(self, attr) is None:
                if not isinstance(dtype, tuple):
                    setattr(self, attr, np.empty(0, dtype=dtype))
                else:
                    size, dtype = dtype
                    setattr(self, attr, np.empty((0, size), dtype=dtype))

        # Make sure the strings are not binary
        for attr in self._str_attrs:
            if isinstance(getattr(self, attr), bytes):
                setattr(self, attr, getattr(self, attr).decode())

    @property
    def fixed_length_attrs(self):
        """Fetches the dictionary of fixed-length array attributes.

        Returns
        -------
        Dict[str, int]
            Dictioary which maps fixed-length attributes onto their length
        """
        return self._fixed_length_attrs

    @property
    def var_length_attrs(self):
        """Fetches the list of variable-length array attributes.

        Returns
        -------
        Dict[str, type]
            Dictionary which maps variable-length attributes onto their type
        """
        return self._fixed_length_attrs

    @property
    def enum_attrs(self):
        """Fetches the list of enumerated attributes.

        Returns
        -------
        Dict[int, Dict[int, str]]
            Dictionary which maps names onto enumerator descriptors
        """
        return self._enum_attrs

    @property
    def skip_attrs(self):
        """Fetches the list attributes to not store to file.

        Returns
        -------
        List[str]
            List of attributes to exclude from the storage process
        """
        return self._skip_attrs


@dataclass
class PosDataStructBase(DataStructBase):
    """Base class of for data structes with positional attributes.

    Includes method to convert positional attributes

    Attributes
    ----------
    units : str
        Units in which the position attributes are expressed
    """
    units = 'cm'

    def __post_init__(self):
        """Immediately called after building the class attributes.

        Makes sure the units are not binary and that they are recognized.
        """
        # Call the main post initialization function
        super().__post_init__()

        # Parse the units
        if isinstance(self.units, bytes):
            self.units = self.units.decode()

        assert self.units in ['cm', 'px'], "Units can only be `cm` or `px`."

    def to_cm(self, meta):
        """Converts the coordinates of the positional attributes to cm.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        assert self.units != 'cm', "Units already expressed in cm"
        self.units = 'cm'
        for attr in self._pos_attrs:
            setattr(self, attr, meta.to_cm(getattr(self, attr)))

    def to_pixel(self, meta):
        """Converts the coordinates of the positional attributes to pixel.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        assert self.units != 'px', "Units already expressed in pixels"
        self.units = 'px'
        for attr in self._pos_attrs:
            setattr(self, attr, meta.to_pixel(getattr(self, attr)))
