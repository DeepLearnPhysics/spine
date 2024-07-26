"""Module with a parent class of all data structures."""

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class DataBase:
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

    # Attributes specifying vector components
    _vec_attrs = []

    # String attributes
    _str_attrs = []

    # Attributes to concatenate when merging objects
    _cat_attrs = []

    # Attributes that should not be stored to file (long-form attributes)
    _skip_attrs = []

    # Euclidean axis labels
    _axes = ['x', 'y', 'z']

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
                if not isinstance(size, tuple):
                    dtype = np.float32
                else:
                    size, dtype = size
                setattr(self, attr, np.full(size, -np.inf, dtype=dtype))

        # Provide default values to the variable-length array attributes
        for attr, dtype in self._var_length_attrs.items():
            if getattr(self, attr) is None:
                if not isinstance(dtype, tuple):
                    setattr(self, attr, np.empty(0, dtype=dtype))
                else:
                    width, dtype = dtype
                    setattr(self, attr, np.empty((0, width), dtype=dtype))

        # Make sure the strings are not binary
        for attr in self._str_attrs:
            if isinstance(getattr(self, attr), bytes):
                setattr(self, attr, getattr(self, attr).decode())

    def __getstate__(self):
        """Returns the variables to be pickled.

        This is needed because the derived variables are stored as property
        objects and are not naturally pickleable. This function simply skips
        the private attributes which might be problematic to pickle.

        Returns
        -------
        dict
            Dictionary representation of the object
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __setstate__(self, state):
        """Sets the object state from a dictionary.

        Parameters
        ----------
        dict
            Dictionary representation of the object
        """
        self.__dict__.update(state)

    def set_precision(self, precision):
        """Casts all the vector attributes to a different precision.

        Parameters
        ----------
        int : default 4
            Precision in number of bytes (half=2, single=4, double=8)
        """
        assert precision in [2, 4, 8], (
                "Set the vector attribute precision for this object.")
        for attr in self.fixed_length_attrs + self.variable_length_attrs:
            val = getattr(self, attr)
            dtype = f'{val.dtype.str[:-1]}{precision}'
            setattr(self, attr, val.astype(dtype))

    def as_dict(self):
        """Returns the data class as dictionary of (key, value) pairs.

        Returns
        -------
        dict
            Dictionary of attribute names and their values
        """
        obj_dict = {}
        for k, v in self.__dict__.items():
            if not k in self._skip_attrs:
                if not k.startswith('_'):
                    obj_dict[k] = v
                else:
                    obj_dict[k[1:]] = getattr(self, k[1:])

        return obj_dict

    def scalar_dict(self, attrs=None):
        """Returns the data class attributes as a dictionary of scalars.

        This is useful when storing data classes in CSV files, which expect
        a single scalar per column in the table.

        Parameters
        ----------
        attrs : List[str], optional
            List of attribute names to include in the dictionary. If not
            specified, all the keys are included.
        """
        # Loop over the attributes of the data class
        scalar_dict, found = {}, []
        for attr, value in self.as_dict().items():
            # If the attribute is not requested, skip
            if attrs is not None and attr not in attrs:
                continue
            else:
                found.append(attr)

            # If the attribute is long-form attribute, skip it
            if attr in self._skip_attrs or attr in self._var_length_attrs:
                continue

            # Dispatch
            if np.isscalar(value):
                # If the attribute is a scalar, store as is
                scalar_dict[attr] = value

            elif attr in (self._pos_attrs + self._vec_attrs):
                # If the attribute is a position or vector, expand with axis
                for i, v in enumerate(value):
                    scalar_dict[f'{attr}_{self._axes[i]}'] = v

            elif attr in self._fixed_length_attrs:
                # If the attribute is a fixed length array, expand with index
                for i, v in enumerate(value):
                    scalar_dict[f'{attr}_{i}'] = v

            else:
                raise ValueError(
                        f"Cannot expand the `{attr}` attribute of "
                        f"`{self.__cls__.__name__}` to scalar values.")

        if attrs is not None and len(attrs) != len(found):
            class_name = self.__class__.__name__
            miss = list(set(attrs).difference(set(found)))
            for attr in miss:
                if attr in scalar_dict:
                    scalar_dict.pop(attr)
            # raise AttributeError(
            #         f"Attribute(s) {miss} do(es) not appear in {class_name}.")

        return scalar_dict

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
        """Fetches the list of attributes to not store to file.

        Returns
        -------
        List[str]
            List of attributes to exclude from the storage process
        """
        return self._skip_attrs


@dataclass
class PosDataBase(DataBase):
    """Base class of for data structures with positional attributes.

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

    def to_px(self, meta):
        """Converts the coordinates of the positional attributes to pixel.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        assert self.units != 'px', "Units already expressed in pixels"
        self.units = 'px'
        for attr in self._pos_attrs:
            setattr(self, attr, meta.to_px(getattr(self, attr)))
