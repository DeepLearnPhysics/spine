"""Module with a parent class of all data structures.

Units Metadata Convention
--------------------------
Fields can specify units in their metadata using the FieldMetadata class:

1. **Fixed Units** (e.g., 'MeV', 'GeV', 'ns', 'us', 'MeV/c'):
   Use fixed unit strings for physical quantities that are independent of the
   coordinate system representation. These units never change regardless of
   whether spatial coordinates are in pixels or centimeters.

   Examples:
   - Energy: 'MeV', 'GeV'
   - Time: 'ns', 'us', 's'
   - Momentum: 'MeV/c', 'GeV/c'
   - dE/dx: 'MeV/cm'

   >>> from spine.data.field import FieldMetadata
   >>> energy: float = field(default=np.nan, metadata=FieldMetadata(units='MeV'))

2. **Instance Units** (use the literal string 'instance'):
   Use 'instance' for spatial quantities that transform with unit conversion
   methods (to_cm(), to_px()). These fields follow the instance's `units`
   attribute, which can be either 'cm' or 'px'.

   Examples:
   - Position coordinates (x, y, z positions)
   - Spatial distances and lengths
   - Spatial extents and sizes

   >>> position: np.ndarray = field(
   ...     default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
   ...     metadata=FieldMetadata(
   ...         length=3, dtype=np.float32, category='position', units='instance'
   ...     )
   ... )
   >>> length: float = field(default=np.nan, metadata=FieldMetadata(units='instance'))

The field_units property dynamically resolves 'instance' to the current value
of the instance's units attribute, providing accurate units for all fields at
runtime.

Note: Direction vectors (unit vectors) typically have no units metadata since
they are unitless normalized directions.
"""

from dataclasses import dataclass, fields
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Self,
    Tuple,
    Union,
    cast,
)

import numpy as np

from spine.utils.docstring import merge_ancestor_docstrings

from .field import FieldMetadata

if TYPE_CHECKING:
    from spine.data.larcv.meta import Meta


@dataclass(eq=False)
class DataBase:
    """Base class of all data structures.

    Defines basic methods shared by all data structures.
    """

    # Euclidean axis labels
    _axes: ClassVar[Tuple[str, str, str]] = ("x", "y", "z")

    # Cached attribute lists (computed once per class on first instance)
    # Includes both field-based and derived property attributes
    _attrs_cached: ClassVar[bool] = False  # Sentinel to track if caching is done
    _fixed_length_attrs: ClassVar[Tuple[str, ...]] = ()
    _var_length_attrs: ClassVar[Tuple[str, ...]] = ()
    _pos_attrs: ClassVar[Tuple[str, ...]] = ()
    _vec_attrs: ClassVar[Tuple[str, ...]] = ()
    _index_attrs: ClassVar[Tuple[str, ...]] = ()
    _skip_attrs: ClassVar[Tuple[str, ...]] = ()
    _lite_skip_attrs: ClassVar[Tuple[str, ...]] = ()
    _cat_attrs: ClassVar[Tuple[str, ...]] = ()
    _str_attrs: ClassVar[Tuple[str, ...]] = ()
    _bool_attrs: ClassVar[Tuple[str, ...]] = ()
    _derived_attrs: ClassVar[Tuple[str, ...]] = ()
    _global_units_attrs: ClassVar[Tuple[str, ...]] = ()

    _enum_dicts: ClassVar[Dict[str, Dict[str, int]]] = {}
    _field_units: ClassVar[Dict[str, str]] = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically merge docstrings from parent classes.

        This hook is called whenever a class inherits from DataBase. It
        automatically merges the Attributes sections from all parent class
        docstrings into the child class docstring.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to super().__init_subclass__
        """
        super().__init_subclass__(**kwargs)
        merge_ancestor_docstrings(cls)

    def __post_init__(self) -> None:
        """Immediately called after building the class attributes.

        Provides basic functionalities:
        - Caches attribute lists on first instance (once per class)
        - Casts strings when they are provided as binary objects, which is the
          format one gets when loading string from HDF5 files.
        - Casts 8-bit unsigned integers to booleans when they are provided as
          numpy arrays, which is the format one gets when loading booleans from
          HDF5 files.
        """
        # Ensure attribute lists are cached (happens once per class)
        type(self)._ensure_cached_attrs()

        # Cast arrays to the correct type, check length
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type == np.ndarray:
                meta = field.metadata
                # Check that the length of the array matches the expected length
                if isinstance(meta, FieldMetadata) and meta.length is not None:
                    if len(value) != meta.length:
                        raise ValueError(
                            f"The `{field.name}` attribute of `{self.__class__.__name__}` "
                            f"objects must have length {meta.length}."
                        )

                # Cast the array to the correct type
                dtype = meta.dtype if isinstance(meta, FieldMetadata) else None
                setattr(self, field.name, np.asarray(value, dtype=dtype))

        # Cast stored binary strings back to regular strings (from HDF5)
        for attr in self._str_attrs:
            val = getattr(self, attr)
            if isinstance(val, bytes):
                setattr(self, attr, val.decode())

        # Cast stored 8-bit unsigned integers back to booleans (from HDF5)
        for attr in self._bool_attrs:
            val = getattr(self, attr)
            if isinstance(val, np.ndarray) and val.dtype == np.uint8:
                setattr(self, attr, bool(val.item()))

    def __eq__(self, other: object) -> bool:
        """Checks that all attributes of two class instances are the same.

        This overloads the default dataclass `__eq__` method to include an
        appopriate check for vector (numpy) attributes.

        Parameters
        ----------
        other : object
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
        for field in fields(self):
            key = field.name
            value = getattr(self, key)
            value_other = getattr(other, key)
            if field.type in (int, float, str, bool):
                # For scalars, handle NaN specially for floats
                if field.type == float:
                    # Both NaN -> equal; otherwise use regular comparison
                    both_nan = np.isnan(value) and np.isnan(value_other)
                    if not both_nan and value_other != value:
                        return False
                else:
                    # For int, str, bool, regular comparison
                    if value_other != value:
                        return False

            elif field.type == np.ndarray:
                # For numpy vectors, use array_equal with equal_nan=True
                if not np.array_equal(value, value_other, equal_nan=True):
                    return False

            elif field.type == list:
                # For object lists, compare the length and all elements individually
                if len(value) != len(value_other) or any(
                    v1 != v2 for v1, v2 in zip(value, value_other)
                ):
                    return False

            else:
                raise TypeError(
                    f"Cannot compare the `{key}` attribute of "
                    f"`{self.__class__.__name__}` objects. Unsupported type: "
                    f"{field.type}"
                )

        return True

    def set_precision(self, precision: int = 4) -> None:
        """Casts all the vector attributes to a different precision.

        Parameters
        ----------
        precision : int, default 4
            Precision in number of bytes (half=2, single=4, double=8)
        """
        if precision not in (2, 4, 8):
            raise ValueError(
                "Set the vector attribute precision for this object. "
                "Supported precisions are: 2 (half), 4 (single), and 8 (double)."
            )

        for field in fields(self):
            meta = field.metadata
            if field.type == np.ndarray and isinstance(meta, FieldMetadata):
                if meta.dtype is not None and "float" in str(meta.dtype):
                    val = getattr(self, field.name)
                    dtype = f"{val.dtype.str[:-1]}{precision}"
                    setattr(self, field.name, val.astype(dtype))

    def shift_indexes(self, shifts: Union[int, Dict[str, int]]) -> None:
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
            shift = shifts if not isinstance(shifts, dict) else shifts[attr]
            if isinstance(value, int) and value > -1:
                setattr(self, attr, value + shift)
            elif isinstance(value, np.ndarray):
                setattr(self, attr, value + shift)

    def as_dict(self, lite: bool = False) -> Dict[str, Any]:
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

        # Store all fields
        return_dict = {}
        for field in fields(self):
            if field.name not in skip_attrs:
                value = getattr(self, field.name)
                return_dict[field.name] = value

        return return_dict

    def scalar_dict(
        self,
        attrs: Optional[List[str]] = None,
        lengths: Optional[Dict[str, int]] = None,
        lite: bool = False,
    ) -> Dict[str, Union[float, int, str, bool]]:
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
                    scalar_dict[f"{attr}_{self._axes[i]}"] = v.item()

            elif attr in self._fixed_length_attrs:
                # If the attribute is a fixed-length array, expand with index
                for i, v in enumerate(value):
                    scalar_dict[f"{attr}_{i}"] = v.item()

            elif attr in self._var_length_attrs:
                if attr in lengths:
                    # If the attribute is a variable-length array with a length
                    # provided, resize it to match that length and store it
                    for i in range(lengths[attr]):
                        if i < len(value):
                            scalar_dict[f"{attr}_{i}"] = value[i].item()
                        else:
                            scalar_dict[f"{attr}_{i}"] = None

                else:
                    # If the attribute is a variable-length array of
                    # indeterminate length, cannot store it as scalars
                    if attrs is not None and attr in attrs:
                        raise ValueError(
                            f"Cannot cast the `{attr}` attribute of "
                            f"`{self.__class__.__name__}` to scalars. To cast a "
                            "variable-length array, must provide a fixed length."
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
    def enum_dicts(self) -> Dict[str, Dict[str, int]]:
        """Fetches the dictionary of enumerated attributes and their enumerator descriptors.

        Returns
        -------
        Dict[str, Dict[str, int]]
            Dictionary which maps names onto enumerator descriptors
        """
        return self._enum_dicts

    @property
    def field_units(self) -> Dict[str, str]:
        """Fetches the documented units for each field.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping field names to their units
        """
        # Resolve 'default' to current instance units
        return self._field_units

    @classmethod
    def from_hdf5(cls, cls_dict: Dict[str, Any]) -> Self:
        """Builds and returns an object of the class from a dictionary of attributes.

        This is used to build objects from HDF5 files, which are stored as
        dictionaries of attributes. This ensures that the attributes that are
        derived but stored to file are not loaded back to the object.

        Parameters
        ----------
        cls_dict : dict
            Dictionary of attributes to initialize the object with

        Returns
        -------
        Self
            Object of the class initialized with the provided attributes
        """
        # Ensure caching is done (needed for derived_names lookup)
        cls._ensure_cached_attrs()

        # Remove keys that are derived attributes and should not be loaded from file
        return cls(
            **{
                key: value
                for key, value in cls_dict.items()
                if key not in cls._derived_attrs
            }
        )

    @classmethod
    def _ensure_cached_attrs(cls) -> None:
        """Compute and cache attribute lists for this class if not already done.

        This runs once per class, on first instance creation.
        """
        # Check if already cached
        if cls._attrs_cached:
            return

        # Get fields for this class
        cls_fields = fields(cls)

        # Introspect derived properties once
        derived_props = cls._get_derived_properties()

        # Compute the fixed length arrays
        field_fixed = tuple(
            f.name
            for f in cls_fields
            if f.type == np.ndarray
            and isinstance(f.metadata, FieldMetadata)
            and f.metadata.length is not None
        )
        derived_fixed = tuple(
            k for k, v in derived_props.items() if v.length is not None
        )

        cls._fixed_length_attrs = field_fixed + derived_fixed

        # Compute variable length arrays
        field_var = tuple(
            f.name
            for f in cls_fields
            if f.type == np.ndarray
            and isinstance(f.metadata, FieldMetadata)
            and f.metadata.length is None
        )
        derived_var = tuple(k for k, v in derived_props.items() if v.length is None)

        cls._var_length_attrs = field_var + derived_var

        # Compute position attributes
        field_pos = tuple(
            f.name
            for f in cls_fields
            if isinstance(f.metadata, FieldMetadata)
            and f.metadata.category == "position"
        )
        derived_pos = tuple(
            k for k, v in derived_props.items() if v.category == "position"
        )
        cls._pos_attrs = field_pos + derived_pos

        # Compute vector attributes
        field_vec = tuple(
            f.name
            for f in cls_fields
            if isinstance(f.metadata, FieldMetadata) and f.metadata.category == "vector"
        )
        derived_vec = tuple(
            k for k, v in derived_props.items() if v.category == "vector"
        )

        cls._vec_attrs = field_vec + derived_vec

        # Compute index attributes (no derived version since indexes should not be derived)
        cls._index_attrs = tuple(
            f.name
            for f in cls_fields
            if isinstance(f.metadata, FieldMetadata) and f.metadata.index
        )

        # Compute skip attributes (no derived version)
        cls._skip_attrs = tuple(
            f.name
            for f in cls_fields
            if isinstance(f.metadata, FieldMetadata) and f.metadata.skip
        )
        cls._lite_skip_attrs = tuple(
            f.name
            for f in cls_fields
            if isinstance(f.metadata, FieldMetadata) and f.metadata.lite_skip
        )

        # Compute concatenation attributes (no derived version)
        cls._cat_attrs = tuple(
            f.name
            for f in cls_fields
            if isinstance(f.metadata, FieldMetadata) and f.metadata.cat
        )

        # Compute type-based attributes (no derived version)
        cls._str_attrs = tuple(f.name for f in cls_fields if f.type == str)
        cls._bool_attrs = tuple(f.name for f in cls_fields if f.type == bool)

        # Cache derived property names
        cls._derived_attrs = tuple(derived_props.keys())

        # Compute enumerated attributes dictionary
        cls._enum_dicts = {}
        for f in cls_fields:
            if isinstance(f.metadata, FieldMetadata):
                meta = cast(FieldMetadata, f.metadata)
                if meta.enum is not None:
                    cls._enum_dicts[f.name] = {v: k for k, v in meta.enum.items()}

        # Cache global units attributes (no field-based version)
        cls._global_units_attrs = tuple(
            k for k, v in derived_props.items() if v.units == "instance"
        )

        # Compute field units dictionary from field metadata and derived properties
        field_units = {
            f.name: f.metadata.units
            for f in cls_fields
            if isinstance(f.metadata, FieldMetadata) and f.metadata.units is not None
        }
        derived_units = {
            k: v.units for k, v in derived_props.items() if v.units is not None
        }
        cls._field_units = {**field_units, **derived_units}

        # Mark as cached
        cls._attrs_cached = True

    @classmethod
    def _get_derived_properties(cls) -> Dict[str, FieldMetadata]:
        """Introspect the class to find all derived_property descriptors.

        Returns
        -------
        Dict[str, FieldMetadata]
            Dictionary mapping property names to their FieldMetadata
        """
        result = {}
        for name in dir(cls):
            try:
                attr = getattr(cls, name)
                # Check if it's a property with derived metadata on the getter
                if isinstance(attr, property) and attr.fget is not None:
                    metadata = getattr(attr.fget, "__derived_property_metadata__", None)
                    if metadata is not None:
                        result[name] = metadata
                    else:
                        # Check if it's an alias property
                        target_name = getattr(
                            attr.fget, "__alias_property_target__", None
                        )
                        if target_name is not None:
                            # Look up the target and copy its metadata
                            # Aliases should NEVER be stored (skip=True)
                            # First check if target is a derived property
                            target_attr = getattr(cls, target_name, None)
                            if (
                                isinstance(target_attr, property)
                                and target_attr.fget is not None
                            ):
                                target_metadata = getattr(
                                    target_attr.fget,
                                    "__derived_property_metadata__",
                                    None,
                                )
                                if target_metadata is not None:
                                    # Copy metadata but force skip=True
                                    meta_dict = target_metadata.as_dict()
                                    meta_dict["skip"] = True
                                    result[name] = FieldMetadata(**meta_dict)
                            else:
                                # Target might be a regular dataclass field
                                try:
                                    cls_fields = fields(cls)
                                    for f in cls_fields:
                                        if f.name == target_name:
                                            # Check if metadata looks like FieldMetadata
                                            # (dataclasses wraps it in mappingproxy)
                                            meta = f.metadata
                                            if "units" in meta or "dtype" in meta:
                                                # It's a FieldMetadata (wrapped in mappingproxy)
                                                # Copy metadata but force skip=True
                                                meta_dict = dict(meta)
                                                meta_dict["skip"] = True
                                                result[name] = FieldMetadata(
                                                    **meta_dict
                                                )
                                                break
                                except TypeError:
                                    # Not a dataclass
                                    pass
            except AttributeError:
                # Some descriptors may raise AttributeError when accessed on class
                continue

        return result


@dataclass(eq=False)
class PosDataBase(DataBase):
    """Base class of for data structures with positional attributes.

    Includes method to convert positional attributes.

    Attributes
    ----------
    units : str
        Units in which the position attributes are expressed
    """

    units: str = "cm"

    def __post_init__(self):
        """Immediately called after building the class attributes.

        Makes sure the units are not binary and that they are recognized.
        """
        # Call the main post initialization function
        super().__post_init__()

        if self.units not in ("cm", "px"):
            raise ValueError(
                "Units of the positional attributes must be either `cm` or `px`."
            )

    @property
    def field_units(self) -> Dict[str, str]:
        """Fetches the documented units for each field.

        This property provides a dictionary mapping field names to their units,
        dynamically resolving 'instance' units to the current coordinate system.

        Unit Types:
        -----------
        - **Fixed units** (e.g., 'MeV', 'ns'): Physical quantities independent
          of coordinate representation. These remain constant.
        - **Instance units** ('instance' in metadata): Spatial quantities that
          follow self.units ('cm' or 'px'). These change with to_cm()/to_px().

        For the complete units metadata convention, see the module docstring.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping field names to their units. Fields with
            units='instance' in metadata are resolved to the current value
            of self.units.

        Examples
        --------
        >>> obj.units
        'cm'
        >>> obj.field_units['position']  # has units='instance' in metadata
        'cm'
        >>> obj.to_px(meta)
        >>> obj.field_units['position']
        'px'
        >>> obj.field_units['energy']  # has units='MeV' in metadata
        'MeV'
        """
        # Resolve 'instance' to current instance units
        return {
            name: self.units if unit == "instance" else unit
            for name, unit in self._field_units.items()
        }

    def to_cm(self, meta: "Meta") -> None:
        """Converts the coordinates of the positional attributes to cm in place.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        if self.units == "cm":
            raise ValueError("Units already expressed in centimeters")

        self.units = "cm"
        for attr in self._global_units_attrs:
            if attr in self._pos_attrs:
                setattr(self, attr, meta.to_cm(getattr(self, attr)))
            elif attr in self._vec_attrs:
                setattr(self, attr, getattr(self, attr) * meta.size)
            else:
                setattr(self, attr, getattr(self, attr) * meta.size[0])

    def to_px(self, meta: "Meta") -> None:
        """Converts the coordinates of the positional attributes to pixel in place.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        if self.units == "px":
            raise ValueError("Units already expressed in pixels")

        self.units = "px"
        for attr in self._global_units_attrs:
            if attr in self._pos_attrs:
                setattr(self, attr, meta.to_px(getattr(self, attr)))
            elif attr in self._vec_attrs:
                setattr(self, attr, getattr(self, attr) / meta.size)
            else:
                setattr(self, attr, getattr(self, attr) / meta.size[0])
