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
   ...         length=3, dtype=np.float32, position=True, units='instance'
   ...     )
   ... )
   >>> length: float = field(default=np.nan, metadata=FieldMetadata(units='instance'))

The field_units property dynamically resolves 'instance' to the current value
of the instance's units attribute, providing accurate units for all fields at
runtime.

Note: Direction vectors (unit vectors) typically have no units metadata since
they are unitless normalized directions.
"""

from dataclasses import dataclass, fields, replace
from enum import IntEnum
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np

from spine.utils.docstring import merge_ancestor_docstrings

from .field import FieldMetadata

if TYPE_CHECKING:  # pragma: no cover
    from spine.data.larcv.meta import Meta


@dataclass(eq=False, repr=False)
class DataBase:
    """Base class of all data structures.

    Defines basic methods shared by all data structures.
    """

    # Euclidean axis labels
    _axes: ClassVar[tuple[str, str, str]] = ("x", "y", "z")

    # Explicitly permitted runtime attributes which are not dataclass fields.
    # Subclasses should extend this only for intentional per-instance caches.
    _dynamic_attrs: ClassVar[frozenset[str]] = frozenset()

    # NOTE: Cached attribute lists are NOT declared here as ClassVar
    # to avoid sharing across inheritance hierarchy. Each subclass gets
    # its own independent copies via __init_subclass__.

    @staticmethod
    def _annotation_matches(annotation: Any, target: type) -> bool:
        """Check if an annotation is, or contains, a target runtime type."""
        if annotation is target:
            return True

        origin = get_origin(annotation)
        if origin is target:
            return True

        args = get_args(annotation)
        if origin is UnionType or type(None) in args:
            return any(DataBase._annotation_matches(arg, target) for arg in args)

        return False

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Automatically merge docstrings from parent classes and initialize
        class-level cached attribute lists.

        This hook is called whenever a class inherits from DataBase. It
        automatically merges the Attributes sections from all parent class
        docstrings into the child class docstring and initializes the
        class-level cached attribute lists to be filled on first instance creation.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to super().__init_subclass__
        """
        super().__init_subclass__(**kwargs)
        merge_ancestor_docstrings(cls)

        # Give each subclass its own independent cached attribute lists
        # This prevents inheritance from parent classes
        cls._attrs_cached = False
        cls._fields = ()
        cls._field_names = ()
        cls._field_name_set = frozenset()
        cls._field_types = {}
        cls._field_kinds = {}
        cls._field_meta = {}
        cls._prop_meta = {}
        cls._metadata = {}
        cls._array_specs = ()
        cls._array_spec_map = {}
        cls._float_array_attrs = ()
        cls._fixed_length_attrs = ()
        cls._var_length_attrs = ()
        cls._pos_attrs = ()
        cls._vec_attrs = ()
        cls._normed_vec_attrs = ()
        cls._index_attrs = ()
        cls._skip_attrs = ()
        cls._lite_skip_attrs = ()
        cls._cat_attrs = ()
        cls._str_attrs = ()
        cls._bool_attrs = ()
        cls._derived_attrs = ()
        cls._global_units_attrs = ()
        cls._enum_dicts = {}
        cls._enum_attrs = {}
        cls._enum_values = {}
        cls._field_units = {}

    def __post_init__(self) -> None:
        """Immediately called after building the class attributes.

        Provides basic functionalities:
        - Caches attribute lists on first instance (once per class)
        """
        # Ensure attribute lists are cached (happens once per class)
        type(self)._ensure_cached_attrs()

        # Cast arrays to the correct type and check their lengths. Field
        # reflection and metadata validation are cached once per data class.
        for name, length, dtype in type(self)._array_specs:
            value = getattr(self, name)
            if length is not None and len(value) != length:
                raise ValueError(
                    f"The `{name}` attribute of `{self.__class__.__name__}` "
                    f"objects must have length {length}."
                )

            object.__setattr__(self, name, np.asarray(value, dtype=dtype))

    def __setattr__(self, name: str, value: Any) -> None:
        """Validate attributes and preserve array dtypes after initialization.

        Dataclass construction is normalized in :meth:`__post_init__`. From
        that point onward, an array field keeps its current dtype, including
        precision changes made explicitly through :meth:`set_precision`.
        ``np.asarray`` is zero-copy when the incoming array already has the
        target dtype. Undeclared attributes are rejected unless a subclass
        explicitly lists them in :attr:`_dynamic_attrs`.
        """
        cls = type(self)
        if (
            getattr(cls, "_attrs_cached", False)
            and name not in cls._field_name_set
            and name not in cls._dynamic_attrs
        ):
            raise AttributeError(
                f"Cannot set undeclared attribute `{name}` on "
                f"`{self.__class__.__name__}`."
            )

        spec = getattr(cls, "_array_spec_map", {}).get(name)
        current = self.__dict__.get(name)
        if spec is not None and isinstance(current, np.ndarray):
            length, _ = spec
            value = np.asarray(value, dtype=current.dtype)
            if length is not None and len(value) != length:
                raise ValueError(
                    f"The `{name}` attribute of `{self.__class__.__name__}` "
                    f"objects must have length {length}."
                )

        object.__setattr__(self, name, value)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickled instances and rebuild per-class cached metadata.

        Objects transferred across multiprocessing worker boundaries are
        unpickled without running ``__post_init__`` again. Rebuild the cached
        class attribute lists in the receiving process so methods relying on
        them, such as unit conversion, still work.

        Parameters
        ----------
        state : dict
            Pickled instance state
        """
        self.__dict__.update(state)
        type(self)._ensure_cached_attrs()

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
        cls = type(self)
        for key in cls._field_names:
            value = getattr(self, key)
            value_other = getattr(other, key)
            field_type = cls._field_types[key]
            field_kind = cls._field_kinds[key]
            if field_kind in (int, float, str, bool):
                # For scalars, handle NaN specially for floats
                if field_kind == float:
                    # Both NaN -> equal; otherwise use regular comparison
                    both_nan = np.isnan(value) and np.isnan(value_other)
                    if not both_nan and value_other != value:
                        return False
                else:
                    # For int, str, bool, regular comparison
                    if value_other != value:
                        return False

            elif field_kind is np.ndarray:
                # For numpy vectors, use array_equal with equal_nan=True
                if not np.array_equal(value, value_other, equal_nan=True):
                    return False

            elif field_kind is list:
                # For object lists, compare the length and all elements individually
                if len(value) != len(value_other) or any(
                    v1 != v2 for v1, v2 in zip(value, value_other)
                ):
                    return False

            else:
                # For unsupported types, raise an error (could be extended to
                # support more types as needed)
                raise TypeError(
                    f"Cannot compare the `{key}` attribute of "
                    f"`{self.__class__.__name__}` objects. Unsupported type: "
                    f"{field_type}"
                )

        return True

    def __repr__(self) -> str:
        """Compact representation of the data object.

        Numpy arrays and lists are summarized rather than expanded, which keeps
        interactive inspection readable for large event objects.
        """
        parts = []
        for name in type(self)._field_names:
            value = getattr(self, name)
            parts.append(f"{name}={self._repr_value(value)}")

        return f"{self.__class__.__name__}({', '.join(parts)})"

    @staticmethod
    def _repr_value(value: object) -> str:
        """Compactly represent a field value."""
        if isinstance(value, np.ndarray):
            return f"array(shape={value.shape}, dtype={value.dtype})"

        if isinstance(value, list):
            return f"list(len={len(value)})"

        return repr(value)

    def set_precision(self, precision: int = 4) -> None:
        """Casts all the vector attributes to a different precision.

        Parameters
        ----------
        precision : int, default 4
            Precision in number of bytes (half=2, single=4, double=8)
        """
        if precision not in (2, 4, 8):
            raise ValueError(
                "Precision must be one of: 2 (half), 4 (single), or 8 (double)."
            )

        for attr in type(self)._float_array_attrs:
            val = getattr(self, attr)
            dtype = f"{val.dtype.str[:-1]}{precision}"
            object.__setattr__(self, attr, val.astype(dtype))

    def shift_indexes(self, shifts: int | dict[str, int]) -> None:
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

    def as_dict(
        self, lite: bool = False, include_derived: bool = True
    ) -> dict[str, Any]:
        """Returns the data class as dictionary of (key, value) pairs.

        Parameters
        ----------
        lite : bool, default False
            If `True`, the `_lite_skip_attrs` are dropped
        include_derived : bool, default True
            If `True`, include computed properties marked with `@stored_property`.

        Returns
        -------
        dict
            Dictionary of attribute names and their values
        """
        # Build a list of attributes to skip
        skip_attrs = self._skip_attrs if not lite else self._lite_skip_attrs

        # Store all fields
        return_dict = {}
        for name in type(self)._field_names:
            if name not in skip_attrs:
                return_dict[name] = getattr(self, name)

        # Store computed properties explicitly marked for serialization. These
        # are not dataclass fields, so they are not included above.
        if include_derived:
            for attr in self._derived_attrs:
                if attr not in skip_attrs:
                    return_dict[attr] = getattr(self, attr)

        return return_dict

    @classmethod
    def attr_names(
        cls,
        include_derived: bool = True,
        include_skipped: bool = True,
        lite: bool = False,
    ) -> tuple[str, ...]:
        """Return the names of valid attributes on this data class.

        Parameters
        ----------
        include_derived : bool, default True
            If `True`, include computed properties marked with
            `@stored_property` or `@stored_alias`.
        include_skipped : bool, default True
            If `True`, include attributes which are skipped by serialization.
        lite : bool, default False
            If `True` and `include_skipped` is `False`, apply the lite skip
            policy used by :meth:`as_dict`.

        Returns
        -------
        tuple[str, ...]
            Tuple of valid attribute names.
        """
        cls._ensure_cached_attrs()

        attrs = list(cls._field_names)
        if include_derived:
            attrs.extend(cls._derived_attrs)

        if not include_skipped:
            skip_attrs = cls._skip_attrs if not lite else cls._lite_skip_attrs
            attrs = [attr for attr in attrs if attr not in skip_attrs]

        return tuple(attrs)

    def scalar_dict(
        self,
        attrs: list[str] | None = None,
        lengths: dict[str, int] | None = None,
        lite: bool = False,
    ) -> dict[str, float | int | str | bool]:
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

    def value_with_units(self, attr: str) -> tuple[Any, str | None]:
        """Fetch an attribute value with its documented units.

        Parameters
        ----------
        attr : str
            Name of the attribute or stored property to fetch

        Returns
        -------
        tuple
            Attribute value and unit string. If no units are documented for
            the attribute, the unit is `None`.
        """
        if not hasattr(self, attr):
            raise AttributeError(
                f"Attribute `{attr}` does not appear in {self.__class__.__name__}."
            )

        return getattr(self, attr), self.field_units.get(attr)

    @property
    def index_attrs(self) -> tuple[str, ...]:
        """Return the tuple of index-bearing attributes.

        Returns
        -------
        tuple[str, ...]
            Names of attributes that should be shifted during batching or
            overlay operations.
        """
        return self._index_attrs

    @property
    def enum_dicts(self) -> dict[str, dict[str, int]]:
        """Fetches the dictionary of enumerated attributes and their enumerator descriptors.

        Returns
        -------
        Dict[str, Dict[str, int]]
            Dictionary which maps names onto enumerator descriptors
        """
        return self._enum_dicts

    @property
    def enum_attrs(self) -> dict[str, dict[str, int]]:
        """Fetches HDF5-friendly enumerator descriptors for enum-backed fields."""
        return self._enum_attrs

    @property
    def enum_values(self) -> dict[str, dict[int, str]]:
        """Fetches reverse enum lookups from stored values to symbolic names."""
        return self._enum_values

    @property
    def field_units(self) -> dict[str, str]:
        """Fetches the documented units for each field.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping field names to their units
        """
        # Resolve 'default' to current instance units
        return self._field_units

    @classmethod
    def from_dict(cls, cls_dict: dict[str, Any]) -> "DataBase":
        """Builds and returns an object of the class from a dictionary of attributes.

        This is used to build objects from serialized dictionaries of
        attributes. This ensures that the attributes that are derived but stored
        to file are not loaded back to the object.

        Provides basic functionalities:
        - Casts strings when they are provided as binary objects, which is the
          format one gets when loading string from HDF5 files.
        - Casts 8-bit unsigned integers to booleans when they are provided as
          numpy arrays, which is the format one gets when loading booleans from
          HDF5 files.

        Parameters
        ----------
        cls_dict : dict
            Dictionary of attributes to initialize the object with

        Returns
        -------
        DataBase
            Object of the class initialized with the provided attributes
        """
        # Ensure caching is done (needed for derived_names lookup)
        cls._ensure_cached_attrs()

        # Cast stored binary strings back to regular strings (from HDF5)
        for attr in cls._str_attrs:
            if isinstance(cls_dict.get(attr), bytes):
                cls_dict[attr] = cls_dict[attr].decode()

        # Cast stored 8-bit unsigned integers back to booleans (from HDF5)
        for attr in cls._bool_attrs:
            val = cls_dict.get(attr)
            if isinstance(val, np.ndarray) and val.dtype == np.uint8:
                cls_dict[attr] = bool(val.item())
            elif isinstance(val, np.generic) and val.dtype == np.uint8:
                cls_dict[attr] = bool(val.item())

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
        # Check if already cached, if so, do nothing
        if cls._attrs_cached:
            return

        # Get the list of fields defined on the class
        cls_fields = fields(cls)
        field_types = get_type_hints(cls)
        field_meta = {f.name: FieldMetadata(**f.metadata) for f in cls_fields}

        # Cache all immutable dataclass reflection products. These are used by
        # instance construction, equality, representation and serialization.
        cls._fields = cls_fields
        cls._field_names = tuple(f.name for f in cls_fields)
        cls._field_name_set = frozenset(cls._field_names)
        cls._field_types = {f.name: field_types.get(f.name, f.type) for f in cls_fields}
        cls._field_kinds = {}
        for name, field_type in cls._field_types.items():
            if field_type in (int, float, str, bool):
                kind = field_type
            elif cls._annotation_matches(field_type, np.ndarray):
                kind = np.ndarray
            elif cls._annotation_matches(field_type, list):
                kind = list
            else:
                kind = None
            cls._field_kinds[name] = kind
        cls._field_meta = field_meta
        cls._array_specs = tuple(
            (f.name, field_meta[f.name].length, field_meta[f.name].dtype)
            for f in cls_fields
            if cls._annotation_matches(cls._field_types[f.name], np.ndarray)
        )
        cls._array_spec_map = {
            name: (length, dtype) for name, length, dtype in cls._array_specs
        }
        cls._float_array_attrs = tuple(
            name
            for name, _, dtype in cls._array_specs
            if dtype is not None and "float" in str(dtype)
        )

        # Cache type-based attributes (only used to load from HDF5, not for general use)
        cls._str_attrs = tuple(
            f.name for f in cls_fields if field_types.get(f.name, f.type) == str
        )
        cls._bool_attrs = tuple(
            f.name for f in cls_fields if field_types.get(f.name, f.type) == bool
        )

        # Get field and derived property metadata, combine them
        prop_meta = cls._get_stored_properties(field_meta)
        meta = {**field_meta, **prop_meta}
        cls._prop_meta = prop_meta
        cls._metadata = meta

        # Cache the list of derived properties to not load from file
        cls._derived_attrs = tuple(prop_meta.keys())

        # Cache the fixed and variable length arrays
        cls._fixed_length_attrs = tuple(
            k for k, v in meta.items() if v.dtype is not None and v.length is not None
        )
        cls._var_length_attrs = tuple(
            k for k, v in meta.items() if v.dtype is not None and v.length is None
        )

        # Cache specific types of arrays
        cls._pos_attrs = tuple(k for k, v in meta.items() if v.position)
        cls._vec_attrs = tuple(k for k, v in meta.items() if v.vector)
        cls._normed_vec_attrs = tuple(
            k for k in cls._vec_attrs if meta[k].units == "instance"
        )

        # Cache index attributes (categorical attributes with an enum are not considered index)
        cls._index_attrs = tuple(k for k, v in meta.items() if v.index)

        # Cache attributes to concatenate when merging objects
        cls._cat_attrs = tuple(k for k, v in field_meta.items() if v.cat)

        # Cache attributes not to store to file and not to include in lite dicts
        cls._skip_attrs = tuple(k for k, v in meta.items() if v.skip)
        cls._lite_skip_attrs = (
            *cls._skip_attrs,  # All the skip attributes are also lite_skip
            *(k for k, v in meta.items() if v.lite_skip),
        )

        # Cache enumerated attributes dictionary (enumerated attributes are not deri
        cls._enum_dicts = {}
        cls._enum_attrs = {}
        cls._enum_values = {}
        for k, v in meta.items():
            if v.enum is not None:
                assert isinstance(v.enum, type) and issubclass(v.enum, IntEnum)
                cls._enum_dicts[k] = {
                    member.name.lower(): member.value for member in v.enum
                }
                cls._enum_attrs[k] = {member.name: member.value for member in v.enum}
                cls._enum_values[k] = {member.value: member.name for member in v.enum}

        # Cache field units dictionary from field metadata and derived properties
        cls._field_units = {k: v.units for k, v in meta.items() if v.units is not None}

        # Cache global units attributes (specified as 'instance' in metadata)
        cls._global_units_attrs = tuple(
            k for k, v in cls._field_units.items() if v == "instance"
        )

        # Mark as cached
        cls._attrs_cached = True

    @classmethod
    def _get_stored_properties(
        cls, field_meta: dict[str, FieldMetadata] | None = None
    ) -> dict[str, FieldMetadata]:
        """Introspect the class to find all stored_property descriptors and
        any alias properties, and return a dictionary mapping their names to their
        FieldMetadata.

        This walks the entire MRO to find properties defined on parent classes.

        Returns
        -------
        Dict[str, FieldMetadata]
            Dictionary mapping property names to their FieldMetadata
        """
        if field_meta is None and cls._attrs_cached:
            return dict(cls._prop_meta)

        result, aliases = {}, {}

        # Walk through MRO to get all stored properties and aliases from parent
        # classes too. Child definitions win over parent definitions.
        for klass in cls.__mro__:
            # Skip object base class
            if klass is object:
                continue

            # Use __dict__ to only get attributes defined directly on this class
            for name, attr in klass.__dict__.items():
                # Skip if already processed from a child class (child overrides win)
                if name in result or name in aliases:
                    continue

                # Check if it's a property with stored metadata
                if isinstance(attr, property) and attr.fget is not None:
                    metadata = getattr(attr.fget, "__stored_property_metadata__", None)
                    if metadata is not None:
                        result[name] = metadata
                    else:
                        # Check if it's an alias property
                        target_name = getattr(
                            attr.fget, "__stored_alias_target__", None
                        )
                        if target_name is not None:
                            aliases[name] = target_name

        # Resolve aliases once all stored properties have been discovered.
        if field_meta is None:
            field_meta = {f.name: FieldMetadata(**f.metadata) for f in fields(cls)}
        for name, target_name in aliases.items():
            if target_name in result:
                metadata = result[target_name]
            elif target_name in field_meta:
                metadata = field_meta[target_name]
            else:
                raise AttributeError(
                    f"Stored alias `{name}` on `{cls.__name__}` targets "
                    f"unknown attribute `{target_name}`."
                )

            # Aliases are not stored as independent fields.
            result[name] = replace(metadata, skip=True)

        return result


@dataclass(eq=False, repr=False)
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
    def field_units(self) -> dict[str, str]:
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
            elif attr in self._normed_vec_attrs:
                setattr(self, attr, getattr(self, attr) * meta.size)
            else:
                # Imperfect: assumes cubic pixels
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
            elif attr in self._normed_vec_attrs:
                setattr(self, attr, getattr(self, attr) / meta.size)
            else:
                # Imperfect: assumes cubic pixels
                setattr(self, attr, getattr(self, attr) / meta.size[0])
