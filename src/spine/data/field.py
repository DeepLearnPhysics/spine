"""Metadata descriptor for data structure fields.

This module provides a type-safe alternative to dictionary-based field metadata,
offering better IDE support, type checking, and documentation.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import IntEnum

__all__ = ["FieldMetadata"]


@dataclass(frozen=True)
class FieldMetadata(Mapping[str, object]):
    """Metadata for data structure fields.

    Provides a type-safe way to specify field metadata with validation and
    clear documentation of available options. Implements the Mapping protocol
    for compatibility with Python's dataclass field() function.

    Parameters
    ----------
    return_type : type, optional
        The expected type of the field value (e.g., int, float, np.ndarray).
    length : int, optional
        Expected length for fixed-size numpy arrays. If specified, arrays
        must have exactly this length.
    dtype : type, optional
        NumPy dtype for array fields. Used to cast arrays to the correct type.
    position : bool, default=False
        Whether this field represents a spatial position coordinate.
    vector : bool, default=False
        Whether this field represents a non-unitary vector quantity.
    index : bool, default=False
        Whether this field is an index (e.g., ID, parent_id).
    skip : bool, default=False
        Whether to skip this field in standard serialization.
    lite_skip : bool, default=False
        Whether to skip this field in lightweight serialization.
    cat : bool, default=False
        Whether this is a concatenation attribute.
    units : str, optional
        Physical units for the field. Common values:
        - Fixed units: ``'MeV'``, ``'GeV'``, ``'ns'``, ``'us'``, ``'MeV/c'``
        - Instance units: ``'instance'`` (follows the instance's units attribute)
    enum : IntEnum subclass, optional
        Enumerated type for categorical fields (e.g., ``ParticlePID``).

    >>> id: int = field(default=-1, metadata=FieldMetadata(index=True))
    >>>
    >>> # Enumerated field
    >>> particle_type: int = field(
    ...     default=-1,
    ...     metadata=FieldMetadata(enum=ParticlePID)
    ... )
    """

    return_type: type | None = None
    length: int | None = None
    dtype: type | None = None
    position: bool = False
    vector: bool = False
    units: str | None = None
    enum: type[IntEnum] | None = None
    index: bool = False
    skip: bool = False
    lite_skip: bool = False
    cat: bool = False

    def __post_init__(self):
        """Validate field constraints."""
        if self.enum is not None and not (
            isinstance(self.enum, type) and issubclass(self.enum, IntEnum)
        ):
            raise TypeError("'enum' must be an IntEnum subclass")

    # Implement Mapping protocol for dataclass field() compatibility
    def __getitem__(self, key: str) -> object:
        """Get metadata value by key."""
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        """Iterate over metadata keys with non-None/non-False values."""
        for key in (
            "return_type",
            "length",
            "dtype",
            "position",
            "vector",
            "units",
            "enum",
            "index",
            "skip",
            "lite_skip",
            "cat",
        ):
            value = getattr(self, key)
            # Only include non-None and non-False values
            if value is not None and value is not False:
                yield key

    def __len__(self) -> int:
        """Return number of non-None/non-False metadata entries."""
        return sum(1 for _ in self)

    def as_dict(self) -> dict[str, object]:
        """Convert to plain dictionary.

        Returns
        -------
        dict
            Dictionary with non-None/non-False metadata values.
        """
        return {key: self[key] for key in self}
