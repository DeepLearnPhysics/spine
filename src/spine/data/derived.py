"""Decorator for marking derived properties in data structures.

This module provides the @derived_property decorator for methods that compute
values dynamically but should be treated like fields for serialization and
introspection purposes.
"""

from typing import Any, Callable, Optional, TypeVar, get_type_hints, overload

from .field import FieldMetadata

__all__ = ["DerivedProperty", "derived_property"]

T = TypeVar("T")
R = TypeVar("R")


class DerivedProperty(property):
    """Descriptor for derived properties that should be stored to HDF5.

    This decorator behaves like :class:`property` but stores metadata directly
    on the descriptor. Derived properties are discovered via introspection, with
    no manual registration required.

    **Requirements:**
    - The decorated function must have a return type annotation
    - The return type is automatically introspected and stored in metadata

    Parameters
    ----------
    fget : Callable
        Getter function used to compute the property.
    **metadata : Any
        Metadata keyword arguments. See :class:`FieldMetadata` for supported keys
        including:
        - type : str
            Semantic type (e.g. ``'position'``, ``'vector'``)
        - index : bool
            Whether this is an index attribute
        - units : str
            Physical units (fixed like ``'MeV'`` or semantic like ``'instance'``)
        - length : int
            Expected array length
        - dtype : type
            NumPy dtype for arrays

    Raises
    ------
    TypeError
        If the decorated function lacks a return type annotation
    """

    metadata: FieldMetadata
    name: Optional[str]
    __name__: str
    __qualname__: str
    __module__: str
    __annotations__: dict[str, Any]
    __wrapped__: Callable[..., Any]

    def __init__(self, fget: Callable[[T], R], **metadata: Any) -> None:
        """Initialize the derived property descriptor.

        Parameters
        ----------
        fget : Callable
            Function used to compute the property.
        **metadata : Any
            Metadata keyword arguments passed to FieldMetadata.
        """
        super().__init__(fget)

        # Introspect return type before creating FieldMetadata
        return_type = self._introspect_return_type(fget)

        # Convert keyword arguments to FieldMetadata, including return_type
        self.metadata = FieldMetadata(return_type=return_type, **metadata)
        self.name = None

        # Copy useful function metadata for introspection tools.
        self.__doc__ = fget.__doc__
        self.__name__ = fget.__name__
        self.__qualname__ = fget.__qualname__
        self.__module__ = fget.__module__
        self.__annotations__ = getattr(fget, "__annotations__", {})
        self.__wrapped__ = fget

    def __set_name__(self, owner: type, name: str) -> None:
        """Called when the descriptor is assigned to a class attribute."""
        self.name = name

    def getter(self, fget: Callable[[T], R]) -> "DerivedProperty":
        """Return a copy of this descriptor with a different getter."""
        # Convert metadata to dict for passing as kwargs
        metadata_dict = self.metadata.as_dict()
        # Remove return_type since it will be introspected again
        metadata_dict.pop("return_type", None)
        new_prop = type(self)(fget, **metadata_dict)
        new_prop.name = self.name
        return new_prop

    def _introspect_return_type(self, func: Callable[..., Any]) -> type:
        """Introspect and return the return type from function annotations.

        Parameters
        ----------
        func : Callable
            The function to introspect

        Returns
        -------
        type
            The return type of the function

        Raises
        ------
        TypeError
            If the function lacks a return type annotation or hints cannot be
            resolved
        """
        try:
            hints = get_type_hints(func)
            if "return" in hints:
                return hints["return"]
            else:
                raise TypeError(
                    f"Derived property '{func.__name__}' must have a return type annotation"
                )
        except (NameError, AttributeError, TypeError) as e:
            raise TypeError(
                f"Could not resolve type hints for derived property "
                f"'{func.__name__}': {e}"
            ) from e


@overload
def derived_property(func: Callable[..., Any], /) -> DerivedProperty:
    """Use as @derived_property without metadata."""


@overload
def derived_property(
    **metadata: Any,
) -> Callable[[Callable[..., Any]], DerivedProperty]:
    """Use as @derived_property(...) with metadata."""


def derived_property(
    func: Optional[Callable[..., Any]] = None,
    /,
    **metadata: Any,
) -> DerivedProperty | Callable[[Callable[..., Any]], DerivedProperty]:
    """Convenience decorator for defining derived properties.

    Can be used either with or without parentheses:

    >>> @derived_property
    ... def foo(self) -> int:
    ...     return 1

    >>> @derived_property(units='cm')
    ... def bar(self) -> float:
    ...     return 2.0
    """
    if func is not None:
        return DerivedProperty(func, **metadata)

    def wrapper(f: Callable[..., Any]) -> DerivedProperty:
        return DerivedProperty(f, **metadata)

    return wrapper
