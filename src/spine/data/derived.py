"""Decorator for marking derived properties in data structures.

This module provides the @derived_property decorator for methods that compute
values dynamically but should be treated like fields for serialization and
introspection purposes.
"""

from typing import (
    Any,
    Callable,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from .field import FieldMetadata

__all__ = ["derived_property"]

T = TypeVar("T")
R = TypeVar("R")


def _decorate_derived_getter(
    func: Callable[..., Any], **metadata: Any
) -> Callable[..., Any]:
    """Attach derived-property metadata to a getter function.

    Parameters
    ----------
    func : Callable
        The getter function to decorate
    **metadata : Any
        Metadata keyword arguments passed to FieldMetadata

    Returns
    -------
    Callable
        The same function with metadata attached

    Raises
    ------
    TypeError
        If the function lacks a return type annotation or type hints cannot be resolved
    """
    # Get type hints, which resolves forward references
    try:
        hints = get_type_hints(func)
    except (NameError, AttributeError, TypeError) as e:
        raise TypeError(
            f"Could not resolve type hints for derived property "
            f"'{func.__name__}': {e}"
        ) from e

    # Check if return annotation exists
    if "return" not in hints:
        raise TypeError(
            f"Derived property '{func.__name__}' must have a return type annotation"
        )

    return_type = hints["return"]

    # Handle common type constructs
    origin = get_origin(return_type)
    if origin is Union:
        # For Optional[X] or Union[X, Y], take the first non-None type
        args = get_args(return_type)
        for arg in args:
            if arg is not type(None):
                return_type = arg
                break
        else:
            raise TypeError(
                f"Derived property '{func.__name__}' cannot have None as the only return type"
            )

    # Create FieldMetadata with introspected return type
    metadata_obj = FieldMetadata(return_type=return_type, **metadata)

    # Attach metadata to the function
    func.__derived_property_metadata__ = metadata_obj  # type: ignore[attr-defined]
    return func


@overload
def derived_property(func: Callable[..., Any], /) -> property:
    """Use as @derived_property without metadata."""


@overload
def derived_property(**metadata: Any) -> Callable[[Callable[..., Any]], property]:
    """Use as @derived_property(...) with metadata."""


def derived_property(
    func: Optional[Callable[..., Any]] = None, /, **metadata: Any
) -> property | Callable[[Callable[..., Any]], property]:
    """Decorator for marking derived properties in data structures.

    This decorator behaves like :func:`property` but attaches metadata to the
    getter function for serialization and introspection. Derived properties are
    discovered via introspection, with no manual registration required.

    The decorated property appears as a normal Python property to type checkers
    and linters, avoiding issues with custom descriptors.

    **Requirements:**
    - The decorated function must have a return type annotation
    - The return type is automatically introspected and stored in metadata

    Parameters
    ----------
    func : Callable, optional
        The getter function (when used without parentheses)
    **metadata : Any
        Metadata keyword arguments. See :class:`FieldMetadata` for supported keys
        including:

        - category : str
            Semantic category (e.g. ``'position'``, ``'vector'``)
        - units : str
            Physical units (fixed like ``'MeV'`` or semantic like ``'instance'``)
        - length : int
            Expected array length
        - dtype : type
            NumPy dtype for arrays

    Returns
    -------
    property or Callable
        A standard Python property object with metadata attached to its getter

    Raises
    ------
    TypeError
        If the decorated function lacks a return type annotation

    Examples
    --------
    >>> @derived_property
    ... def momentum_mag(self) -> float:
    ...     '''Total momentum magnitude in GeV/c.'''
    ...     return np.linalg.norm(self.momentum)

    >>> @derived_property(units='instance', category='position')
    ... def vertex(self) -> np.ndarray:
    ...     '''Interaction vertex position.'''
    ...     return self.particles[0].start_point
    """
    if func is not None:
        # Used as @derived_property (without parentheses)
        fget = _decorate_derived_getter(func, **metadata)
        return property(fget)

    # Used as @derived_property(...) (with metadata)
    def wrapper(f: Callable[[T], R]) -> property:
        fget = _decorate_derived_getter(f, **metadata)
        return property(fget)

    return wrapper
