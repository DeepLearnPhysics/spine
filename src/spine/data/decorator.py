"""Decorators for properties in data structures.

This module provides decorators for properties that need metadata for
serialization and introspection:
- @derived_property: Computed properties with metadata
- @alias_property: Aliases that inherit metadata from their targets
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

__all__ = ["derived_property", "alias_property"]

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


def alias_property(target_name: str) -> Callable[[Callable[..., Any]], property]:
    """Decorator for property aliases that inherit metadata from their targets.

    This decorator creates a property alias that automatically inherits the
    metadata from the target property during introspection. The alias appears
    as a normal Python property to type checkers and linters.

    Metadata inheritance happens at introspection time by looking up the target
    property's metadata, ensuring aliases always stay in sync with their targets.

    Parameters
    ----------
    target_name : str
        Name of the target property to alias

    Returns
    -------
    Callable
        Decorator that returns a standard Python property with alias information

    Examples
    --------
    >>> @derived_property(units='MeV')
    ... def ke(self) -> float:
    ...     '''Kinetic energy in MeV.'''
    ...     return self._ke
    >>>
    >>> @alias_property('ke')
    ... def reco_ke(self) -> float:
    ...     '''Alias for ke to match nomenclature in truth.'''
    ...     return self.ke

    Notes
    -----
    The alias function should typically just return the target property's value.
    The decorator will mark the function so that introspection can find and
    copy metadata from the target.
    """

    def decorator(func: Callable[..., Any]) -> property:
        # Mark the function with the alias target name
        func.__alias_property_target__ = target_name  # type: ignore[attr-defined]
        return property(func)

    return decorator
