"""Decorators for properties in data structures.

This module provides decorators for properties that need metadata for
serialization and introspection:
- @stored_property: Properties with metadata for serialization and introspection
- @stored_alias: Properties that are aliases of other stored properties
"""

from typing import (
    Any,
    Callable,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from .field import FieldMetadata

__all__ = ["stored_property", "stored_alias"]


@overload
def stored_property(func: Callable[..., Any], /) -> Callable[..., Any]:
    """Use as @stored_property."""


@overload
def stored_property(
    **metadata: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Use as @stored_property(...)."""


def stored_property(
    func: Optional[Callable[..., Any]] = None, /, **metadata: Any
) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a property getter for serialization/introspection.

    This decorator does not create the property itself. It attaches metadata to
    the getter function and is intended to be used together with :func:`property`:

    >>> @property
    ... @stored_property
    ... def size(self) -> int:
    ...     return len(self.index)

    >>> @property
    ... @stored_property(units='us')
    ... def time(self) -> float:
    ...     return self.ts1_ns / 1000.0

    Parameters
    ----------
    func : Callable, optional
        The getter function when used as ``@stored_property`` without
        parentheses.
    **metadata : Any
        Metadata keyword arguments passed to :class:`FieldMetadata`.

    Returns
    -------
    Callable
        Either the decorated getter function itself, or a decorator that
        attaches metadata to a getter function.

    Raises
    ------
    TypeError
        If the decorated function lacks a return type annotation or type hints
        cannot be resolved.
    """

    def wrapper(f: Callable[..., Any]) -> Callable[..., Any]:
        """Attach metadata to a property getter function.

        Parameters
        ----------
        f : Callable
            The getter function to decorate.

        Returns
        -------
        Callable
            The same function with metadata attached.

        Raises
        ------
        TypeError
            If the getter lacks a return type annotation or its type hints
            cannot be resolved.
        """
        try:
            hints = get_type_hints(f)
        except (NameError, AttributeError, TypeError) as e:
            raise TypeError(
                f"Could not resolve type hints for stored property "
                f"'{f.__name__}': {e}"
            ) from e

        if "return" not in hints:
            raise TypeError(
                f"Stored property '{f.__name__}' must have a return type annotation"
            )

        return_type = hints["return"]

        origin = get_origin(return_type)
        if origin is Union:
            args = get_args(return_type)
            for arg in args:
                if arg is not type(None):
                    return_type = arg
                    break
            else:
                raise TypeError(
                    f"Stored property '{f.__name__}' cannot have None as the only return type"
                )

        metadata_obj = FieldMetadata(return_type=return_type, **metadata)
        setattr(f, "__stored_property_metadata__", metadata_obj)

        return f

    if func is not None:
        return wrapper(func)

    return wrapper


def stored_alias(
    target_name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a property getter as an alias of another stored property.

    This decorator does not create the property itself. It attaches alias
    information to the getter function and is intended to be used together with
    :func:`property`:

    >>> @property
    ... @stored_alias('ke')
    ... def reco_ke(self) -> float:
    ...     return self.ke

    Parameters
    ----------
    target_name : str
        Name of the target property to alias.

    Returns
    -------
    Callable
        Decorator that attaches alias information to a getter function.
    """

    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "__stored_alias_target__", target_name)
        return func

    return wrapper
