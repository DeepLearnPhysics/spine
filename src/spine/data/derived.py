"""Decorator for marking derived properties in data structures.

This module provides the @derived_property decorator for methods that compute
values dynamically but should be treated like fields for serialization and
introspection purposes.
"""

import functools
from typing import Any, Callable, Dict, Optional, get_type_hints

__all__ = ["DerivedProperty", "derived_property"]


class DerivedProperty:
    """Decorator for derived properties that should be stored to HDF5.

    This decorator acts like @property but stores metadata directly on the descriptor.
    Properties are discovered via introspection, no manual registration needed.

    **Requirements:**
    - Must have a return type annotation
    - Return type is automatically introspected and stored in metadata

    Parameters
    ----------
    **metadata : dict
        Metadata about the derived property. Supported keys:
        - type : str ('position', 'vector') - Semantic type
        - index : bool - Whether this is an index attribute
        - units : str - Physical units (fixed like 'MeV' or 'instance')
        - length : int - Expected array length
        - dtype : type - NumPy dtype for arrays

    Examples
    --------
    Position with instance-dependent units:

    >>> @derived_property(type='position', units='instance')
    ... def vertex(self) -> np.ndarray:
    ...     return self.start_point

    Time with fixed units:

    >>> @derived_property(units='us')
    ... def time(self) -> float:
    ...     return self.ts1_ns / 1000

    For units convention details, see the spine.data.base module docstring.

    Raises
    ------
    TypeError
        If the decorated function lacks a return type annotation
    """

    def __init__(
        self, func: Optional[Callable] = None, **metadata: Dict[str, Any]
    ) -> None:
        """Initialize the derived property descriptor.

        Parameters
        ----------
        func : Callable, optional
            The function to compute the property (if used without parentheses)
        **metadata : dict
            Metadata about the derived property (e.g., type='position', index=True)
        """
        self.metadata = metadata
        self.name = None

        if func is not None:
            # Called as @derived_property (no parentheses)
            self.func = func
            self._introspect_return_type(func)
            # Copy attributes from wrapped function for proper introspection
            functools.update_wrapper(self, func, updated=[])
        else:
            # Called as @derived_property(...) (with metadata)
            self.func = None

    def __call__(self, func: Callable) -> "DerivedProperty":
        """Handle when called with metadata: @derived_property(type='position').

        Parameters
        ----------
        func : Callable
            The function to compute the property

        Returns
        -------
        _DerivedProperty
            The descriptor instance with the function set
        """
        if self.func is None:
            self.func = func
            self._introspect_return_type(func)
            # Copy attributes from wrapped function for proper introspection
            functools.update_wrapper(self, func, updated=[])
            return self
        # Otherwise, this is the property being accessed
        return self.__get__(None, None)

    def __set_name__(self, owner, name: str) -> None:
        """Called when the descriptor is assigned to a class attribute."""
        self.name = name
        # No need to register anywhere - introspection will find it!

    def __get__(self, instance: object, owner) -> Any:
        """Get the property value.

        Parameters
        ----------
        instance : object
            The instance of the class from which the property is accessed
        owner : type
            The class owning the property
        """
        if instance is None:
            return self
        if self.func is None:
            raise AttributeError(f"Derived property '{self.name}' has no getter")
        return self.func(instance)

    def _introspect_return_type(self, func: Callable) -> None:
        """Introspect and store the return type from function annotations.

        Parameters
        ----------
        func : Callable
            The function to introspect

        Raises
        ------
        TypeError
            If the function lacks a return type annotation or hints cannot be resolved
        """
        try:
            hints = get_type_hints(func)
            if "return" in hints:
                self.metadata["return_type"] = hints["return"]
            else:
                raise TypeError(
                    f"Derived property '{func.__name__}' must have a return type annotation"
                )
        except (NameError, AttributeError) as e:
            raise TypeError(
                f"Could not resolve type hints for derived property '{func.__name__}': {e}"
            ) from e


# Convenience alias using lowercase convention
def derived_property(*args, **kwargs):
    """Convenience function to use the DerivedProperty decorator with lowercase."""
    return DerivedProperty(*args, **kwargs)
