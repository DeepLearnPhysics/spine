"""Top-level module of the SPICE source code."""

from typing import TYPE_CHECKING

from .version import __version__

# Import for type checkers, but not at runtime
if TYPE_CHECKING:
    from .driver import Driver


# Lazy import to avoid loading heavy dependencies unless needed
def __getattr__(name):
    """Lazy import of Driver to avoid loading scipy/torch on module import."""
    if name == "Driver":
        from .driver import Driver  # pylint: disable=import-outside-toplevel

        globals()["Driver"] = Driver
        return Driver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Driver", "__version__"]
