"""Top-level module of the SPICE source code."""

# Import main workflow entry point
from .driver import Driver
from .version import __version__

# Import commonly used data structures
from .data.meta import Meta
