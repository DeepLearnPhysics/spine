"""Diagnostic analaysis scripts.

This submodule is used to run basic diagnostics analyses, such as:
- Track energy reconstruction
- Track completeness
- Shower start dE/dx
- Point purity and efficiency
- ...
"""

from .shower import *
from .track import *
from .point import *
