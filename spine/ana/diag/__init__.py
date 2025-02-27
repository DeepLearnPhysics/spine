'''Diagnostic analaysis scripts.

This submodule is use to run basic diagnostics analyses such as:
- Track dE/dx profile
- Track energy reconstruction
- Track completeness
- Shower start dE/dx
- Point purity and efficiency
- ...
'''

from .shower import *
from .track import *
from .point import *
