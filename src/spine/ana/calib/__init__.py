"""Calibration analysis scripts.

This submodule contains analysis scripts that extract calibration inputs from
reconstruction objects. Currently it includes the MCS angular resolution
calibration workflow.

Other examples of sensible future additions:
- dE/dx calibration
- Shower calorimetric energy calibration
- ...
"""

from .mcs import *
