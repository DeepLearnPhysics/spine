"""Module that handles conditional imports for optional packages.

Currently wraps the following packages:
- larcv: only needed when reading larcv-format data in parsers
- MinkowskiEngine: only needed when running sparse CNNs
"""

import os
from warnings import warn


# If LArCV is available, load it
try:
    from larcv import larcv
except ModuleNotFoundError:
    warn("larcv could not be found, cannot parse LArCV data.")


# If MinkowskiEngine is available, load it with the right number of threads
try:
    if os.environ.get('OMP_NUM_THREADS') is None:
        os.environ['OMP_NUM_THREADS'] = '16'
    import MinkowskiEngine as ME
    import MinkowskiFunctional as MF
except ModuleNotFoundError:
    warn("MinkowskiEngine could not be found, cannot run sparse CNNs.")
