"""Dependency-light canonical constants for SPINE.

This package is the canonical home for fixed SPINE conventions:
- tensor column layouts
- categorical enums
- presentation/label maps derived from those enums
- campaign-independent physical constants
- sentinel and invalid values

Guiding principle
-----------------
If a value defines a stable SPINE convention, it belongs here.
If it depends on detector configuration, run conditions, or production choices,
it belongs in YAML/config instead.

Most users should import from focused submodules:

``from spine.constants.columns import ClusterLabelCol``
``from spine.constants.enums import ParticleShape, ParticlePID``
``from spine.constants.physics import LAR_DENSITY_G_CM3``

The package root re-exports the most common symbols for convenience and for
backward-compatible migration from the old ``spine.utils.globals`` API.
"""

from .columns import *
from .enums import *
from .factory import *
from .labels import *
from .physics import *
from .sentinels import *
