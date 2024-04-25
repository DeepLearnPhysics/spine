"""Module that handles the construction of analysis data classes.

It produces the following classes:
- :class:`RecoFragment`, :class:`TruthFragment`
- :class:`RecoParticle`, :class:`TruthParticle`
- :class:`RecoInteraction`, :class:`TruthInteraction`

It operates two basic functions:
- Build one of the aforementioned classes based on the ML chain output
- Load one of the aforementioned classes from an analysis HDF5 file
"""

from .fragment import RecoFragmentBuilder, TruthFragmentBuilder
from .particle import RecoParticleBuilder, TruthParticleBuilder
from .interaction import RecointeractionBuilder, TruthInteractionBuilder
