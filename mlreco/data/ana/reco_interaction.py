"""Module with a data class object which represents a reconstructed
interaction.
"""

from typing import List
from dataclasses import dataclass

import numpy as np

from mlreco.utils.globals import TRACK_SHP, SHAPE_LABELS, PID_LABELS, PID_MASSES
from mlreco.utils.numba_local import cdist

from .interaction_base import InteractionBase

__all__ = ['RecoInteraction']


@dataclass
class RecoInteraction(InteractionBase):
    """Reconstructed interaction information."""

    def __str__(self):
        """Human-readable string representation of the interaction object.

        Results
        -------
        str
            Basic information about the interaction properties
        """
        return 'Reco' + super().__str__()
