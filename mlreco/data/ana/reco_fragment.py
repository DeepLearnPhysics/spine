"""Module with a data class object which represents a reconstructed interaction."""

from dataclasses import dataclass

from mlreco.utils.decorators import inherit_docstring

from .interaction_base import InteractionBase

__all__ = ['RecoInteraction']


@dataclass
@inherit_docstring(InteractionBase)
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
