"""Module with a data class object which represents a truth fragment."""

from dataclasses import dataclass

from mlreco.utils.decorators import inherit_docstring

from .fragment_base import FragmentBase

__all__ = ['TruthFragment']


@dataclass
@inherit_docstring(FragmentBase)
class TruthFragment(FragmentBase):
    """Truthnstructed fragment information.

    Parameters
    ----------
    TODO
    """

    def __str__(self):
        """Human-readable string representation of the fragment object.

        Results
        -------
        str
            Basic information about the fragment properties
        """
        return 'Truth' + super().__str__()
