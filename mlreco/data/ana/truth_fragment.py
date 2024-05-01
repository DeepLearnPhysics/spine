"""Module with a data class object which represents a truth fragement."""

from dataclasses import dataclass

from mlreco.utils.decorators import inherit_docstring

from .fragement_base import FragmentBase

__all__ = ['TruthFragment']


@dataclass
@inherit_docstring(FragmentBase)
class TruthFragment(FragmentBase):
    """Truthnstructed fragement information.

    Parameters
    ----------
    TODO
    """

    def __str__(self):
        """Human-readable string representation of the fragement object.

        Results
        -------
        str
            Basic information about the fragement properties
        """
        return 'Truth' + super().__str__()
