"""Module with a data class object which represents a reconstructed fragment."""

from dataclasses import dataclass

import numpy as np

from mlreco.utils.decorators import inherit_docstring

from .fragment_base import FragmentBase

__all__ = ['RecoFragment']


@dataclass
@inherit_docstring(FragmentBase)
class RecoFragment(FragmentBase):
    """Reconstructed fragment information.

    Parameters
    ----------
    primary_scores : np.ndarray
        (2) Array of softmax scores associated with secondary and primary
    """
    primary_scores: np.ndarray = None

    # Fixed-length attributes
    _fixed_length_attrs = {
            'primary_scores': 2, 
            **FragmentBase._fixed_length_attrs}

    def __str__(self):
        """Human-readable string representation of the fragment object.

        Results
        -------
        str
            Basic information about the fragment properties
        """
        return 'Reco' + super().__str__()
