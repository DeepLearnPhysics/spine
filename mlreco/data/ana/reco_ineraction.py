"""Module with a data class object which represents a reconstructed fragement."""

from dataclasses import dataclass

from mlreco.utils.decorators import inherit_docstring

from .fragement_base import FragmentBase

__all__ = ['RecoFragment']


@dataclass
@inherit_docstring(FragmentBase)
class RecoFragment(FragmentBase):
    """Reconstructed fragement information.

    Parameters
    ----------
    primary_scores : np.ndarray
        (2) Array of softmax scores associated with secondary and primary
    """
    primary_scores: np.ndarray = None

    # Fixed-length attributes
    _fixed_length_attrs = {
            'primary_scores': 2, 
            **ParticleBase._fixed_length_attrs}

    def __str__(self):
        """Human-readable string representation of the fragement object.

        Results
        -------
        str
            Basic information about the fragement properties
        """
        return 'Reco' + super().__str__()
