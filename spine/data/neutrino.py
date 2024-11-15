"""Module with a data class object which represents true neutrino information.

This copies the internal structure of :class:`larcv.Neutrino`.
"""

from dataclasses import dataclass
from warnings import warn

import numpy as np

from spine.utils.globals import NU_CURR_TYPE, NU_INT_TYPE

from .base import PosDataBase

__all__ = ['Neutrino']


@dataclass(eq=False)
class Neutrino(PosDataBase):
    """Neutrino truth information.

    Attributes
    ----------
    id : int
        Index of the neutrino in the list
    interaction_id : int
        Index of the neutrino at the generator stage (e.g. Genie)
    mct_index : int
        Index in the original MCTruth array from whence it came
    track_id : int
        Geant4 track ID of the neutrino
    lepton_track_id : int
        Geant4 track ID of the lepton (if CC)
    pdg_code : int
        PDG code of the neutrino
    lepton_pdg_code : int
        PDF code of the outgoing lepton
    current_type : int
        Enumerated current type of the neutrino interaction
    interaction_mode : int
        Enumerated neutrino interaction mode
    interaction_type : int
        Enumerated neutrino interaction type
    target : int
        PDG code of the target object
    nucleon : int
        PDG code of the target nucleon (if QE)
    quark : int
        PDG code of the target quark (if DIS)
    energy_init : float
        Energy of the neutrino at its interaction point in GeV
    hadronic_invariant_mass : float
        Hadronic invariant mass (W) in GeV/c^2
    bjorken_x : float
        Bjorken scaling factor (x)
    inelasticity : float
        Inelasticity (y)
    momentum_transfer : float
        Squared momentum transfer (Q^2) in (GeV/c)^2
    momentum_transfer_mag : float
        Magnitude of the momentum transfer (Q3) in GeV/c
    energy_transfer : float
        Energy transfer (Q0) in GeV
    lepton_p : float
        Absolute momentum of the lepton
    distance_travel : float
        True amount of distance traveled by the neutrino before interacting
    theta : float
        Angle between incoming and outgoing leptons in radians
    creation_process : str
        Creation process of the neutrino
    position : np.ndarray
        Location of the neutrino interaction
    momentum : np.ndarray
        3-momentum of the neutrino at its interaction point
    units : str
        Units in which the position coordinates are expressed
    """
    # Attributes
    id: int = -1
    interaction_id: int = -1
    mct_index: int = -1
    track_id: int = -1
    lepton_track_id: int = -1
    pdg_code: int = -1
    lepton_pdg_code: int = -1
    current_type: int = -1
    interaction_mode: int = -1
    interaction_type: int = -1
    target: int = -1
    nucleon: int = -1
    quark: int = -1
    energy_init: float = -1.
    hadronic_invariant_mass: float = -1.
    bjorken_x: float = -1.
    inelasticity: float = -1.
    momentum_transfer: float = -1.
    momentum_transfer_mag: float = -1.
    energy_transfer: float = -1.
    lepton_p: float = -1.
    distance_travel: float = -1.
    theta: float = -1.
    creation_process: str = ''
    position: np.ndarray = None
    momentum: np.ndarray = None
    units: str = 'cm'

    # Fixed-length attributes
    _fixed_length_attrs = (('position', 3), ('momentum', 3))

    # Attributes specifying coordinates
    _pos_attrs = ('position',)

    # Attributes specifying vector components
    _vec_attrs = ('momentum',)

    # Enumerated attributes
    _enum_attrs = (
            ('current_type', tuple((v, k) for k, v in NU_CURR_TYPE.items())),
            ('interaction_mode', tuple((v, k) for k, v in NU_INT_TYPE.items())),
            ('interaction_type', tuple((v, k) for k, v in NU_INT_TYPE.items()))
    )

    # String attributes
    _str_attrs = ('creation_process',)

    @classmethod
    def from_larcv(cls, neutrino):
        """Builds and returns a Neutrino object from a LArCV Neutrino object.

        Parameters
        ----------
        neutrino : larcv.Neutrino
            LArCV-format neutrino object

        Returns
        -------
        Neutrino
            Neutrino object
        """
        # Initialize the dictionary to initialize the object with
        obj_dict = {}

        # Load the scalar attributes
        for key in ('id', 'interaction_id', 'mct_index', 'nu_track_id',
                    'lepton_track_id', 'pdg_code', 'lepton_pdg_code',
                    'current_type', 'interaction_mode', 'interaction_type',
                    'target', 'nucleon', 'quark', 'energy_init',
                    'hadronic_invariant_mass', 'bjorken_x', 'inelasticity',
                    'momentum_transfer', 'momentum_transfer_mag',
                    'energy_transfer', 'lepton_p', 'distance_travel',
                    'theta', 'creation_process'):
            if not hasattr(neutrino, key):
                warn(f"The LArCV Neutrino object is missing the {key} "
                      "attribute. It will miss from the Neutrino object.")
                continue
            if key != 'nu_track_id':
                obj_dict[key] = getattr(neutrino, key)()
            else:
                obj_dict['track_id'] = getattr(neutrino, key)()

        # Load the positional attribute
        pos_attrs = ['x', 'y', 'z']
        for key in cls._pos_attrs:
            vector = getattr(neutrino, key)()
            obj_dict[key] = np.asarray(
                    [getattr(vector, a)() for a in pos_attrs], dtype=np.float32)

        # Load the momentum attribute (special care needed)
        mom_attrs = ('px', 'py', 'pz')
        if not hasattr(neutrino, 'momentum'):
            warn("The LArCV Neutrino object is missing the momentum "
                 "attribute. It will miss from the Neutrino object.")
        else:
            obj_dict['momentum'] = np.asarray(
                    [getattr(neutrino, a)() for a in mom_attrs],
                    dtype=np.float32)

        return cls(**obj_dict)
